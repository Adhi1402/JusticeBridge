"""
Input agents: ASR (speech->text) and Vision (document image->text), plus a
`combine` node that merges them into one `combined_text` for the Planner.

Both are fully offline/on-device:
  - ASR wraps Whisper (transformers pipeline) — speech->text, auto language
    detection left to the model.
  - Vision/OCR is exposed as a LangChain @tool-wrapped function
    (tesseract_ocr_tool) so it's a real, independently invocable tool, not
    just inline logic.

ASR and Vision both start from START and run in parallel, so each returns
ONLY the keys it changes (per state.py's concurrent-update rule).
"""

import tempfile
import os

from langchain_core.tools import tool

from ..state import CaseState
from .. import config


# ---------------------------------------------------------------------------
# ASR — Whisper, fully on-device/offline
# ---------------------------------------------------------------------------
_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        # transformers' Whisper pipeline, not faster-whisper: faster-whisper
        # needs ctranslate2, which ships no Windows-ARM64 wheel (and
        # openai-whisper pulls in tiktoken/numba/llvmlite, none of which do
        # either) — this stack (transformers + torch, already required deps)
        # is the one that actually installs on this platform.
        from transformers import pipeline
        model_id = config.WHISPER_MODEL
        if "/" not in model_id:
            model_id = f"openai/whisper-{model_id}"
        _whisper_model = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=config.WHISPER_DEVICE,
        )
    return _whisper_model


@tool
def whisper_stt_tool(audio_path: str) -> dict:
    """Transcribe a WAV file with a local Whisper model via transformers
    (fully on-device/offline). Returns {text, lang, confidence}."""
    import soundfile as sf
    from scipy.signal import resample

    asr = _get_whisper()
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = resample(audio, int(len(audio) * 16000 / sr))
        sr = 16000

    result = asr({"array": audio, "sampling_rate": sr})
    text = result["text"].strip()
    # transformers' pipeline doesn't surface a scalar confidence/detected-
    # language for the base ASR call — fixed placeholder, not a real score.
    return {"text": text, "lang": "unknown", "confidence": 0.8}


def asr_agent(state: CaseState) -> dict:
    """Transcribe recorded audio via Whisper (fully offline). If there's no
    audio but there IS text_input (dev/kiosk-typed/eval path), pass it
    straight through."""
    audio = state.get("audio_bytes")
    if not audio:
        text = state.get("text_input", "") or ""
        if text:
            return {"transcript": text, "asr_confidence": 1.0}
        return {}

    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio)
            tmp = f.name

        r = whisper_stt_tool.invoke({"audio_path": tmp})
        out = {"transcript": r["text"], "asr_confidence": r["confidence"]}
        if r["lang"] != "unknown":
            out["lang"] = r["lang"]
        return out
    except Exception as e:
        return {"error": [f"ASR error: {e}"]}
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Vision / OCR — Tesseract, fully offline
# ---------------------------------------------------------------------------
@tool
def tesseract_ocr_tool(image_path: str) -> dict:
    """Extract text from a document image using offline Tesseract OCR.
    Returns {text, confidence}."""
    import pytesseract
    from PIL import Image

    default_cmd = config.TESSERACT_CMD
    if os.path.exists(default_cmd):
        pytesseract.pytesseract.tesseract_cmd = default_cmd

    img = Image.open(image_path)
    text = pytesseract.image_to_string(img).strip()
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    confs = [int(c) for c in data["conf"] if int(c) >= 0]
    confidence = (sum(confs) / len(confs) / 100) if confs else 0.0

    return {"text": text, "confidence": confidence}


def _ocr_one(image) -> tuple[str, float, list[str]]:
    """Run Tesseract OCR on ONE PIL image. Returns (text, confidence, errors)
    — never raises; a fully-failed document comes back as ("", 0.0, [reasons])
    so one bad upload can't sink the others."""
    tmp = None
    errors = []
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f, format="PNG")
            tmp = f.name

        r = tesseract_ocr_tool.invoke({"image_path": tmp})
        return r["text"], r["confidence"], errors
    except Exception as e:
        errors.append(f"OCR failed for this document: {e}")
        return "", 0.0, errors
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def vision_agent(state: CaseState) -> dict:
    """OCR one or more uploaded document images (supplementary — low
    confidence is fine, and voice/document input are BOTH optional; this node
    is a no-op if neither `images` nor `image` is set).

    Accepts `state["images"]` (a list of PIL Images — the multi-document
    path) OR the singular `state["image"]` (back-compat, single document).
    Every document is OCR'd independently — one failed page doesn't block the
    rest — and results are concatenated, labeled by document number, with the
    average confidence across all pages that returned text."""
    docs = state.get("images") or ([state["image"]] if state.get("image") else [])
    if not docs:
        return {}

    texts, confidences, all_errors = [], [], []
    for i, img in enumerate(docs, start=1):
        text, conf, errors = _ocr_one(img)
        label = f"--- Document {i} ---\n{text}" if len(docs) > 1 else text
        if text:
            texts.append(label)
            confidences.append(conf)
        all_errors.extend(f"Document {i}: {e}" for e in errors)

    out = {
        "doc_text": "\n\n".join(texts).strip(),
        "vision_confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
    }
    if all_errors:
        out["error"] = all_errors
    return out


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------
def combine_node(state: CaseState) -> dict:
    transcript = state.get("transcript", "") or ""
    doc_text = state.get("doc_text", "") or ""
    return {"combined_text": (transcript + " " + doc_text).strip()}
