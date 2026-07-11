"""
Input agents: ASR (speech->text) and Vision (document image->text), plus a
`combine` node that merges them into one `combined_text` for the Planner.

Vision/OCR is exposed as two LangChain @tool-wrapped functions so they're
real, independently invocable tools, not just inline logic:
  - sarvam_ocr_tool   : Sarvam AI Document Intelligence — cloud OCR tuned for
                        Indian-language documents (23 languages), per the
                        Qualcomm-hackathon plan to use Sarvam for OCR.
  - tesseract_ocr_tool: fully offline OCR fallback.

vision_agent() tries Sarvam first (config.VISION_BACKEND == "sarvam" and an
API key is set), then falls back to Tesseract on any failure — same
graceful-degradation pattern as the LLM backend in llm.py, so a kiosk with no
internet still extracts document text locally.

ASR wraps Whisper (on-device, per the arch doc's phone tier). ASR and Vision
both start from START and run in parallel, so each returns ONLY the keys it
changes (per state.py's concurrent-update rule).
"""

import tempfile
import os
import zipfile

from langchain_core.tools import tool

from ..state import CaseState
from .. import config


# ---------------------------------------------------------------------------
# ASR — two interchangeable backends (Sarvam Saaras v3 cloud / Whisper on-device)
# ---------------------------------------------------------------------------
_whisper_model = None
_sarvam_client = None


def _get_sarvam_client():
    global _sarvam_client
    if _sarvam_client is None:
        if not config.SARVAM_API_KEY:
            raise RuntimeError("SARVAM_API_KEY not set")
        from sarvamai import SarvamAI
        _sarvam_client = SarvamAI(api_subscription_key=config.SARVAM_API_KEY)
    return _sarvam_client


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
    return _whisper_model


@tool
def sarvam_stt_tool(audio_path: str) -> dict:
    """Transcribe a WAV file with Sarvam Saaras v3 (cloud, 23 Indian languages,
    auto language detection). Returns {text, lang, confidence}. NOTE: file= must
    be an OPEN BINARY FILE OBJECT, not a path string (verified)."""
    client = _get_sarvam_client()
    with open(audio_path, "rb") as audio_file:
        resp = client.speech_to_text.transcribe(
            file=audio_file,
            model=config.SARVAM_STT_MODEL,
            language_code="unknown",   # auto-detect Hindi/Tamil/Telugu/etc.
            mode="transcribe",
        )
    lang = getattr(resp, "language_code", None) or "en-IN"
    # Saaras v3 has no scalar confidence field — use a fixed high placeholder,
    # same limitation the reference implementation documented.
    return {"text": resp.transcript.strip(), "lang": lang, "confidence": 0.9}


@tool
def whisper_stt_tool(audio_path: str) -> dict:
    """Transcribe a WAV file with faster-whisper (fully on-device/offline).
    Returns {text, lang, confidence}."""
    model = _get_whisper()
    segments, info = model.transcribe(audio_path, language=None)
    text = " ".join(seg.text for seg in segments).strip()
    conf = float(getattr(info, "language_probability", 0.8))
    lang = getattr(info, "language", None) or "unknown"
    return {"text": text, "lang": lang, "confidence": conf}


def asr_agent(state: CaseState) -> dict:
    """Transcribe recorded audio via the configured ASR backend, falling back
    to Whisper (offline) if Sarvam is unavailable. If there's no audio but
    there IS text_input (dev/kiosk-typed/eval path), pass it straight through."""
    audio = state.get("audio_bytes")
    if not audio:
        text = state.get("text_input", "") or ""
        if text:
            return {"transcript": text, "asr_confidence": 1.0}
        return {}

    tmp = None
    errors = []
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio)
            tmp = f.name

        if config.ASR_BACKEND == "sarvam":
            try:
                r = sarvam_stt_tool.invoke({"audio_path": tmp})
                return {"transcript": r["text"], "lang": r["lang"], "asr_confidence": r["confidence"]}
            except Exception as e:
                errors.append(f"Sarvam STT unavailable, falling back to Whisper: {e}")

        r = whisper_stt_tool.invoke({"audio_path": tmp})
        out = {"transcript": r["text"], "asr_confidence": r["confidence"]}
        if r["lang"] != "unknown":
            out["lang"] = r["lang"]
        if errors:
            out["error"] = errors
        return out
    except Exception as e:
        errors.append(f"ASR error: {e}")
        return {"error": errors}
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Vision / OCR tools  (Sarvam client is shared with ASR — defined above)
# ---------------------------------------------------------------------------
@tool
def sarvam_ocr_tool(image_path: str) -> dict:
    """Extract text from a document image using Sarvam AI Document Intelligence
    (cloud OCR specialised for Indian-language documents). Requires network +
    SARVAM_API_KEY. Returns {text, confidence}."""
    client = _get_sarvam_client()

    # Real, verified flow against sarvamai==0.1.28's DocumentIntelligenceJob:
    # create_job -> upload_file -> start -> wait_until_complete -> download_output.
    job = client.document_intelligence.create_job(
        language=config.SARVAM_OCR_LANGUAGE, output_format="md"
    )
    job.upload_file(image_path)
    job.start()
    job.wait_until_complete(poll_interval=2.0, timeout=config.LLM_TIMEOUT)

    # download_output() writes a ZIP archive (verified: magic bytes PK\x03\x04),
    # not a raw markdown file — contains document.md (+ metadata/page_N.json
    # per page). Extract and concatenate all .md entries.
    out_path = image_path + ".sarvam.zip"
    job.download_output(out_path)
    with zipfile.ZipFile(out_path) as z:
        md_names = sorted(n for n in z.namelist() if n.endswith(".md"))
        text = "\n\n".join(z.read(n).decode("utf-8") for n in md_names).strip()
    os.unlink(out_path)

    # Sarvam's API doesn't expose a per-word confidence score (unlike
    # Tesseract) — only page-level success/failure counts. We derive an
    # honest proxy from that rather than inventing a number: 1.0 if every
    # page succeeded, scaled down by the failure fraction otherwise.
    metrics = job.get_page_metrics() or {}
    total = metrics.get("total_pages") or 1
    succeeded = metrics.get("pages_succeeded", total)
    confidence = succeeded / total if total else 0.0

    return {"text": text, "confidence": confidence}


@tool
def tesseract_ocr_tool(image_path: str) -> dict:
    """Extract text from a document image using offline Tesseract OCR (fallback
    when Sarvam is unavailable). Returns {text, confidence}."""
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


def vision_agent(state: CaseState) -> dict:
    """OCR an uploaded document image (supplementary — low confidence is fine).
    Tries Sarvam first if configured, falls back to Tesseract on any failure."""
    image = state.get("image")
    if not image:
        return {}

    tmp = None
    errors = []
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f, format="PNG")
            tmp = f.name

        if config.VISION_BACKEND == "sarvam":
            try:
                result = sarvam_ocr_tool.invoke({"image_path": tmp})
                return {"doc_text": result["text"], "vision_confidence": result["confidence"]}
            except Exception as e:
                errors.append(f"Sarvam OCR unavailable, falling back to Tesseract: {e}")

        result = tesseract_ocr_tool.invoke({"image_path": tmp})
        out = {"doc_text": result["text"], "vision_confidence": result["confidence"]}
        if errors:
            out["error"] = errors
        return out
    except Exception as e:
        errors.append(f"Vision error: {e}")
        return {"error": errors}
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------
def combine_node(state: CaseState) -> dict:
    transcript = state.get("transcript", "") or ""
    doc_text = state.get("doc_text", "") or ""
    return {"combined_text": (transcript + " " + doc_text).strip()}
