"""
TTS agent — final guidance text -> spoken audio (WAV bytes), fully offline.

config.TTS_BACKEND:
  - "mms"     : Meta's MMS-TTS (facebook/mms-tts-<lang>), a real neural
                on-device TTS model with dedicated checkpoints per language
                (English/Hindi/Tamil/Telugu). Default. This is the ONLY
                backend that can correctly speak the translated (Hindi/Tamil/
                Telugu) answer: pyttsx3 just wraps whatever OS voices happen
                to be installed, and a plain Windows install commonly ships
                only English (+ maybe Chinese) SAPI5 voices — verified on
                this machine (`pyttsx3.init().getProperty('voices')` returned
                only en-GB/en-US/zh-CN/zh-TW). Without a matching voice,
                pyttsx3 silently mispronounces/mangles non-Latin-script text
                with the default English voice rather than failing loudly.
  - "pyttsx3" : offline OS TTS (SAPI5/NSSpeech/espeak). English-only in
                practice unless the OS has other language voices installed.
  - "none"    : skip spoken output.

This is exposed as speak_response(text, lang) -> bytes for the Streamlit app
to play, and as an optional graph node (tts_node) that writes WAV bytes into
state["audio_response"] after the answer is assembled.
"""

import io
import os
import tempfile

from ..state import CaseState
from .. import config

_MMS_LANG_MODELS = {
    "en": "facebook/mms-tts-eng",
    "hi": "facebook/mms-tts-hin",
    "ta": "facebook/mms-tts-tam",
    "te": "facebook/mms-tts-tel",
}
_mms_models = {}  # lang -> (tokenizer, model), lazy-loaded per language


def _get_mms(lang: str):
    if lang not in _mms_models:
        from transformers import VitsModel, AutoTokenizer

        name = _MMS_LANG_MODELS[lang]
        tok = AutoTokenizer.from_pretrained(name)
        model = VitsModel.from_pretrained(name)
        model.eval()
        _mms_models[lang] = (tok, model)
    return _mms_models[lang]


def _mms_tts(text: str, lang: str) -> bytes:
    lang = lang if lang in _MMS_LANG_MODELS else "en"
    tok, model = _get_mms(lang)

    import torch
    import soundfile as sf

    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        waveform = model(**inputs).waveform[0].numpy()

    buf = io.BytesIO()
    sf.write(buf, waveform, model.config.sampling_rate, format="WAV")
    return buf.getvalue()


def _pyttsx3_tts(text: str) -> bytes:
    import pyttsx3
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        engine = pyttsx3.init()
        engine.save_to_file(text, tmp)
        engine.runAndWait()
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def speak_response(text: str, lang: str = "en") -> bytes:
    """Synthesize speech; returns WAV bytes. `lang` should be a short code
    (en/hi/ta/te) — the MMS backend picks the matching neural voice."""
    if config.TTS_BACKEND == "none" or not text.strip():
        raise RuntimeError("TTS disabled or empty text")
    if config.TTS_BACKEND == "mms":
        return _mms_tts(text, lang)
    return _pyttsx3_tts(text)


def tts_node(state: CaseState) -> dict:
    """Optional graph node: synthesize the final (local-language) answer to
    WAV bytes. Never blocks the pipeline — records a note on failure.

    Only speaks for VOICE interactions (audio input given) or when the caller
    explicitly sets want_tts — so text-only runs (eval, typed CLI) don't
    trigger synthesis. Voice in -> voice out."""
    if config.TTS_BACKEND == "none":
        return {}
    if not state.get("audio_bytes") and not state.get("want_tts"):
        return {}
    text = state.get("final_answer_local") or state.get("final_answer_en") or ""
    if not text.strip():
        return {}
    lang = state.get("lang") or "en"
    try:
        audio = speak_response(text, lang)
        return {"audio_response": audio}
    except Exception as e:
        return {"error": [f"TTS skipped: {e}"]}
