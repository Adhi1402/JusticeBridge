"""
TTS agent — final guidance text -> spoken audio (WAV bytes).

Two interchangeable backends, same graceful-degradation contract as ASR/OCR:
  - Sarvam Bulbul v3 (cloud): speaks the answer back in the language the ASR
    agent detected, so a Tamil/Hindi speaker hears the answer in their own
    language. Verified response shape: `response.audios` is a LIST of
    base64-encoded WAV strings — decode audios[0].
  - pyttsx3 (offline OS voices): fallback so the kiosk still speaks with no
    internet. English-only in practice, but keeps the voice-first UX alive.

This is exposed as speak_response(text, lang) -> bytes for the Streamlit app
to play, and as an optional graph node (tts_node) that writes WAV bytes into
state["audio_response"] after the answer is assembled.
"""

import base64
import io
import os
import tempfile

from ..state import CaseState
from .. import config


def _sarvam_tts(text: str, lang: str) -> bytes:
    from sarvamai import SarvamAI
    if not config.SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY not set")
    client = SarvamAI(api_subscription_key=config.SARVAM_API_KEY)
    resp = client.text_to_speech.convert(
        text=text,
        target_language_code=lang or "en-IN",
        model=config.SARVAM_TTS_MODEL,
        speaker=config.SARVAM_TTS_SPEAKER,
    )
    audios = getattr(resp, "audios", None)
    if not audios:
        raise RuntimeError(f"Sarvam TTS returned no audio: {resp!r}")
    # response.audios is a list of base64-encoded WAV strings (verified).
    return base64.b64decode(audios[0])


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


def speak_response(text: str, lang: str = "en-IN") -> bytes:
    """Synthesize speech; returns WAV bytes. Tries the configured TTS backend,
    falls back to offline pyttsx3. Raises only if BOTH fail."""
    if config.TTS_BACKEND == "none" or not text.strip():
        raise RuntimeError("TTS disabled or empty text")

    if config.TTS_BACKEND == "sarvam":
        try:
            return _sarvam_tts(text, lang)
        except Exception:
            pass  # fall through to offline
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
    lang = state.get("lang") or "en-IN"
    # Sarvam wants a BCP-47 code (en-IN, ta-IN, ...); map bare Whisper codes.
    if lang in ("en", "ta", "hi", "te"):
        lang = {"en": "en-IN", "ta": "ta-IN", "hi": "hi-IN", "te": "te-IN"}[lang]
    try:
        audio = speak_response(text, lang)
        return {"audio_response": audio}
    except Exception as e:
        return {"error": [f"TTS skipped: {e}"]}
