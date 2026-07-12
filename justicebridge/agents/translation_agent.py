"""
Translation agent — renders the assembled English answer into the citizen's
language (Tamil/Hindi/Telugu) for TTS on the phone.

config.TRANSLATION_BACKEND:
  - "nllb"  : facebook/nllb-200-distilled-600M — a real ON-DEVICE machine
              -translation model (not an LLM), covering 200 languages
              including Hindi/Tamil/Telugu, via transformers. Lazy-loaded;
              once weights are cached, translation runs fully offline/
              locally. Default backend.

              NOTE: ai4bharat/indictrans2-en-indic-dist-200M (India-specific,
              in principle higher quality for Indian languages) was tried
              first but is a GATED HuggingFace repo — it 401s on download
              without a manually-approved HF access request, which defeats
              "clone and run offline" for anyone who hasn't gone through
              that approval. NLLB is public/ungated and needs no HF account,
              so it's the one that actually works out of the box. Swap back
              to indictrans2 (config.INDICTRANS2_MODEL) once you have access
              approved on your HF account, if you want the quality bump.
  - "none"  : skip translation entirely, English passthrough.

Any failure degrades to English passthrough, never crashes.

Why this is a SEPARATE agent from TTS (agents/tts_agent.py), not merged:
Translation is text -> text (machine translation, a very different model
class — MT encoder-decoder, not a chat LLM); TTS is text -> audio (speech
synthesis). They compose in a pipeline (translate, THEN speak the translated
text) rather than being two views of the same operation. Keeping them
separate agents also means each can independently degrade (e.g. translation
fails -> English text still gets spoken by TTS) instead of one failure
silently killing both.
"""

from ..state import CaseState
from .. import config

_nllb_model = None
_INDIC_CODES = {
    "english": "eng_Latn",
    "en": "eng_Latn",
    "hindi": "hin_Deva",
    "hi": "hin_Deva",
    "tamil": "tam_Taml",
    "ta": "tam_Taml",
    "telugu": "tel_Telu",
    "te": "tel_Telu",
    "kannada": "kan_Knda",
    "malayalam": "mal_Mlym",
    "marathi": "mar_Deva",
    "gujarati": "guj_Gujr",
    "bengali": "ben_Beng",
    "odia": "ory_Orya",
    "punjabi": "pan_Guru",
    "assamese": "asm_Beng",
}


def _get_nllb():
    global _nllb_model
    if _nllb_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        name = config.NLLB_MODEL  # e.g. "facebook/nllb-200-distilled-600M"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
        model.eval()
        _nllb_model = (tok, model)
    return _nllb_model


def _translate_nllb(text, lang):
    tgt = _INDIC_CODES.get(lang.lower())
    if tgt is None or tgt == "eng_Latn":
        return None

    tok, model = _get_nllb()
    tok.src_lang = "eng_Latn"

    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    if not sentences:
        return None
    inputs = tok(sentences, truncation=True, padding="longest", return_tensors="pt")

    import torch
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
            use_cache=True, min_length=0, max_length=256,
            num_beams=config.NLLB_NUM_BEAMS,
        )
    translated = tok.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return ". ".join(translated)


def _translate(text, lang):
    """Try the configured on-device backend. Returns None if unavailable or
    it fails — caller passes through English. Never raises."""
    if config.TRANSLATION_BACKEND != "nllb":
        return None  # "none" or unrecognised
    try:
        return _translate_nllb(text, lang)
    except Exception:
        return None


def translation_agent(state: CaseState) -> dict:
    answer_en = state.get("final_answer_en", "")
    lang = state.get("lang", "en")

    if lang in ("en", "unknown") or not answer_en or config.TRANSLATION_BACKEND == "none":
        local = answer_en
        out = {"final_answer_local": local}
    else:
        translated = _translate(answer_en, lang)
        local = translated or answer_en
        out = {"final_answer_local": local}
        if translated is None:
            out["error"] = [f"Translation unavailable for '{lang}' (NLLB failed); using English"]

    phone_message = dict(state.get("phone_message") or {})
    phone_message["answer_local"] = local
    out["phone_message"] = phone_message
    return out
