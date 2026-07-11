"""
Translation agent — renders the assembled English answer into the citizen's
language (Tamil/Hindi/Telugu) for TTS on the phone.

Two interchangeable backends (config.TRANSLATION_BACKEND), same
graceful-degradation contract as every other backend in this codebase:
  - "sarvam"      : Sarvam text.translate (cloud; verified live —
                     model=sarvam-translate:v1, response.translated_text).
                     Same network/SARVAM_API_KEY dependency as ASR/OCR/TTS.
  - "indictrans2" : ai4bharat/indictrans2-en-indic-dist-200M — a real
                     ON-DEVICE machine-translation model (~200M params, not
                     an LLM). Lazy-loaded via transformers; once its weights
                     are cached, translation runs fully offline/locally. This
                     is the model to point at for a genuinely offline
                     deployment (matches the arch doc's on-device requirement).
  - "none"        : skip translation entirely.

Fallback chain: sarvam -> indictrans2 -> English passthrough. Any failure at
any step degrades to the next, never crashes — English-only output is always
a safe, complete answer, just not localized.

Why this is a SEPARATE agent from TTS (agents/tts_agent.py), not merged:
Translation is text -> text (machine translation, a very different model
class — MT encoder-decoder, not a chat LLM); TTS is text -> audio (speech
synthesis). They use different models, different providers even independently
of each other (e.g. you could translate with IndicTrans2 but still speak with
Sarvam Bulbul), and they compose in a pipeline (translate, THEN speak the
translated text) rather than being two views of the same operation. Keeping
them separate agents also means each can independently degrade (e.g.
translation fails -> English text still gets spoken by TTS) instead of one
failure silently killing both.
"""

from ..state import CaseState
from .. import config

_indictrans2 = None
_INDIC_CODES = {"ta": "tam_Taml", "hi": "hin_Deva", "te": "tel_Telu"}       # IndicTrans2 codes
_SARVAM_CODES = {"ta": "ta-IN", "hi": "hi-IN", "te": "te-IN", "en": "en-IN"}  # Sarvam BCP-47 codes


def _get_indictrans2():
    global _indictrans2
    if _indictrans2 is None:
        from IndicTransToolkit import IndicProcessor  # type: ignore
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        name = config.INDICTRANS2_MODEL
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True)
        _indictrans2 = (tok, model, IndicProcessor(inference=True))
    return _indictrans2


def _translate_indictrans2(text, lang):
    tgt = _INDIC_CODES.get(lang)
    if not tgt:
        return None
    tok, model, ip = _get_indictrans2()
    batch = ip.preprocess_batch([text], src_lang="eng_Latn", tgt_lang=tgt)
    enc = tok(batch, return_tensors="pt", padding=True, truncation=True)
    out = model.generate(**enc, max_length=512, num_beams=5)
    dec = tok.batch_decode(out, skip_special_tokens=True)
    return ip.postprocess_batch(dec, lang=tgt)[0]


SARVAM_TRANSLATE_MAX_CHARS = 2000  # verified live: "String should have at most 2000 characters"


def _chunk_text(text, max_len):
    """Split on sentence boundaries into chunks under max_len chars — the
    JusticeBridge answer (rights + aid pitch + deadline + DLSA + disclaimer)
    routinely runs 2000-2500 chars, over Sarvam's per-call limit, so a single
    long answer must be translated in pieces and rejoined."""
    sentences = text.replace("\n", " ").split(". ")
    chunks, current = [], ""
    for i, s in enumerate(sentences):
        piece = s if i == len(sentences) - 1 else s + ". "
        if current and len(current) + len(piece) > max_len:
            chunks.append(current)
            current = piece
        else:
            current += piece
    if current:
        chunks.append(current)
    return chunks


def _translate_sarvam(text, lang):
    tgt = _SARVAM_CODES.get(lang)
    if not tgt or not config.SARVAM_API_KEY:
        return None
    from sarvamai import SarvamAI
    client = SarvamAI(api_subscription_key=config.SARVAM_API_KEY)

    chunks = _chunk_text(text, SARVAM_TRANSLATE_MAX_CHARS)
    translated_parts = []
    for chunk in chunks:
        resp = client.text.translate(
            input=chunk,
            source_language_code="en-IN",
            target_language_code=tgt,
            model=config.SARVAM_TRANSLATE_MODEL,
        )
        translated_parts.append(resp.translated_text)
    return " ".join(translated_parts)


def _translate(text, lang):
    """Try the configured backend, then the other on-device/cloud option,
    then give up (caller passes through English). Returns None if nothing
    worked, never raises."""
    order = []
    if config.TRANSLATION_BACKEND == "sarvam":
        order = [_translate_sarvam, _translate_indictrans2]
    elif config.TRANSLATION_BACKEND == "indictrans2":
        order = [_translate_indictrans2, _translate_sarvam]
    else:
        return None  # "none" or unrecognised

    for fn in order:
        try:
            result = fn(text, lang)
            if result:
                return result
        except Exception:
            continue
    return None


def translation_agent(state: CaseState) -> dict:
    answer_en = state.get("final_answer_en", "") or ""
    lang = state.get("lang", "en") or "en"

    if lang in ("en", "unknown") or not answer_en or config.TRANSLATION_BACKEND == "none":
        local = answer_en
        out = {"final_answer_local": local}
    else:
        translated = _translate(answer_en, lang)
        local = translated or answer_en
        out = {"final_answer_local": local}
        if translated is None:
            out["error"] = [f"Translation unavailable for '{lang}' (both backends failed); using English"]

    phone_message = dict(state.get("phone_message") or {})
    phone_message["answer_local"] = local
    out["phone_message"] = phone_message
    return out
