"""
Translation agent — renders the assembled English answer into the citizen's
language (Tamil/Hindi/Telugu) for TTS on the phone.

Production path: IndicTrans2 (per the arch doc). It's a ~1GB model, so it's
imported lazily and only used if installed. If it's not available (or lang is
English), we pass the English text through unchanged and tag it — the pipeline
still completes end-to-end. Swapping in IndicTrans2 later changes nothing
upstream.
"""

from ..state import CaseState

_translator = None
_INDIC_CODES = {"ta": "tam_Taml", "hi": "hin_Deva", "te": "tel_Telu"}


def _get_translator():
    global _translator
    if _translator is None:
        from IndicTransToolkit import IndicProcessor  # type: ignore
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        name = "ai4bharat/indictrans2-en-indic-dist-200M"
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True)
        _translator = (tok, model, IndicProcessor(inference=True))
    return _translator


def _translate(text, lang):
    tgt = _INDIC_CODES.get(lang)
    if not tgt:
        return None
    tok, model, ip = _get_translator()
    batch = ip.preprocess_batch([text], src_lang="eng_Latn", tgt_lang=tgt)
    enc = tok(batch, return_tensors="pt", padding=True, truncation=True)
    out = model.generate(**enc, max_length=512, num_beams=5)
    dec = tok.batch_decode(out, skip_special_tokens=True)
    return ip.postprocess_batch(dec, lang=tgt)[0]


def translation_agent(state: CaseState) -> dict:
    answer_en = state.get("final_answer_en", "") or ""
    lang = state.get("lang", "en") or "en"

    if lang in ("en", "unknown") or not answer_en:
        local = answer_en
    else:
        try:
            local = _translate(answer_en, lang) or answer_en
        except Exception as e:
            # IndicTrans2 not installed / failed — degrade to English, don't crash.
            local = answer_en
            state.setdefault("error", [])
            return {"final_answer_local": local, "error": [f"Translation skipped: {e}"]}

    phone_message = dict(state.get("phone_message") or {})
    phone_message["answer_local"] = local

    return {"final_answer_local": local, "phone_message": phone_message}
