"""
HTTP API — the integration surface for a separate UI/frontend team.

Wraps the LangGraph pipeline (graph.py) behind a plain JSON contract so a UI
team can build against this backend over HTTP without importing Python or
knowing anything about LangGraph, agents, or CaseState internals. Audio/image
travel as base64 strings (the only two fields in CaseState that aren't
natively JSON-safe); every other field is already plain str/int/float/bool/
list/dict and is returned as-is.

Run:
    pip install fastapi uvicorn
    uvicorn justicebridge.api:app --host 0.0.0.0 --port 8080

Endpoints:
    GET  /health      -> which backends are configured/live (for a status bar)
    GET  /kb-stores    -> the legal-topic catalogue (for a picker/menu)
    POST /ask          -> run one query through the full pipeline

This file is intentionally separate from app.py (the Streamlit demo UI) — the
API is the stable contract other teams build against; the Streamlit app is
just one consumer of the same graph.
"""

import base64
import io

from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

from .graph import get_app
from . import config, llm
from .kb_registry import KB_STORES, STUB_VERTICALS

app = FastAPI(title="JusticeBridge API", version="0.1.0")


class AskRequest(BaseModel):
    text_input: str | None = None
    audio_base64: str | None = None     # WAV bytes, base64-encoded
    image_base64: str | None = None     # single document, base64 (back-compat)
    images_base64: list[str] | None = None  # multiple documents, base64 each
    lang: str = "en"                    # en | ta | hi | te
    want_tts: bool = False              # force spoken-answer synthesis

    # Every input is OPTIONAL — text_input, audio_base64, image(s)_base64 can
    # be given in any combination, as long as at least one is present. Voice
    # and document are never both required.


# Fields returned to the caller — a deliberate ALLOWLIST, not "dump all of
# state". Keeps the HTTP contract stable even as internal CaseState fields
# change; anything new added to the graph doesn't leak out until it's
# reviewed and added here.
_RESPONSE_FIELDS = [
    "transcript", "asr_confidence", "doc_text", "vision_confidence",
    "vertical", "kb_stores", "supported", "planner_backend",
    "retrieval_sim", "citations",
    "reasoning_backend", "grounded", "ungrounded_claims",
    "severity", "deadline_days", "deadline_basis", "composite_confidence",
    "escalate", "eligibility_reasons", "dlsa_contact",
    "final_answer_en", "final_answer_local", "signal_packet", "lang",
    "error",
]


@app.get("/health")
def health():
    """Status of every backend — for a UI status bar / debug panel."""
    return {
        "llm_backend": config.LLM_BACKEND,
        "llm_live": llm.is_live(),
        "asr_backend": config.ASR_BACKEND,
        "vision_backend": config.VISION_BACKEND,
        "tts_backend": config.TTS_BACKEND,
        "sarvam_key_configured": bool(config.SARVAM_API_KEY),
    }


@app.get("/kb-stores")
def kb_stores():
    """The legal-topic catalogue — lets a UI show a "what can I ask about"
    menu without hard-coding the vertical list on the frontend."""
    supported = {
        sid: {"topic": cfg["topic"], "description": cfg["description"],
              "cross_cutting": bool(cfg.get("always_include"))}
        for sid, cfg in KB_STORES.items()
    }
    coming_soon = {sid: {"topic": cfg["topic"]} for sid, cfg in STUB_VERTICALS.items()}
    return {"supported": supported, "coming_soon": coming_soon}


@app.post("/ask")
def ask(req: AskRequest):
    """Run one query through the full pipeline. text_input/audio/image(s) are
    all optional — give any combination, at least one. Returns the
    allowlisted result fields; `audio_response_base64` is included only if
    TTS ran."""
    init = {"lang": req.lang, "want_tts": req.want_tts}
    if req.text_input:
        init["text_input"] = req.text_input
    if req.audio_base64:
        init["audio_bytes"] = base64.b64decode(req.audio_base64)

    image_b64_list = list(req.images_base64 or [])
    if req.image_base64:
        image_b64_list.append(req.image_base64)
    if image_b64_list:
        init["images"] = [Image.open(io.BytesIO(base64.b64decode(b))) for b in image_b64_list]

    if not init.get("text_input") and not init.get("audio_bytes") and not init.get("images"):
        return {"error": ["Provide at least one of: text_input, audio_base64, image_base64/images_base64"]}

    state = get_app().invoke(init)

    out = {k: state.get(k) for k in _RESPONSE_FIELDS}
    if state.get("audio_response"):
        out["audio_response_base64"] = base64.b64encode(state["audio_response"]).decode("ascii")
    return out
