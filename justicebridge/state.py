"""
CaseState — the single shared state object passed between every LangGraph
node. This is the full pipeline state from Section 2 of the architecture
doc, extending the original agents/state.py (which only covered the
ASR+Vision+Planner handoff).

Design rule (carried over from the original state.py): parallel nodes must
return ONLY the keys they change (partial dicts), never the whole state.
`error` uses an additive reducer so ASR + Vision can both append in the same
step without tripping LangGraph's concurrent-update check.
"""

from typing import TypedDict, Optional, Any, Annotated
import operator


class Section(TypedDict, total=False):
    act: str
    section_no: str
    title: str
    text: str
    score: float


class Citation(TypedDict, total=False):
    act: str
    section_no: str
    title: str


class CaseState(TypedDict, total=False):
    # ---- raw inputs (from phone / kiosk) ----
    audio_bytes: Optional[bytes]
    image: Optional[Any]              # PIL Image
    text_input: str                  # typed/seed query (dev + fallback path)
    lang: str                        # ta | hi | te | en
    want_tts: bool                   # force TTS even without audio input

    # ---- ASR + Vision output ----
    transcript: str
    asr_confidence: float
    doc_text: str
    vision_confidence: float
    combined_text: str
    error: Annotated[list[str], operator.add]

    # ---- Planner / routing ----
    vertical: Optional[str]          # primary KB store id, e.g. "wages"
    supported: bool
    kb_stores: list[str]             # which KB stores the Retrieval agent searches
    corpus_subset: list[str]         # alias of kb_stores (back-compat)
    output_template: Optional[str]
    planner_backend: str             # "llm" | "keyword" | "unavailable"
    off_topic: bool                  # True: no legal signal matched anything

    # ---- Retrieval ----
    retrieved_sections: list[Section]
    retrieval_sim: float
    retry_count: int

    # ---- Reasoning ----
    draft_answer: str
    draft_claims: list[dict]         # [{"claim": str, "section_no": str}]
    citations: list[Citation]
    insufficient_context: bool
    reasoning_backend: str           # which LLM backend actually produced the draft

    # ---- Grounding verification ----
    grounded: bool
    ungrounded_claims: list[str]
    grounding_retries: int
    needs_redraft: bool              # explicit router flag: loop back to Reasoning

    # ---- Risk / deadline ----
    deadline_days: Optional[int]
    deadline_basis: Optional[str]    # human-readable statute reference for the clock
    severity: str                    # red | amber | green
    composite_confidence: float

    # ---- Escalation / aid ----
    escalate: bool
    eligibility_reasons: list[str]   # why the user likely qualifies (Section 12)
    dlsa_contact: Optional[dict]

    # ---- Output ----
    final_answer_en: str
    final_answer_local: str
    signal_packet: dict
    phone_message: dict
    audio_response: bytes            # TTS WAV bytes (spoken answer)
