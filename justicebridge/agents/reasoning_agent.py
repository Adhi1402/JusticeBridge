"""
Reasoning agent — turns retrieved statute sections into a plain-language
explanation that cites ONLY those sections. Never invents law.

Two backends, same contract:
  - on_device_reasoning_tool: an on-device LLM (Qualcomm GenieX / onnxruntime-
    genai+QNN on the NPU, or an OpenAI-compatible server for dev) asked for a
    JSON object with `answer` (plain language) and `claims` (each
    {claim, section_no}) so the Grounding agent can verify every claim maps to
    a real retrieved section. See llm.py for backend selection.
  - Extractive fallback (no model, always available): builds the answer
    directly from the retrieved section titles/text. Every sentence is, by
    construction, tied to a retrieved section — so it is inherently grounded
    and cannot hallucinate. This guarantees the pipeline produces a correct,
    citable answer even when no on-device model is loaded.

It also sets `insufficient_context` when retrieval was too weak — the graph
loops back to Retrieval (bounded) before ever drafting from thin air.
"""

from langchain_core.tools import tool

from ..state import CaseState
from .. import config
from .. import llm

SYSTEM = (
    "You are a careful legal-information assistant for India. You explain the "
    "law in plain, simple language a worker with no legal background can "
    "understand. You give general legal INFORMATION, never advice for a "
    "specific case. You must rely ONLY on the numbered statute sections given "
    "to you. Never state a legal rule that is not supported by one of those "
    "sections. Do not mention penalties, amounts, or deadlines unless they "
    "appear in the provided sections."
)


def _sections_block(sections):
    lines = []
    for s in sections:
        lines.append(
            f"[{s['section_no']}] {s['act']}, Section {s['section_no']} "
            f"({s.get('title','')}): {s['text']}"
        )
    return "\n\n".join(lines)


@tool
def on_device_reasoning_tool(query: str, sections: list[dict]) -> dict:
    """Draft a plain-language legal explanation using ONLY the given retrieved
    statute sections, via the configured on-device LLM (GenieX/onnx_qnn/
    openai — see config.LLM_BACKEND). Each section dict needs act, section_no,
    title, text. Returns {"answer": str, "claims": [{"claim","section_no"}]}."""
    block = _sections_block(sections)
    valid_ids = ", ".join(s["section_no"] for s in sections)
    user = (
        f"The person said:\n\"{query}\"\n\n"
        f"You may use ONLY these statute sections (cite by their [section_no]):\n\n"
        f"{block}\n\n"
        f"Write a short, plain-language explanation (4-6 sentences) of the "
        f"person's rights and next practical step, using ONLY the sections "
        f"above. Then list the individual legal claims you made, each tied to "
        f"the section number that supports it.\n\n"
        f"Respond as JSON: {{\"answer\": \"...\", \"claims\": "
        f"[{{\"claim\": \"...\", \"section_no\": \"<one of: {valid_ids}>\"}}]}}"
    )
    obj = llm.chat_json(SYSTEM, user)
    answer = (obj.get("answer") or "").strip()
    claims = []
    for c in obj.get("claims", []):
        if isinstance(c, dict) and c.get("claim"):
            claims.append({"claim": str(c["claim"]).strip(),
                           "section_no": str(c.get("section_no", "")).strip()})
    if not answer:
        raise llm.LLMUnavailable("empty answer from model")
    return {"answer": answer, "claims": claims}


def _llm_draft(query, sections):
    result = on_device_reasoning_tool.invoke({"query": query, "sections": sections})
    return result["answer"], result["claims"]


def _extractive_draft(query, sections):
    """Deterministic, hallucination-proof draft straight from retrieved text."""
    top = sections[:3]
    parts = ["Here is what the law says about your situation:"]
    claims = []
    for s in top:
        title = (s.get("title") or "").strip().rstrip(".")
        # retrieval prepends the title to the body for better matching; strip
        # it back off here so the snippet doesn't repeat the title we print.
        body = s["text"]
        if title and body.startswith(title):
            body = body[len(title):].lstrip(". ").strip()
        snippet = " ".join(body.split()[:45]).strip()
        sentence = f"Under {s['act']}, Section {s['section_no']} ({title}): {snippet}."
        parts.append(sentence)
        claims.append({"claim": f"{title} — {s['act']} s.{s['section_no']}",
                       "section_no": s["section_no"]})
    answer = " ".join(parts)
    return answer, claims


def reasoning_agent(state: CaseState) -> dict:
    sections = state.get("retrieved_sections", []) or []
    query = (state.get("combined_text") or "").strip()
    retry = state.get("retry_count", 0)
    sim = state.get("retrieval_sim", 0.0)

    # Weak retrieval -> ask the graph to try retrieval again before drafting.
    if not sections or sim < config.RETRIEVAL_MIN_SIM:
        if retry < config.MAX_RETRIEVAL_RETRIES:
            return {"insufficient_context": True, "retry_count": retry + 1}
        # exhausted retries: fall through and draft from whatever we have

    if not sections:
        return {
            "insufficient_context": False,
            "draft_answer": "",
            "draft_claims": [],
            "citations": [],
            "reasoning_backend": "none",
        }

    backend = config.LLM_BACKEND
    try:
        answer, claims = _llm_draft(query, sections)
        used = backend
    except llm.LLMUnavailable:
        answer, claims = _extractive_draft(query, sections)
        used = "extractive"

    citations = [{"act": s["act"], "section_no": s["section_no"], "title": s.get("title", "")}
                 for s in sections[:3]]

    return {
        "insufficient_context": False,
        "draft_answer": answer,
        "draft_claims": claims,
        "citations": citations,
        "reasoning_backend": used,
    }
