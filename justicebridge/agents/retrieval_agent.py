"""
Retrieval agent — hybrid BM25 + vector search over ONLY the KB stores the
Planner selected (state["kb_stores"]). Thin node around retrieval.retrieve();
the interesting logic (per-store RRF fusion, cross-store merge, the similarity
signal) lives in retrieval.py.

On a retry (Reasoning flagged insufficient_context) it widens k so the second
attempt sees more candidate sections.
"""

from ..state import CaseState
from .. import config
from ..retrieval import retrieve


def retrieval_agent(state: CaseState) -> dict:
    query = (state.get("combined_text") or "").strip()
    kb_stores = state.get("kb_stores") or ([state["vertical"]] if state.get("vertical") else None)
    retry = state.get("retry_count", 0)

    k = config.RETRIEVAL_K + (3 * retry)  # widen on retry
    sections, sim = retrieve(query, kb_stores=kb_stores, k=k)

    return {
        "retrieved_sections": sections,
        "retrieval_sim": sim,
    }
