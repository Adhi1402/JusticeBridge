"""
Planner / Router agent — the "which knowledge base?" decision.

Given the citizen's words, the Planner looks at the FULL catalogue of legal-
topic KB stores (kb_registry.KB_STORES) and decides which store(s) the
Retrieval agent should search next. It writes:
    state["vertical"]  = primary topic (e.g. "wages")
    state["kb_stores"] = the stores to search (e.g. ["wages", "free_aid"])
    state["supported"] = is this topic actually backed by a KB store?

This is what stops a wage complaint from being matched against family law:
retrieval only ever searches the stores the Planner picked. The cross-cutting
free_aid store is always appended for supported queries, since free-legal-aid
eligibility applies regardless of the substantive topic.

Two routing modes, same output contract:
  - LLM classification (when an on-device LLM is live): the model picks the
    best store id from the catalogue — robust to paraphrase/code-mixing.
  - Keyword scoring (always available fallback): transparent keyword hits per
    store. The pipeline never DEPENDS on the LLM to route.

Unsupported-but-recognised topics (tenancy, fir) route to supported=False so
the graph short-circuits them to the human-handoff branch ("coming soon").

`off_topic` distinguishes "no legal signal matched ANYTHING, not even a stub
vertical" (e.g. "what's the weather today") from "matched a real-but-stub
legal topic" (e.g. an eviction question) — the Output agent uses this to give
a plain "this tool is for legal problems" response instead of a legal-aid
pitch when there's genuinely no legal content in the query.
"""

from ..state import CaseState
from ..kb_registry import KB_STORES, STUB_VERTICALS, always_include_ids
from .. import config
from .. import llm
from ..text_match import count_phrases


def _keyword_scores(text):
    scores = {}
    for sid, cfg in KB_STORES.items():
        if cfg.get("always_include"):
            continue  # free_aid is appended, not matched on its own
        scores[sid] = count_phrases(text, cfg["planner_keywords"])
    for sid, cfg in STUB_VERTICALS.items():
        scores[sid] = count_phrases(text, cfg["planner_keywords"])
    return scores


def _llm_route(text):
    """Ask the on-device LLM to pick the best store id. Returns a store id
    string, a stub vertical id, or None. Raises llm.LLMUnavailable if no model."""
    catalogue = "\n".join(
        f"- {sid}: {cfg['topic']} — {cfg['description']}"
        for sid, cfg in KB_STORES.items() if not cfg.get("always_include")
    )
    stub_list = ", ".join(STUB_VERTICALS.keys())
    system = (
        "You are a legal-intake router. Given a citizen's problem, pick the "
        "single best-matching legal topic id from the catalogue. Answer with "
        "ONLY the id, nothing else. If it clearly matches a not-yet-supported "
        f"topic ({stub_list}) use that id. If nothing matches, answer 'none'."
    )
    user = f"Catalogue:\n{catalogue}\n\nCitizen said: \"{text}\"\n\nBest id:"
    raw = llm.chat(system, user, temperature=0.0).strip().lower()
    # take the first token that matches a known id
    known = set(KB_STORES) | set(STUB_VERTICALS) | {"none"}
    for token in raw.replace(",", " ").split():
        if token in known:
            return None if token == "none" else token
    return None


def planner_agent(state: CaseState) -> dict:
    text = (state.get("combined_text") or "").lower()
    empty = {
        "vertical": None, "supported": False, "kb_stores": [],
        "corpus_subset": [], "output_template": None, "planner_backend": "keyword",
        "off_topic": True,
    }
    if not text.strip():
        return empty

    chosen, backend = None, "keyword"
    llm_unavailable = False

    # Try LLM routing first (if a model is live).
    try:
        chosen = _llm_route(text)
        if chosen is not None:
            backend = "llm"
    except llm.LLMUnavailable as e:
        chosen = None
        llm_unavailable = True

    if llm_unavailable and not config.ALLOW_PLANNER_FALLBACK:
        # Fallback disabled: don't silently switch to keyword routing when the
        # model is simply down. Surface it honestly and route to the safe
        # unsupported/human-handoff branch, same as "nothing matched". This is
        # a backend-availability issue, not a signal about the query itself,
        # so it is NOT treated as off_topic.
        return {**empty, "planner_backend": "unavailable", "off_topic": False,
                "error": ["Planner LLM unavailable and fallback disabled"]}

    # Keyword scoring — either the normal secondary signal (LLM is live but
    # found no match) or the allowed fallback (LLM unavailable).
    if chosen is None:
        scores = _keyword_scores(text)
        best = max(scores, key=scores.get) if scores else None
        if best is None or scores[best] == 0:
            return empty  # off_topic=True: no legal signal matched anything
        chosen, backend = best, "keyword"

    # ---- resolve the chosen id to a routing decision ----
    if chosen in STUB_VERTICALS:
        return {
            "vertical": chosen, "supported": False, "kb_stores": [],
            "corpus_subset": [], "output_template": None, "planner_backend": backend,
            "off_topic": False,  # matched a real (if unsupported) legal topic
        }

    if chosen in KB_STORES:
        cfg = KB_STORES[chosen]
        kb_stores = [chosen] + [s for s in always_include_ids() if s != chosen]
        return {
            "vertical": chosen,
            "supported": True,
            "kb_stores": kb_stores,
            "corpus_subset": kb_stores,
            "output_template": cfg.get("output_template"),
            "planner_backend": backend,
            "off_topic": False,
        }

    return empty
