"""
Grounding-Verification agent (the trust layer).

Every legal claim the Reasoning agent made must map to a section that was
ACTUALLY retrieved. This is the line between trustworthy and dangerous in a
legal tool, and it's a textbook agentic self-critique loop.

Checks per claim (in order, each one strictly ADDITIVE — a claim only needs
to pass, and later checks can only REJECT what earlier checks accepted, never
accept what was already rejected):
  1. Citation check: the claim's section_no must be one of the retrieved
     sections' numbers (the model can't cite a section it wasn't given).
  2. Lexical support check (for LLM drafts): the claim's key terms should have
     real overlap with the cited section's text — catches a model that cites a
     valid section number but attaches a rule the section doesn't contain.
     Free, deterministic, always runs.
  3. Optional LLM entailment check (config.LLM_ASSISTED_GROUNDING): a second,
     independent model call asks "does this section actually support this
     claim?" — catches subtler hallucinations lexical overlap misses (e.g. a
     claim that shares vocabulary with the section but inverts its meaning).
     OFF by default; requires a live LLM; on any failure (no model, parse
     error) this check is skipped and the claim keeps whatever verdict steps
     1-2 gave it — it can only make grounding STRICTER, never looser, and it
     never blocks the pipeline.

If any claim is ungrounded:
  - retry (bounded) -> back to Reasoning to redraft, OR
  - if retries are exhausted, STRIP the ungrounded claims and keep only the
    grounded ones (fail safe, never fail loud with a hallucinated rule).
"""

import re

from ..state import CaseState
from .. import config
from .. import llm

_STOP = set("the a an of to for and or is are be by in on at as with that this "
            "shall may must under section act person any who has have not no "
            "if then from within into it its their his her he she you your".split())


def _keywords(text):
    words = re.findall(r"[a-z]{4,}", text.lower())
    return {w for w in words if w not in _STOP}


def _llm_entails(claim_text, section_text):
    """Ask the LLM whether the section actually supports the claim. Returns
    True/False, or None if the check couldn't run (model unavailable/bad
    response) — callers must treat None as "no opinion", not as a rejection."""
    system = (
        "You are a strict legal fact-checker. Given a statute excerpt and a "
        "claimed rule, answer ONLY 'yes' if the excerpt genuinely supports the "
        "claim, or 'no' if it doesn't (including if it's unrelated or inverts "
        "the meaning). One word only."
    )
    user = f"Statute excerpt:\n{section_text}\n\nClaimed rule: \"{claim_text}\"\n\nSupported?"
    try:
        raw = llm.chat(system, user, temperature=0.0).strip().lower()
    except llm.LLMUnavailable:
        return None
    if raw.startswith("yes"):
        return True
    if raw.startswith("no"):
        return False
    return None


def grounding_agent(state: CaseState) -> dict:
    claims = state.get("draft_claims", []) or []
    sections = state.get("retrieved_sections", []) or []
    backend = state.get("reasoning_backend", "")
    retries = state.get("grounding_retries", 0)

    valid_ids = {s["section_no"] for s in sections}
    text_by_id = {s["section_no"]: s.get("text", "") for s in sections}

    grounded_claims, ungrounded = [], []
    for c in claims:
        sec = c.get("section_no", "")
        claim_text = c.get("claim", "")
        ok = sec in valid_ids
        # Extractive drafts are grounded by construction; only apply the
        # stricter checks to LLM-authored claims.
        if ok and backend not in ("extractive", "none"):
            kw = _keywords(claim_text)
            sec_kw = _keywords(text_by_id.get(sec, ""))
            if kw and len(kw & sec_kw) == 0:
                ok = False  # cites a real section but says something it doesn't
            elif ok and config.LLM_ASSISTED_GROUNDING:
                entails = _llm_entails(claim_text, text_by_id.get(sec, ""))
                if entails is False:
                    ok = False  # only tightens the verdict, never loosens it
        (grounded_claims if ok else ungrounded).append(c)

    all_grounded = len(ungrounded) == 0 and len(grounded_claims) > 0

    if not all_grounded and retries < config.MAX_GROUNDING_RETRIES and claims:
        # Send it back to Reasoning to try again (explicit router flag).
        return {
            "grounded": False,
            "needs_redraft": True,
            "ungrounded_claims": [c.get("claim", "") for c in ungrounded],
            "grounding_retries": retries + 1,
        }

    # Retries exhausted (or nothing to retry): keep only grounded claims.
    return {
        "grounded": len(grounded_claims) > 0,
        "needs_redraft": False,
        "ungrounded_claims": [c.get("claim", "") for c in ungrounded],
        "draft_claims": grounded_claims,
        "grounding_retries": retries,
    }
