"""
Grounding-Verification agent (the trust layer).

Every legal claim the Reasoning agent made must map to a section that was
ACTUALLY retrieved. This is the line between trustworthy and dangerous in a
legal tool, and it's a textbook agentic self-critique loop.

Two checks per claim:
  1. Citation check: the claim's section_no must be one of the retrieved
     sections' numbers (the model can't cite a section it wasn't given).
  2. Lexical support check (for LLM drafts): the claim's key terms should have
     real overlap with the cited section's text — catches a model that cites a
     valid section number but attaches a rule the section doesn't contain.

If any claim is ungrounded:
  - retry (bounded) -> back to Reasoning to redraft, OR
  - if retries are exhausted, STRIP the ungrounded claims and keep only the
    grounded ones (fail safe, never fail loud with a hallucinated rule).
"""

import re

from ..state import CaseState
from .. import config

_STOP = set("the a an of to for and or is are be by in on at as with that this "
            "shall may must under section act person any who has have not no "
            "if then from within into it its their his her he she you your".split())


def _keywords(text):
    words = re.findall(r"[a-z]{4,}", text.lower())
    return {w for w in words if w not in _STOP}


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
        # stricter lexical-overlap test to LLM-authored claims.
        if ok and backend not in ("extractive", "none"):
            kw = _keywords(claim_text)
            sec_kw = _keywords(text_by_id.get(sec, ""))
            if kw and len(kw & sec_kw) == 0:
                ok = False  # cites a real section but says something it doesn't
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
