"""
Risk / Deadline agent — turns signals into (a) a composite confidence number
and (b) a grounded urgency colour.

Urgency is a REAL legal clock (limitation / action window from deadlines.json),
never a vibe: red = a deadline is close, amber = act soon, green = no clock.
Composite confidence blends ASR, Vision (only when a document was actually
scanned), retrieval strength, and a penalty for how many retries it took —
and feeds the escalation decision.
"""

import json

from ..state import CaseState
from .. import config

with open(config.DEADLINES_FILE, "r", encoding="utf-8") as _f:
    _DEADLINES = json.load(_f)["matters"]


def _severity_from_days(days):
    if days is None:
        return "green"
    if days <= config.DEADLINE_RED_DAYS:
        return "red"
    if days <= config.DEADLINE_AMBER_DAYS:
        return "amber"
    return "green"


def risk_agent(state: CaseState) -> dict:
    asr = state.get("asr_confidence", 0.0) or 0.0
    vis = state.get("vision_confidence", None)
    sim = state.get("retrieval_sim", 0.0) or 0.0
    retries = state.get("retry_count", 0) + state.get("grounding_retries", 0)
    grounded = state.get("grounded", False)

    # --- composite confidence ---
    # retrieval strength is the biggest factor; a document, if present, adds a
    # little; each retry shaves confidence; ungrounded output caps it low.
    conf = 0.55 * min(sim / 0.5, 1.0) + 0.25 * asr
    if vis is not None:
        conf += 0.10 * vis
    else:
        conf += 0.10  # no doc expected for a spoken-only wage claim; don't penalise
    conf -= 0.08 * retries
    if not grounded:
        conf = min(conf, 0.45)
    composite = round(max(0.0, min(1.0, conf)), 3)

    # --- deadline / severity ---
    # Map the KB store to its limitation-table key (wages->wages,
    # consumer->consumer, family->general, ...) via the registry.
    from ..kb_registry import KB_STORES
    vertical = state.get("vertical") or "general"
    deadline_key = KB_STORES.get(vertical, {}).get("deadline_key", vertical)
    matter = _DEADLINES.get(deadline_key, _DEADLINES["general"])
    deadline_days = matter.get("recommended_action_days")
    base_signal = matter.get("signal", "green")
    computed = _severity_from_days(deadline_days)

    # severity = the more urgent of (matter's declared signal, computed-from-days)
    order = {"green": 0, "amber": 1, "red": 2}
    severity = base_signal if order[base_signal] >= order[computed] else computed

    return {
        "composite_confidence": composite,
        "deadline_days": deadline_days,
        "deadline_basis": matter.get("limitation_basis"),
        "severity": severity,
    }
