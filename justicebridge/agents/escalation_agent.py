"""
Escalation / Aid agent — the headline feature.

Does two things, both pure lookups over structured data (zero hallucination):
  1. Section 12 eligibility: scans the citizen's own words for the automatic
     free-aid categories (woman, industrial workman, SC/ST, disability, ...).
     For a wage case, "industrial workman" alone usually already qualifies.
  2. DLSA handoff: attaches the nearest DLSA (name, phone, hours, what to
     bring) + the Tele-Law option, so "talk to a real free lawyer" is a
     first-class output.

It ALWAYS attaches a human handoff. It fires as the primary branch when the
matter is unsupported/uncertain, and also runs on the supported happy path so
every answer ends by walking the user to a free lawyer.
"""

import json

from ..state import CaseState
from .. import config

with open(config.ELIGIBILITY_FILE, "r", encoding="utf-8") as _f:
    _ELIG = json.load(_f)

with open(config.DLSA_FILE, "r", encoding="utf-8") as _f:
    _DLSA = json.load(_f)


def _match_eligibility(text):
    text = f" {text.lower()} "
    reasons = []
    for cat in _ELIG["categories"]:
        if any(cue in text for cue in cat["cues"]):
            reasons.append(cat["explanation"])
    return reasons


def _lookup_dlsa(district):
    districts = _DLSA["districts"]
    entry = districts.get((district or "").lower()) or districts["default"]
    return {
        "name": entry["name"],
        "phone": entry["phone"],
        "hours": entry["hours"],
        "address": entry.get("address", ""),
        "taluka_committee": entry.get("taluka_committee", ""),
        "bring": entry["bring"],
        "tele_law": _DLSA["tele_law"]["helpline"],
        "helpline": _DLSA.get("state_helpline", ""),
    }


def escalation_agent(state: CaseState) -> dict:
    text = state.get("combined_text", "") or ""
    supported = state.get("supported", False)
    composite = state.get("composite_confidence", 0.0) or 0.0
    severity = state.get("severity", "green")

    reasons = _match_eligibility(text)

    # Vertical-based presumption: a wage dispute is, by its nature, brought by
    # a worker — and industrial workmen qualify automatically under Section 12.
    # If the citizen's words didn't already trigger a category, add the
    # worker category so the headline "you likely qualify" output still fires
    # (softer wording is handled downstream). Arch doc: "This covers most
    # unpaid-wage cases."
    # Per-vertical Section-12 presumptions: the substantive law of some
    # verticals implies an automatic-eligibility category even when the
    # citizen's exact words didn't name it.
    #   wages  -> industrial workman (most wage claims are by workers)
    #   family -> woman (the Protection of Women from Domestic Violence Act is,
    #             by definition, invoked by a woman)
    _VERTICAL_PRESUMPTION = {"wages": "industrial_workman", "family": "woman"}
    presumed = _VERTICAL_PRESUMPTION.get(state.get("vertical"))
    if presumed:
        expl = next((c["explanation"] for c in _ELIG["categories"]
                     if c["id"] == presumed), None)
        if expl and expl not in reasons:
            reasons.insert(0, expl)

    dlsa = _lookup_dlsa(config.DEFAULT_DISTRICT)

    # Escalate (make the human handoff prominent) when: unsupported vertical,
    # low confidence, tight deadline, or the user clearly qualifies for aid.
    escalate = (
        (not supported)
        or composite < config.LOW_CONFIDENCE_ESCALATE
        or severity == "red"
        or len(reasons) > 0
    )

    return {
        "escalate": escalate,
        "eligibility_reasons": reasons,
        "dlsa_contact": dlsa,
        # On the unsupported path the Risk agent is skipped, so severity is
        # still unset here — default it to green so the signal is always defined.
        "severity": state.get("severity") or "green",
    }
