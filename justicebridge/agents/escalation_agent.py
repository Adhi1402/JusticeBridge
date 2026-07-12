"""
Escalation / Aid agent — the headline feature.

Does two things, both pure lookups over structured data (zero hallucination):
  1. Section 12 eligibility: scans the citizen's own words for the automatic
     free-aid categories (woman, industrial workman, SC/ST, disability, ...).
     For a wage case, "industrial workman" alone usually already qualifies.
     Optional LLM assist (config.LLM_ASSISTED_ELIGIBILITY, off by default):
     the keyword cue list is deliberately literal ("disabled", "wheelchair")
     and misses implied categories ("I've needed a wheelchair since the
     accident"). When enabled, a second model call re-checks the same fixed
     category list against the citizen's words and can ADD a category the
     keywords missed — it can never remove a keyword-matched category, so
     enabling this can only make eligibility detection MORE generous, never
     less. Any failure (no model) is skipped silently — same guarantee as
     every other optional LLM path in this codebase.
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
from .. import llm
from ..text_match import any_phrase

with open(config.ELIGIBILITY_FILE, "r", encoding="utf-8") as _f:
    _ELIG = json.load(_f)

with open(config.DLSA_FILE, "r", encoding="utf-8") as _f:
    _DLSA = json.load(_f)


def _match_eligibility(text):
    text = text.lower()
    reasons = []
    for cat in _ELIG["categories"]:
        if any_phrase(text, cat["cues"]):
            reasons.append(cat["explanation"])
    return reasons


def _llm_extra_eligibility(text, already_found_ids):
    """Ask the LLM to spot Section-12 categories the keyword scan missed.
    Returns a list of explanation strings to ADD (never a removal). Any
    failure returns [] — this check can only add, and silently no-ops."""
    remaining = [c for c in _ELIG["categories"] if c["id"] not in already_found_ids]
    if not remaining:
        return []
    catalogue = "\n".join(f"- {c['id']}: {c['label']}" for c in remaining)
    system = (
        "You determine free-legal-aid eligibility under India's Legal Services "
        "Authorities Act, Section 12. Given what a citizen said, list ONLY the "
        "category ids from the catalogue that clearly apply, even if the exact "
        "keyword wasn't used (e.g. 'needed a wheelchair since the accident' "
        "implies disability). One id per line. If none apply, answer 'none'."
    )
    user = f"Catalogue:\n{catalogue}\n\nCitizen said: \"{text}\"\n\nApplicable ids:"
    try:
        raw = llm.chat(system, user, temperature=0.0).strip().lower()
    except llm.LLMUnavailable:
        return []
    found_ids = {c["id"] for c in remaining if c["id"] in raw}
    return [c["explanation"] for c in remaining if c["id"] in found_ids]


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
    off_topic = state.get("off_topic", False)
    composite = state.get("composite_confidence", 0.0) or 0.0
    severity = state.get("severity", "green")

    # A query with NO legal signal at all gets no eligibility claim and no
    # DLSA push — "you likely qualify for free legal aid" is meaningless (and
    # was, before this fix, occasionally FALSE-POSITIVE-matched — see
    # text_match.py) for a query like "what's the weather today".
    if off_topic:
        return {
            "escalate": False,
            "eligibility_reasons": [],
            "dlsa_contact": None,
            "severity": state.get("severity") or "green",
        }

    reasons = _match_eligibility(text)

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

    if config.LLM_ASSISTED_ELIGIBILITY:
        found_ids = {c["id"] for c in _ELIG["categories"] if c["explanation"] in reasons}
        reasons.extend(r for r in _llm_extra_eligibility(text, found_ids) if r not in reasons)

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
