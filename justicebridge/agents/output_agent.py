"""
Output agent — assembles the final spoken script (English) and the multi-device
message contracts.

The spoken answer is built from grounded pieces only:
  1. the grounded rights explanation (Reasoning, post Grounding-Verify)
  2. the free-legal-aid headline (Section 12 eligibility) — the killer output
  3. the real deadline / time limit
  4. the human handoff (nearest DLSA + Tele-Law)
  5. the mandatory "information, not advice" close

It also emits the two JSON messages from arch doc Section 5:
  - signal_packet  (AI PC -> UNO Q):  drives the LED + OLED
  - phone_message  (AI PC -> Phone):  drives TTS + on-screen severity
"""

from ..state import CaseState

DISCLAIMER = (
    "This is general legal information, not legal advice for your specific "
    "case. For help with your own situation, please contact the legal aid "
    "office mentioned above — it is free."
)


def _aid_line(state):
    reasons = state.get("eligibility_reasons", []) or []
    if not reasons:
        return ("You may still qualify for free legal aid depending on your "
                "income — the legal aid office can confirm this for you.")
    why = " ".join(reasons[:2])
    return ("Based on what you described, you likely qualify for FREE legal aid "
            "under Section 12 of the Legal Services Authorities Act, 1987. " + why)


def _deadline_line(state):
    days = state.get("deadline_days")
    basis = state.get("deadline_basis") or ""
    if not days:
        return ""
    weeks = max(1, round(days / 7))
    return (f"Please act soon — there is a time limit of about {weeks} weeks "
            f"to take action. {basis}")


def _dlsa_line(state):
    d = state.get("dlsa_contact") or {}
    if not d:
        return ""
    parts = [f"Your nearest free legal aid office is {d.get('name','the District Legal Services Authority')}."]
    if d.get("phone"):
        parts.append(f"Phone: {d['phone']}.")
    if d.get("hours"):
        parts.append(f"Open {d['hours']}.")
    if d.get("bring"):
        parts.append(f"Please bring: {d['bring']}.")
    if d.get("tele_law"):
        parts.append(f"You can also use Tele-Law for a free lawyer: {d['tele_law']}")
    return " ".join(parts)


def _unsupported_answer(state):
    return (
        "Sorry — this kind of legal problem is not yet supported by this "
        "assistant, so I don't want to guess. The best next step is to speak "
        "to a real lawyer for free. " + _aid_line(state) + " " + _dlsa_line(state)
        + " " + DISCLAIMER
    )


def _off_topic_answer():
    """No legal content was detected at all — do NOT push a free-aid/DLSA
    pitch (there is nothing to escalate), just clarify what this tool is for."""
    return (
        "I couldn't find a legal problem in what you said. This assistant "
        "helps with legal questions — for example unpaid wages, a consumer "
        "complaint, or a family/domestic issue. Please describe what "
        "happened and I'll try to help."
    )


def output_agent(state: CaseState) -> dict:
    supported = state.get("supported", False)

    if state.get("off_topic"):
        answer = _off_topic_answer()
    elif not supported:
        answer = _unsupported_answer(state)
    else:
        pieces = [
            state.get("draft_answer", "").strip(),
            _aid_line(state),
            _deadline_line(state),
            _dlsa_line(state),
            DISCLAIMER,
        ]
        answer = " ".join(p for p in pieces if p).strip()

    d = state.get("dlsa_contact") or {}
    signal_packet = {
        "severity": state.get("severity", "green"),
        "category": state.get("vertical") or "unsupported",
        "confidence": state.get("composite_confidence", 0.0),
        "deadline_days": state.get("deadline_days"),
        "dlsa": {
            "name": d.get("name", ""),
            "phone": d.get("phone", ""),
            "bring": d.get("bring", ""),
        },
        "qualifies_for_aid": bool(state.get("eligibility_reasons")),
    }

    phone_message = {
        "answer_local": None,  # filled by the Translation agent
        "answer_en": answer,
        "severity": state.get("severity", "green"),
        "category": state.get("vertical") or "unsupported",
        "deadline_days": state.get("deadline_days"),
        "escalated": bool(state.get("escalate")),
    }

    return {
        "final_answer_en": answer,
        "signal_packet": signal_packet,
        # stash the phone message inside signal_packet's sibling via state;
        # keep it discoverable for the CLI/UI:
        "phone_message": phone_message,
    }
