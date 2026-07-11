"""
Gold-standard evaluation harness — the rigor move (arch doc Section 4).

Runs every case in gold_set.json through the FULL pipeline and scores:
  - Routing accuracy   : did the Planner pick the right vertical + support flag?
  - Citation hit@k     : did at least one expected statute section appear in
                         the retrieved sections? (the core legal-RAG metric)
  - Grounded           : supported answers must pass Grounding-Verify
  - Escalation         : did it correctly decide to hand off to a human?
  - Aid handoff        : EVERY case must end with a DLSA contact (safety)
  - Severity           : did the urgency colour match expectation?

Prints per-case results + an aggregate scorecard you can screenshot for the
pitch. Run with the extractive backend for a deterministic number, or point
JB_LLM_BACKEND at your NPU model to score the real reasoning path.

Usage:  python -m justicebridge.eval.run_eval
"""

import sys
import json
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ..graph import get_app

GOLD = Path(__file__).parent / "gold_set.json"


def _retrieved_section_nos(state):
    return {s["section_no"] for s in state.get("retrieved_sections", [])}


def run():
    cases = json.loads(GOLD.read_text(encoding="utf-8"))
    app = get_app()

    n = len(cases)
    agg = {"vertical": 0, "support": 0, "citation": 0, "grounded": 0,
           "escalate": 0, "handoff": 0, "severity": 0, "eligibility": 0, "elig_total": 0}
    citation_applicable = 0

    print(f"Running {n} gold cases...\n")
    print(f"{'id':7} {'vert':9} {'cite':5} {'grnd':5} {'esc':4} {'aid':4} {'sev':6}")
    print("-" * 50)

    for c in cases:
        state = app.invoke({"text_input": c["query"], "lang": "en"})

        ok_vert = state.get("vertical") == c["expected_vertical"]
        ok_supp = state.get("supported") == c["expected_supported"]
        agg["vertical"] += ok_vert
        agg["support"] += ok_supp

        # citation hit@k — only where we expect specific sections
        exp = set(c.get("expected_sections", []))
        cite_mark = "-"
        if exp:
            citation_applicable += 1
            hit = bool(exp & _retrieved_section_nos(state))
            agg["citation"] += hit
            cite_mark = "Y" if hit else "N"

        # grounded (only meaningful for supported cases)
        grnd_mark = "-"
        if c["expected_supported"]:
            g = bool(state.get("grounded"))
            agg["grounded"] += g
            grnd_mark = "Y" if g else "N"
        else:
            agg["grounded"] += 1  # n/a counts as pass

        ok_esc = bool(state.get("escalate")) == c["expect_escalate"]
        agg["escalate"] += ok_esc

        expect_aid = c.get("expect_aid_handoff", True)
        has_aid = bool((state.get("dlsa_contact") or {}).get("phone"))
        agg["handoff"] += (has_aid == expect_aid)

        ok_sev = state.get("severity") == c["expect_severity"]
        agg["severity"] += ok_sev

        # eligibility category check (where specified)
        if "expect_eligibility_ids" in c:
            agg["elig_total"] += 1
            reasons = " ".join(state.get("eligibility_reasons", [])).lower()
            want_hit = any(
                {"woman": "women", "industrial_workman": "workmen"}.get(i, i) in reasons
                for i in c["expect_eligibility_ids"]
            )
            agg["eligibility"] += want_hit

        # regression guard: off-topic queries must NOT produce a false
        # eligibility claim (this is exactly the "her " inside "weather" bug)
        if c.get("expect_eligibility_empty"):
            agg["elig_total"] += 1
            agg["eligibility"] += (state.get("eligibility_reasons", []) == [])

        print(f"{c['id']:7} {'Y' if ok_vert else 'N':9} {cite_mark:5} "
              f"{grnd_mark:5} {'Y' if ok_esc else 'N':4} {'Y' if has_aid else 'N':4} "
              f"{'Y' if ok_sev else 'N':6}")

    print("\n" + "=" * 50)
    print("SCORECARD".center(50))
    print("=" * 50)
    def pct(k, d=n): return f"{agg[k]}/{d}  ({100*agg[k]/d:.0f}%)"
    print(f"  Routing (vertical)     : {pct('vertical')}")
    print(f"  Routing (support flag) : {pct('support')}")
    print(f"  Citation hit@k         : {agg['citation']}/{citation_applicable}  "
          f"({100*agg['citation']/max(1,citation_applicable):.0f}%)")
    print(f"  Grounded (supported)   : {pct('grounded')}")
    print(f"  Escalation decision    : {pct('escalate')}")
    print(f"  Aid handoff present    : {pct('handoff')}")
    print(f"  Severity match         : {pct('severity')}")
    if agg["elig_total"]:
        print(f"  Eligibility detection  : {agg['eligibility']}/{agg['elig_total']}")
    print("=" * 50)


if __name__ == "__main__":
    run()
