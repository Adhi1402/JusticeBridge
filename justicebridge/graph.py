"""
The JusticeBridge LangGraph state machine (arch doc Section 2).

Flow (with the two self-correcting loops that read as engineered):

    START ─► asr ┐
    START ─► vision ┘─► combine ─► planner
    planner ─[supported?]─ no ─► escalation ─► output ─► translation ─► END
            └─ yes ─► retrieval
    retrieval ─► reasoning
    reasoning ─[insufficient_context]─► retrieval               (loop 1, bounded)
              └─► grounding
    grounding ─[needs_redraft]─► reasoning                      (loop 2, bounded)
              └─► risk ─► escalation ─► output ─► translation ─► END

Every path ends by handing the user to a real free lawyer. The routers read
explicit flags the agents set (insufficient_context / needs_redraft), so the
wiring stays simple and the loop bounds live inside the agents.
"""

import time
from typing import Any

from langgraph.graph import StateGraph, START, END

from .state import CaseState
from .agents.io_agents import asr_agent, vision_agent, combine_node
from .agents.planner_agent import planner_agent
from .agents.retrieval_agent import retrieval_agent
from .agents.reasoning_agent import reasoning_agent
from .agents.grounding_agent import grounding_agent
from .agents.risk_agent import risk_agent
from .agents.escalation_agent import escalation_agent
from .agents.translation_agent import translation_agent
from .agents.output_agent import output_agent
from .agents.tts_agent import tts_node


def _safe_trace_value(v: Any):
    """Make a node's output values UI/JSON-friendly for agent_trace: no raw
    bytes/PIL images, no runaway-long strings or lists."""
    if isinstance(v, bytes):
        return f"<{len(v)} bytes>"
    if isinstance(v, (list, tuple)):
        if v and hasattr(v[0], "save") and hasattr(v[0], "size"):  # PIL Images
            return f"<{len(v)} image(s)>"
        return [_safe_trace_value(x) for x in list(v)[:8]]
    if isinstance(v, dict):
        return {k: _safe_trace_value(x) for k, x in v.items()}
    if isinstance(v, str):
        return v if len(v) <= 300 else v[:300] + "…"
    if hasattr(v, "save") and hasattr(v, "size"):  # single PIL Image
        return "<image>"
    return v


def _instrumented(name: str, fn):
    """Wrap a node so every agent's outcome (status, duration, output) is
    recorded to state["agent_trace"] — lets the UI show what each of the 10
    pipeline agents actually did, with zero changes to the agent files
    themselves. `error` reducer is additive so this composes safely with the
    two agents (asr/vision) that run in parallel from START."""

    def wrapper(state: CaseState) -> dict:
        start = time.perf_counter()
        try:
            update = fn(state) or {}
            status = "error" if update.get("error") else "ok"
        except Exception as e:
            update = {"error": [f"{name} failed: {e}"]}
            status = "error"
        entry = {
            "agent": name,
            "duration_ms": round((time.perf_counter() - start) * 1000, 1),
            "status": status,
            "output": {k: _safe_trace_value(v) for k, v in update.items()},
        }
        return {**update, "agent_trace": [entry]}

    return wrapper


def _after_planner(state: CaseState) -> str:
    return "retrieval" if state.get("supported") else "escalation"


def _after_reasoning(state: CaseState) -> str:
    return "retrieval" if state.get("insufficient_context") else "grounding"


def _after_grounding(state: CaseState) -> str:
    return "reasoning" if state.get("needs_redraft") else "risk"


def build_graph():
    g = StateGraph(CaseState)

    g.add_node("asr", _instrumented("asr", asr_agent))
    g.add_node("vision", _instrumented("vision", vision_agent))
    g.add_node("combine", _instrumented("combine", combine_node))
    g.add_node("planner", _instrumented("planner", planner_agent))
    g.add_node("retrieval", _instrumented("retrieval", retrieval_agent))
    g.add_node("reasoning", _instrumented("reasoning", reasoning_agent))
    g.add_node("grounding", _instrumented("grounding", grounding_agent))
    g.add_node("risk", _instrumented("risk", risk_agent))
    g.add_node("escalation", _instrumented("escalation", escalation_agent))
    g.add_node("output", _instrumented("output", output_agent))
    g.add_node("translation", _instrumented("translation", translation_agent))
    g.add_node("tts", _instrumented("tts", tts_node))

    g.add_edge(START, "asr")
    g.add_edge(START, "vision")
    g.add_edge("asr", "combine")
    g.add_edge("vision", "combine")
    g.add_edge("combine", "planner")

    g.add_conditional_edges("planner", _after_planner,
                            {"retrieval": "retrieval", "escalation": "escalation"})
    g.add_edge("retrieval", "reasoning")
    g.add_conditional_edges("reasoning", _after_reasoning,
                            {"retrieval": "retrieval", "grounding": "grounding"})
    g.add_conditional_edges("grounding", _after_grounding,
                            {"reasoning": "reasoning", "risk": "risk"})
    g.add_edge("risk", "escalation")
    g.add_edge("escalation", "output")
    g.add_edge("output", "translation")
    g.add_edge("translation", "tts")
    g.add_edge("tts", END)

    return g.compile()


# module-level singleton so the CLI/UI compile once
_APP = None


def get_app():
    global _APP
    if _APP is None:
        _APP = build_graph()
    return _APP
