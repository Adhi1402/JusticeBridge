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


def _after_planner(state: CaseState) -> str:
    return "retrieval" if state.get("supported") else "escalation"


def _after_reasoning(state: CaseState) -> str:
    return "retrieval" if state.get("insufficient_context") else "grounding"


def _after_grounding(state: CaseState) -> str:
    return "reasoning" if state.get("needs_redraft") else "risk"


def build_graph():
    g = StateGraph(CaseState)

    g.add_node("asr", asr_agent)
    g.add_node("vision", vision_agent)
    g.add_node("combine", combine_node)
    g.add_node("planner", planner_agent)
    g.add_node("retrieval", retrieval_agent)
    g.add_node("reasoning", reasoning_agent)
    g.add_node("grounding", grounding_agent)
    g.add_node("risk", risk_agent)
    g.add_node("escalation", escalation_agent)
    g.add_node("output", output_agent)
    g.add_node("translation", translation_agent)
    g.add_node("tts", tts_node)

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
