"""
core/graph.py
─────────────
LangGraph graph construction.

Graph topology:
  researcher → fact_checker → summarizer → analyst → critic
                                                         ↓
                                          approved? → report_writer → END
                                                         ↓
                                          rejected? → researcher (loop)

Conditional edge at critic:
  - If approved OR max_loops reached → report_writer
  - If rejected AND loops < max → researcher
"""

from __future__ import annotations
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from core.state import ResearchState
from agents.researcher import researcher_node
from agents.fact_checker import fact_checker_node
from agents.summarizer import summarizer_node
from agents.analyst import analyst_node
from agents.critic import critic_node
from agents.report_writer import report_writer_node
from core.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Node status update wrappers ───────────────────────────────────────────────

def mark_running(agent_name: str):
    """Return a node wrapper that marks agent as running before execution."""
    def _set_running(state: ResearchState) -> dict:
        return {
            "current_agent": agent_name,
            "agent_status": {**state.agent_status, agent_name: "running"},
        }
    _set_running.__name__ = f"start_{agent_name}"
    return _set_running


# ── Conditional edge logic ────────────────────────────────────────────────────

def should_loop_or_proceed(state: ResearchState) -> str:
    """
    Routing function for the conditional edge after the Critic.
    
    Returns:
        'report_writer' — if approved or max loops reached
        'researcher'    — if rejected and loops remaining
    """
    if state.approved:
        logger.info(f"✅ Critic approved (score={state.quality_score:.1f}) → proceeding to report")
        return "report_writer"

    if state.critique_loop_count >= settings.max_critique_loops:
        logger.warning(
            f"⚠️  Max loops ({settings.max_critique_loops}) reached — forcing report generation"
        )
        return "report_writer"

    logger.info(
        f"🔄 Critic rejected (score={state.quality_score:.1f}) → "
        f"re-researching (loop {state.critique_loop_count}/{settings.max_critique_loops})"
    )
    return "researcher"


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph(use_checkpointing: bool = False) -> StateGraph:
    """
    Build and compile the research multi-agent graph.
    
    Args:
        use_checkpointing: Whether to add MemorySaver for persistence.
    
    Returns:
        Compiled LangGraph StateGraph
    """
    # ── Initialise graph with state schema ───────────────────────────────────
    graph = StateGraph(ResearchState)

    # ── Add all agent nodes ──────────────────────────────────────────────────
    graph.add_node("researcher", researcher_node)
    graph.add_node("fact_checker", fact_checker_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("critic", critic_node)
    graph.add_node("report_writer", report_writer_node)

    # ── Define the main pipeline edges ──────────────────────────────────────
    graph.set_entry_point("researcher")
    graph.add_edge("researcher",   "fact_checker")
    graph.add_edge("fact_checker", "summarizer")
    graph.add_edge("summarizer",   "analyst")
    graph.add_edge("analyst",      "critic")

    # ── Conditional edge: critic → loop OR proceed ───────────────────────────
    graph.add_conditional_edges(
        "critic",
        should_loop_or_proceed,
        {
            "report_writer": "report_writer",
            "researcher":    "researcher",   # Loop back
        },
    )

    # ── Final edge to END ────────────────────────────────────────────────────
    graph.add_edge("report_writer", END)

    # ── Compile ──────────────────────────────────────────────────────────────
    if use_checkpointing:
        memory = MemorySaver()
        compiled = graph.compile(checkpointer=memory)
    else:
        compiled = graph.compile()

    logger.info("✅ Research graph compiled successfully")
    return compiled


# ── Singleton compiled graph ─────────────────────────────────────────────────
_compiled_graph = None


def get_graph() -> StateGraph:
    """Return the singleton compiled graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph