"""
core/streaming.py
─────────────────
Streaming support for the research graph.

Two streaming modes:
1. stream_events() — yields LangGraph events (node start/end, state updates)
2. stream_to_callback() — calls a callback function for each update
3. run_with_progress() — yields structured progress dicts for Streamlit
"""

from __future__ import annotations
from typing import Any, Callable, Iterator, Generator
import asyncio
from core.state import ResearchState
from core.graph import get_graph
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Human-readable agent labels ──────────────────────────────────────────────
AGENT_LABELS = {
    "researcher":    "🔬 Researcher",
    "fact_checker":  "🔎 Fact-Checker",
    "summarizer":    "📋 Summarizer",
    "analyst":       "🧠 Analyst",
    "critic":        "⚖️  Critic",
    "report_writer": "✍️  Report Writer",
}

AGENT_DESCRIPTIONS = {
    "researcher":    "Searching web and private documents...",
    "fact_checker":  "Verifying factual claims...",
    "summarizer":    "Distilling key insights...",
    "analyst":       "Building reasoning chains...",
    "critic":        "Evaluating research quality...",
    "report_writer": "Writing final report...",
}


def stream_graph(
    query: str,
    config: dict | None = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Stream the graph execution, yielding progress events.
    
    Yields dicts with structure:
    {
        "type": "node_start" | "node_end" | "state_update" | "complete" | "error",
        "node": str,
        "label": str,
        "description": str,
        "state": dict | None,
        "error": str | None,
    }
    """
    graph = get_graph()

    initial_state = ResearchState(
        query=query,
        agent_status={
            "researcher":    "pending",
            "fact_checker":  "pending",
            "summarizer":    "pending",
            "analyst":       "pending",
            "critic":        "pending",
            "report_writer": "pending",
        }
    )

    stream_config = config or {"recursion_limit": 25}

    try:
        for event in graph.stream(
            initial_state.model_dump(),
            config=stream_config,
            stream_mode="updates",
        ):
            # event is a dict: {node_name: state_update_dict}
            for node_name, update in event.items():
                label = AGENT_LABELS.get(node_name, node_name)
                desc = AGENT_DESCRIPTIONS.get(node_name, "Processing...")

                logger.info(f"📡 Stream event: [{node_name}] update received")

                yield {
                    "type": "node_update",
                    "node": node_name,
                    "label": label,
                    "description": desc,
                    "state": update,
                    "error": None,
                }

    except Exception as e:
        logger.error(f"Graph streaming error: {e}", exc_info=True)
        yield {
            "type": "error",
            "node": "graph",
            "label": "Error",
            "description": str(e),
            "state": None,
            "error": str(e),
        }


def run_sync(
    query: str,
    config: dict | None = None,
) -> tuple[str, dict]:
    """
    Run graph synchronously (no streaming). Returns (final_markdown, final_state).
    Used for CLI and testing.
    """
    graph = get_graph()

    initial_state = ResearchState(
        query=query,
        agent_status={
            "researcher":    "pending",
            "fact_checker":  "pending",
            "summarizer":    "pending",
            "analyst":       "pending",
            "critic":        "pending",
            "report_writer": "pending",
        }
    )

    stream_config = config or {"recursion_limit": 25}

    logger.info(f"🚀 Starting research: '{query}'")

    final_state = graph.invoke(
        initial_state.model_dump(),
        config=stream_config,
    )

    markdown = final_state.get("final_report_markdown", "")
    logger.info(f"✅ Research complete | Report: {len(markdown)} chars")

    return markdown, final_state