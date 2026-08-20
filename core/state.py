"""
core/state.py
─────────────
LangGraph State definition with custom reducers.

Key design decisions:
- `messages` uses add_messages reducer (append-only)
- `sources` uses a custom deduplicating reducer
- Numeric counters use simple overwrite (last-write-wins)
- All agent outputs stored separately for traceability
"""

from __future__ import annotations
import operator
from typing import Annotated, Any, Optional, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from schemas.outputs import (
    ResearchOutput,
    FactCheckOutput,
    SummarizerOutput,
    AnalystOutput,
    CriticOutput,
    ReportOutput,
    Source,
)


def dedupe_sources(
    existing: list[Source], new: list[Source]
) -> list[Source]:
    """Custom reducer: merge source lists, deduplicating by URL or title."""
    seen_keys: set[str] = {s.url or s.title for s in existing}
    merged = list(existing)
    for src in new:
        key = src.url or src.title
        if key not in seen_keys:
            merged.append(src)
            seen_keys.add(key)
    return merged


def append_strings(existing: list[str], new: list[str]) -> list[str]:
    """Reducer that appends new strings to existing list."""
    return existing + new


class ResearchState(BaseModel):
    """
    Complete state object passed between all LangGraph nodes.
    
    Reducers:
    - messages      → add_messages (append, dedup by ID)
    - all_sources   → dedupe_sources (deduplicating merge)
    - critique_log  → append_strings (accumulate feedback history)
    - Everything else → last-write-wins (direct assignment)
    """

    # ── Core input ────────────────────────────────────────────────────────────
    query: str = Field(default="", description="The original user research query")
    refined_query: str = Field(default="", description="Query refined by researcher")

    # ── LangChain message history ─────────────────────────────────────────────
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(
        default_factory=list
    )

    # ── Agent outputs (typed) ─────────────────────────────────────────────────
    research_output: Optional[dict] = Field(
        default=None, description="ResearchOutput dict"
    )
    fact_check_output: Optional[dict] = Field(
        default=None, description="FactCheckOutput dict"
    )
    summarizer_output: Optional[dict] = Field(
        default=None, description="SummarizerOutput dict"
    )
    analyst_output: Optional[dict] = Field(
        default=None, description="AnalystOutput dict"
    )
    critic_output: Optional[dict] = Field(
        default=None, description="CriticOutput dict"
    )
    report_output: Optional[dict] = Field(
        default=None, description="ReportOutput dict"
    )

    # ── Deduplicated source accumulator ───────────────────────────────────────
    all_sources: Annotated[list[Source], dedupe_sources] = Field(
        default_factory=list
    )

    # ── Critique loop tracking ────────────────────────────────────────────────
    critique_loop_count: int = Field(default=0)
    critique_log: Annotated[list[str], append_strings] = Field(
        default_factory=list,
        description="History of critic feedback across loops"
    )
    quality_score: float = Field(default=0.0)
    approved: bool = Field(default=False)

    # ── Progress / observability ──────────────────────────────────────────────
    current_agent: str = Field(default="", description="Currently executing agent")
    agent_status: dict[str, str] = Field(
        default_factory=dict,
        description="Status per agent: 'pending' | 'running' | 'done' | 'error'"
    )
    error_log: list[str] = Field(default_factory=list)

    # ── Final report ──────────────────────────────────────────────────────────
    final_report_markdown: str = Field(default="")
    
    class Config:
        arbitrary_types_allowed = True