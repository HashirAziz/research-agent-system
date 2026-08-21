"""
agents/analyst.py
─────────────────
Analyst agent — multi-hop reasoning and causal analysis.
"""

from __future__ import annotations
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from core.config import settings
from core.state import ResearchState
from schemas.outputs import AnalystOutput
from prompts.analyst_prompt import ANALYST_SYSTEM_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=0.4,
        api_key=settings.openai_api_key,
        max_tokens=settings.max_tokens_per_agent,
    )


def analyst_node(state: ResearchState) -> dict:
    logger.info("[agent.analyst]🧠 Analyst Agent starting...")

    research = state.research_output
    summary = state.summarizer_output
    fact_check = state.fact_check_output

    if not research or not summary:
        logger.warning("Insufficient data for analysis.")
        return {
            "analyst_output": None,
            "current_agent": "analyst",
            "agent_status": {**state.agent_status, "analyst": "skipped"},
        }

    key_points = "\n".join(f"• {p}" for p in summary.get("key_points", []))
    themes = ", ".join(summary.get("themes", []))
    contradictions = "\n".join(f"⚡ {c}" for c in summary.get("contradictions", []))
    data_points = "\n".join(f"📊 {d}" for d in summary.get("data_points", []))
    research_gaps = "\n".join(f"❓ {g}" for g in research.get("research_gaps", []))

    reliability = ""
    if fact_check:
        reliability = f"\nOverall source reliability: {fact_check.get('overall_reliability', 'N/A')}"

    analyst_prompt = f"""Research Query: {state.query}

EXECUTIVE SUMMARY:
{summary.get('executive_summary', '')}

KEY FINDINGS:
{key_points}

MAJOR THEMES: {themes}

CONTRADICTIONS / TENSIONS:
{contradictions}

QUANTITATIVE DATA:
{data_points}

RESEARCH GAPS:
{research_gaps}
{reliability}

Perform deep multi-hop reasoning analysis. Build explicit reasoning chains.
Identify root causes, map implications, and provide actionable recommendations."""

    llm = get_llm().with_structured_output(AnalystOutput, method="function_calling")

    try:
        analysis: AnalystOutput = llm.invoke([
            SystemMessage(content=ANALYST_SYSTEM_PROMPT),
            HumanMessage(content=analyst_prompt),
        ])
        logger.info(
            f"   → {len(analysis.multi_hop_chains)} reasoning chains | "
            f"{len(analysis.implications)} implications | "
            f"{len(analysis.recommendations)} recommendations"
        )
        return {
            "analyst_output": analysis.model_dump(),
            "current_agent": "analyst",
            "agent_status": {**state.agent_status, "analyst": "done"},
        }
    except Exception as e:
        logger.error(f"Analyst LLM call failed: {e}")
        fallback = AnalystOutput(
            multi_hop_chains=[],
            root_causes=[],
            implications=[],
            connections=[],
            confidence_assessment="Analysis failed due to technical error.",
            recommendations=[],
        )
        return {
            "analyst_output": fallback.model_dump(),
            "current_agent": "analyst",
            "agent_status": {**state.agent_status, "analyst": "done"},
        }