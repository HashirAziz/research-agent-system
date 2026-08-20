"""
agents/summarizer.py
────────────────────
Summarizer agent — distills research + fact-check into structured summary.
"""

from __future__ import annotations
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from core.config import settings
from core.state import ResearchState
from schemas.outputs import SummarizerOutput
from prompts.summarizer_prompt import SUMMARIZER_SYSTEM_PROMPT
from utils.logger import get_logger
from utils.helpers import combine_findings

logger = get_logger(__name__)


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=0.3,
        api_key=settings.openai_api_key,
        max_tokens=settings.max_tokens_per_agent,
    )


def summarizer_node(state: ResearchState) -> dict:
    """LangGraph node: Summarize research findings."""
    logger.info("[agent.summarizer]📋 Summarizer Agent starting...")

    research = state.research_output
    fact_check = state.fact_check_output

    if not research:
        logger.warning("No research to summarize.")
        return {
            "summarizer_output": None,
            "current_agent": "summarizer",
            "agent_status": {**state.agent_status, "summarizer": "skipped"},
        }

    # Combine context
    web_findings = research.get("web_findings", [])
    rag_findings = research.get("rag_findings", [])
    combined = combine_findings(web_findings, rag_findings)

    verified_facts = ""
    flagged_issues = ""
    if fact_check:
        verified = "\n".join(f"✓ {f}" for f in fact_check.get("verified_facts", []))
        issues = "\n".join(f"⚠ {i}" for i in fact_check.get("flagged_issues", []))
        verified_facts = f"\n\nVERIFIED FACTS:\n{verified}" if verified else ""
        flagged_issues = f"\n\nFLAGGED ISSUES:\n{issues}" if issues else ""

    summary_prompt = f"""Research Query: {state.query}

{combined}
{verified_facts}
{flagged_issues}

Research gaps identified: {', '.join(research.get('research_gaps', ['None identified']))}

Please provide a comprehensive, structured summary of all findings.
Highlight the most important insights for decision-makers."""

    llm = get_llm().with_structured_output(SummarizerOutput)

    try:
        summary: SummarizerOutput = llm.invoke([
            SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT),
            HumanMessage(content=summary_prompt),
        ])
        logger.info(
            f"   → {len(summary.key_points)} key points | "
            f"{len(summary.themes)} themes | "
            f"{len(summary.contradictions)} contradictions"
        )
    except Exception as e:
        logger.error(f"Summarizer LLM call failed: {e}")
        summary = SummarizerOutput(
            executive_summary=f"Research on '{state.query}' completed with {len(web_findings)} web findings.",
            key_points=web_findings[:7],
            themes=[],
            contradictions=[],
            data_points=[],
        )

    return {
        "summarizer_output": summary.model_dump(),
        "current_agent": "summarizer",
        "agent_status": {**state.agent_status, "summarizer": "done"},
    }