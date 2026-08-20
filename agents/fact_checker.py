"""
agents/fact_checker.py
──────────────────────
Fact-Checker agent — verifies claims from the Researcher output.
"""

from __future__ import annotations
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from core.config import settings
from core.state import ResearchState
from schemas.outputs import FactCheckOutput
from prompts.fact_checker_prompt import FACT_CHECKER_SYSTEM_PROMPT
from utils.logger import get_logger
from utils.helpers import truncate_to_tokens

logger = get_logger(__name__)


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=0.1,  # Low temp for fact-checking reliability
        api_key=settings.openai_api_key,
        max_tokens=settings.max_tokens_per_agent,
    )


def fact_checker_node(state: ResearchState) -> dict:
    """LangGraph node: Verify factual claims in the research."""
    logger.info("[agent.fact_checker]🔎 Fact-Checker Agent starting...")

    research = state.research_output
    if not research:
        logger.warning("No research output to fact-check.")
        return {
            "fact_check_output": None,
            "current_agent": "fact_checker",
            "agent_status": {**state.agent_status, "fact_checker": "skipped"},
        }

    # Build context from research output
    web_findings = "\n".join(f"- {f}" for f in research.get("web_findings", []))
    rag_findings = "\n".join(f"- {f}" for f in research.get("rag_findings", []))
    raw_content = research.get("raw_content", "")

    fact_check_prompt = f"""Research to verify:

QUERY: {state.query}

WEB FINDINGS:
{web_findings}

PRIVATE DOCUMENT FINDINGS:
{rag_findings}

RAW RESEARCH CONTENT (for cross-referencing):
{truncate_to_tokens(raw_content, 1500)}

SOURCES CONSULTED:
{chr(10).join(f"- {s.get('title', 'Unknown')} ({s.get('url', 'no URL')})" for s in research.get('sources', [])[:10])}

Please systematically verify all significant factual claims in these findings.
Pay particular attention to statistics, dates, attributions, and causal claims."""

    llm = get_llm().with_structured_output(FactCheckOutput)

    try:
        fact_check: FactCheckOutput = llm.invoke([
            SystemMessage(content=FACT_CHECKER_SYSTEM_PROMPT),
            HumanMessage(content=fact_check_prompt),
        ])

        logger.info(
            f"   → {len(fact_check.checks)} claims checked | "
            f"Reliability: {fact_check.overall_reliability:.2f} | "
            f"Issues: {len(fact_check.flagged_issues)}"
        )
    except Exception as e:
        logger.error(f"Fact-checker LLM call failed: {e}")
        from schemas.outputs import FactCheckOutput, FactCheckItem
        fact_check = FactCheckOutput(
            checks=[],
            overall_reliability=0.5,
            flagged_issues=[f"Fact-checking failed: {str(e)}"],
            verified_facts=[],
        )

    return {
        "fact_check_output": fact_check.model_dump(),
        "current_agent": "fact_checker",
        "agent_status": {**state.agent_status, "fact_checker": "done"},
    }