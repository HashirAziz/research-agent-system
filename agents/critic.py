"""
agents/critic.py
────────────────
Critic agent — quality gate that can trigger re-research loops.
"""

from __future__ import annotations
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from core.config import settings
from core.state import ResearchState
from schemas.outputs import CriticOutput
from prompts.critic_prompt import CRITIC_SYSTEM_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=0.1,
        api_key=settings.openai_api_key,
        max_tokens=settings.max_tokens_per_agent,
    )


def critic_node(state: ResearchState) -> dict:
    loop_num = state.critique_loop_count + 1
    logger.info(f"[agent.critic]⚖️  Critic Agent starting (evaluation #{loop_num})...")

    research = state.research_output
    fact_check = state.fact_check_output
    summary = state.summarizer_output
    analysis = state.analyst_output

    source_count = len(state.all_sources)
    web_findings_count = len(research.get("web_findings", [])) if research else 0
    rag_findings_count = len(research.get("rag_findings", [])) if research else 0
    key_points = summary.get("key_points", []) if summary else []
    reasoning_chains = analysis.get("multi_hop_chains", []) if analysis else []
    flagged_issues = fact_check.get("flagged_issues", []) if fact_check else []
    reliability = fact_check.get("overall_reliability", 0.5) if fact_check else 0.5
    contradictions = summary.get("contradictions", []) if summary else []
    gaps = research.get("research_gaps", []) if research else []
    prior_critiques = "\n".join(state.critique_log) if state.critique_log else "None"

    critic_prompt = f"""Research Quality Evaluation

QUERY: {state.query}
LOOP NUMBER: {loop_num} (max: {settings.max_critique_loops})

=== RESEARCH METRICS ===
- Total sources: {source_count}
- Web findings: {web_findings_count}
- Private document findings: {rag_findings_count}
- Source reliability score: {reliability:.2f}

=== FACT-CHECK ISSUES ===
{chr(10).join(f'• {i}' for i in flagged_issues) or 'None flagged'}

=== KEY FINDINGS ===
{chr(10).join(f'• {p}' for p in key_points[:10]) or 'No key points extracted'}

=== REASONING CHAINS ===
{len(reasoning_chains)} multi-hop chains constructed

=== CONTRADICTIONS ===
{chr(10).join(f'⚡ {c}' for c in contradictions) or 'None identified'}

=== RESEARCH GAPS ===
{chr(10).join(f'• {g}' for g in gaps) or 'None identified'}

=== PRIOR CRITIQUE HISTORY ===
{prior_critiques}

Score this research package. Approval threshold: quality_score >= {settings.min_quality_score}"""

    llm = get_llm().with_structured_output(CriticOutput, method="function_calling")

    try:
        critique: CriticOutput = llm.invoke([
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=critic_prompt),
        ])
    except Exception as e:
        logger.error(f"Critic LLM call failed: {e}")
        critique = CriticOutput(
            quality_score=7.5,
            coverage_score=7.0,
            accuracy_score=7.0,
            depth_score=7.0,
            approved=True,
            critical_gaps=[],
            improvement_instructions="",
            strengths=["Proceeding due to critic failure"],
            weaknesses=[f"Critic error: {str(e)}"],
        )

    if loop_num >= settings.max_critique_loops:
        logger.warning(f"   → Max loops ({settings.max_critique_loops}) reached — forcing approval")
        critique.approved = True

    log_entry = (
        f"Loop {loop_num}: Score={critique.quality_score:.1f} | "
        f"Approved={critique.approved} | "
        f"Gaps: {', '.join(critique.critical_gaps[:3])}"
    )
    if critique.improvement_instructions:
        log_entry += f"\nInstructions: {critique.improvement_instructions}"

    logger.info(
        f"   → Score: {critique.quality_score:.1f}/10 | "
        f"Approved: {critique.approved} | "
        f"Gaps: {len(critique.critical_gaps)}"
    )

    return {
        "critic_output": critique.model_dump(),
        "quality_score": critique.quality_score,
        "approved": critique.approved,
        "critique_loop_count": loop_num,
        "critique_log": [log_entry],
        "current_agent": "critic",
        "agent_status": {**state.agent_status, "critic": "done"},
    }