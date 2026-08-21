"""
agents/report_writer.py
───────────────────────
Report Writer agent — produces the final professional Markdown report.
"""

from __future__ import annotations
import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from core.config import settings
from core.state import ResearchState
from schemas.outputs import ReportOutput, ReportSection
from prompts.report_writer_prompt import REPORT_WRITER_SYSTEM_PROMPT
from utils.logger import get_logger
from utils.helpers import build_bibliography, sanitize_filename, generate_report_id

logger = get_logger(__name__)


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=0.3,
        api_key=settings.openai_api_key,
        max_tokens=4000,
    )


def save_report(markdown: str, query: str) -> str:
    os.makedirs(settings.reports_dir, exist_ok=True)
    report_id = generate_report_id(query)
    filename = f"{sanitize_filename(query[:40])}_{report_id}.md"
    filepath = os.path.join(settings.reports_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown)
    logger.info(f"💾 Report saved to: {filepath}")
    return filepath


def report_writer_node(state: ResearchState) -> dict:
    logger.info("[agent.writer]✍️  Report Writer Agent starting...")

    research = state.research_output
    fact_check = state.fact_check_output
    summary = state.summarizer_output
    analysis = state.analyst_output
    critique = state.critic_output

    bibliography = build_bibliography(state.all_sources)

    source_list = "\n".join(
        f"[Source {i+1}] {s.title} — {s.url or 'Private Document'}"
        for i, s in enumerate(state.all_sources[:20])
    )

    key_points = "\n".join(f"• {p}" for p in (summary or {}).get("key_points", []))
    implications = "\n".join(f"• {i}" for i in (analysis or {}).get("implications", []))
    recommendations = "\n".join(f"• {r}" for r in (analysis or {}).get("recommendations", []))
    root_causes = "\n".join(f"• {r}" for r in (analysis or {}).get("root_causes", []))
    verified_facts = "\n".join(f"✓ {f}" for f in (fact_check or {}).get("verified_facts", []))

    q_score = float((critique or {}).get("quality_score", 0.0))
    c_score = float((critique or {}).get("coverage_score", 0.0))
    a_score = float((critique or {}).get("accuracy_score", 0.0))
    d_score = float((critique or {}).get("depth_score", 0.0))

    report_prompt = f"""Research Query: {state.query}

=== EXECUTIVE SUMMARY ===
{(summary or {}).get('executive_summary', 'See key findings below.')}

=== KEY FINDINGS ===
{key_points or 'No key points available'}

=== VERIFIED FACTS ===
{verified_facts or 'Fact-checking not available'}

=== ROOT CAUSES ===
{root_causes or 'Not analyzed'}

=== IMPLICATIONS ===
{implications or 'Not analyzed'}

=== RECOMMENDATIONS ===
{recommendations or 'Not available'}

=== SOURCES FOR CITATION ===
{source_list}

=== QUALITY SCORES ===
Quality: {q_score}/10 | Coverage: {c_score}/10 | Accuracy: {a_score}/10 | Depth: {d_score}/10

DATE: {datetime.now().strftime('%B %d, %Y')}

Write a comprehensive professional research report using [Source N] citations throughout.
The full_markdown field must contain the COMPLETE formatted Markdown report."""

    llm = get_llm().with_structured_output(ReportOutput, method="function_calling")

    try:
        report: ReportOutput = llm.invoke([
            SystemMessage(content=REPORT_WRITER_SYSTEM_PROMPT),
            HumanMessage(content=report_prompt),
        ])

        if bibliography not in report.full_markdown:
            report.full_markdown += f"\n\n---\n\n{bibliography}"

        quality_badge = (
            f"\n\n---\n*Quality Score: {q_score}/10 | "
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
            f"Sources: {len(state.all_sources)}*"
        )
        report.full_markdown += quality_badge

        logger.info(
            f"   → Report: '{report.title}' | "
            f"~{report.word_count} words | "
            f"{len(report.sections)} sections"
        )
        save_report(report.full_markdown, state.query)

        return {
            "report_output": report.model_dump(),
            "final_report_markdown": report.full_markdown,
            "current_agent": "report_writer",
            "agent_status": {**state.agent_status, "report_writer": "done"},
        }

    except Exception as e:
        logger.error(f"Report Writer LLM call failed: {e}")
        fallback_md = f"# Research Report: {state.query}\n\n"
        fallback_md += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
        fallback_md += f"## Executive Summary\n{(summary or {}).get('executive_summary', 'N/A')}\n\n"
        fallback_md += f"## Key Findings\n{key_points}\n\n"
        fallback_md += f"\n\n---\n\n{bibliography}"

        save_report(fallback_md, state.query)

        fallback_report = ReportOutput(
            title=f"Research Report: {state.query}",
            abstract=(summary or {}).get("executive_summary", ""),
            sections=[ReportSection(title="Findings", content=key_points, citations=[])],
            conclusion="Report generation encountered errors.",
            bibliography=[bibliography],
            quality_score=q_score,
            coverage_score=c_score,
            accuracy_score=a_score,
            depth_score=d_score,
            word_count=len(fallback_md.split()),
            full_markdown=fallback_md,
        )

        return {
            "report_output": fallback_report.model_dump(),
            "final_report_markdown": fallback_md,
            "current_agent": "report_writer",
            "agent_status": {**state.agent_status, "report_writer": "done"},
        }