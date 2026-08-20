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
    """Save report to disk and return file path."""
    os.makedirs(settings.reports_dir, exist_ok=True)
    report_id = generate_report_id(query)
    filename = f"{sanitize_filename(query[:40])}_{report_id}.md"
    filepath = os.path.join(settings.reports_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown)
    logger.info(f"💾 Report saved to: {filepath}")
    return filepath


def report_writer_node(state: ResearchState) -> dict:
    """LangGraph node: Write the final structured research report."""
    logger.info("[agent.writer]✍️  Report Writer Agent starting...")

    research = state.research_output
    fact_check = state.fact_check_output
    summary = state.summarizer_output
    analysis = state.analyst_output
    critique = state.critic_output

    # Build bibliography from all collected sources
    bibliography = build_bibliography(state.all_sources)

    # Compile source list for citation reference
    source_list = "\n".join(
        f"[Source {i+1}] {s.title} — {s.url or 'Private Document'}"
        for i, s in enumerate(state.all_sources[:20])
    )

    # Build comprehensive context for the report writer
    key_points = "\n".join(f"• {p}" for p in (summary or {}).get("key_points", []))
    implications = "\n".join(f"• {i}" for i in (analysis or {}).get("implications", []))
    recommendations = "\n".join(f"• {r}" for r in (analysis or {}).get("recommendations", []))
    root_causes = "\n".join(f"• {r}" for r in (analysis or {}).get("root_causes", []))
    verified_facts = "\n".join(f"✓ {f}" for f in (fact_check or {}).get("verified_facts", []))

    quality_metadata = {}
    if critique:
        quality_metadata = {
            "quality_score": critique.get("quality_score"),
            "coverage_score": critique.get("coverage_score"),
            "accuracy_score": critique.get("accuracy_score"),
            "depth_score": critique.get("depth_score"),
        }

    report_prompt = f"""Research Query: {state.query}

=== EXECUTIVE SUMMARY ===
{(summary or {}).get('executive_summary', 'See key findings below.')}

=== KEY FINDINGS ===
{key_points or 'No key points available'}

=== VERIFIED FACTS ===
{verified_facts or 'Fact-checking not available'}

=== ANALYTICAL INSIGHTS ===
Root Causes:
{root_causes or 'Not analyzed'}

Implications:
{implications or 'Not analyzed'}

Recommendations:
{recommendations or 'Not available'}

=== SOURCES AVAILABLE FOR CITATION ===
{source_list}

=== QUALITY SCORES ===
Quality: {quality_metadata.get('quality_score', 'N/A')}/10
Coverage: {quality_metadata.get('coverage_score', 'N/A')}/10
Accuracy: {quality_metadata.get('accuracy_score', 'N/A')}/10
Depth: {quality_metadata.get('depth_score', 'N/A')}/10

CRITIQUE LOOPS: {state.critique_loop_count}
DATE: {datetime.now().strftime('%B %d, %Y')}

Write a comprehensive, professional research report. Use [Source N] citations throughout.
The full_markdown field must contain the COMPLETE formatted Markdown report."""

    llm = get_llm().with_structured_output(ReportOutput)

    try:
        report: ReportOutput = llm.invoke([
            SystemMessage(content=REPORT_WRITER_SYSTEM_PROMPT),
            HumanMessage(content=report_prompt),
        ])

        # Ensure bibliography is appended to full_markdown
        if bibliography not in report.full_markdown:
            report.full_markdown += f"\n\n---\n\n{bibliography}"

        # Add quality badge to report
        quality_badge = f"\n\n---\n*Quality Score: {quality_metadata.get('quality_score', 'N/A')}/10 | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Sources: {len(state.all_sources)}*"
        report.full_markdown += quality_badge

        logger.info(
            f"   → Report written: '{report.title}' | "
            f"~{report.word_count} words | "
            f"{len(report.sections)} sections"
        )

        # Save to disk
        report_path = save_report(report.full_markdown, state.query)

    except Exception as e:
        logger.error(f"Report Writer LLM call failed: {e}")
        # Fallback minimal report
        fallback_md = f"# Research Report: {state.query}\n\n"
        fallback_md += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
        fallback_md += f"## Executive Summary\n{(summary or {}).get('executive_summary', 'N/A')}\n\n"
        fallback_md += f"## Key Findings\n{key_points}\n\n"
        fallback_md += f"\n\n---\n\n{bibliography}"

        report = ReportOutput(
            title=f"Research Report: {state.query}",
            abstract=(summary or {}).get("executive_summary", ""),
            sections=[ReportSection(title="Findings", content=key_points, citations=[])],
            conclusion="Report generation encountered errors.",
            bibliography=[bibliography],
            quality_metadata=quality_metadata,
            word_count=len(fallback_md.split()),
            full_markdown=fallback_md,
        )
        save_report(fallback_md, state.query)

    return {
        "report_output": report.model_dump(),
        "final_report_markdown": report.full_markdown,
        "current_agent": "report_writer",
        "agent_status": {**state.agent_status, "report_writer": "done"},
    }