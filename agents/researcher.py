"""
agents/researcher.py
────────────────────
Researcher agent node — the information gathering engine.

Responsibilities:
1. Decompose query into targeted sub-queries
2. Execute web search via Tavily
3. Execute private document RAG retrieval
4. Optionally scrape key URLs for full content
5. Return structured ResearchOutput
"""

from __future__ import annotations
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from core.config import settings
from core.state import ResearchState
from schemas.outputs import ResearchOutput, Source
from prompts.researcher_prompt import RESEARCHER_SYSTEM_PROMPT
from tools.search_tool import search_web, search_to_sources, multi_query_search
from tools.rag_tool import rag_search, format_rag_context, docs_to_sources
from tools.scraper_tool import scrape_multiple
from utils.logger import get_logger
from utils.helpers import truncate_to_tokens, combine_findings

logger = get_logger(__name__)


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        api_key=settings.openai_api_key,
        max_tokens=settings.max_tokens_per_agent,
    )


def researcher_node(state: ResearchState) -> dict:
    """LangGraph node: Execute comprehensive research."""
    logger.info("[agent.researcher]🔬 Researcher Agent starting...")

    query = state.query
    critique_instructions = ""

    # If this is a re-research loop, extract critic's improvement instructions
    if state.critique_log:
        latest_critique = state.critique_log[-1]
        critique_instructions = f"\n\nPREVIOUS CRITIC FEEDBACK (must address):\n{latest_critique}"
        logger.info(f"   → Re-research loop #{state.critique_loop_count}")

    # ── Step 1: Generate sub-queries ─────────────────────────────────────────
    llm = get_llm()
    subquery_prompt = f"""Given this research query: "{query}"
{critique_instructions}

Generate 4-5 specific search sub-queries to comprehensively cover this topic.
Return ONLY a JSON array of strings. Example: ["query1", "query2", "query3"]
Focus on: main topic, key stakeholders, recent developments, data/statistics, opposing views."""

    try:
        subquery_response = llm.invoke([HumanMessage(content=subquery_prompt)])
        raw = subquery_response.content.strip()
        # Parse JSON array
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        sub_queries: list[str] = json.loads(raw)
        logger.info(f"   → Generated {len(sub_queries)} sub-queries")
    except Exception as e:
        logger.warning(f"Sub-query generation failed: {e}, using original query")
        sub_queries = [query]

    # ── Step 2: Web search ────────────────────────────────────────────────────
    all_web_results, web_sources = multi_query_search(
        sub_queries,
        max_results_per_query=max(2, settings.max_search_results // len(sub_queries))
    )

    # Optionally scrape top URLs for richer content
    top_urls = [r["url"] for r in all_web_results[:4] if r.get("url")]
    scraped = scrape_multiple(top_urls, max_per_url=1500)

    # Build web content string
    web_content_parts = []
    for r in all_web_results:
        url = r.get("url", "")
        content = scraped.get(url, r.get("content", ""))
        if content:
            web_content_parts.append(
                f"**Source**: {r.get('title', 'Untitled')} ({url})\n{content}"
            )
    web_content = "\n\n---\n\n".join(web_content_parts)

    # ── Step 3: RAG search ────────────────────────────────────────────────────
    rag_docs = rag_search(query, k=5)
    rag_context = format_rag_context(rag_docs)
    rag_sources = docs_to_sources(rag_docs)

    # ── Step 4: Synthesize with LLM ───────────────────────────────────────────
    llm_structured = llm.with_structured_output(ResearchOutput)

    research_prompt = f"""Research Query: {query}
{critique_instructions}

=== WEB SEARCH RESULTS ===
{truncate_to_tokens(web_content, 2500)}

=== PRIVATE DOCUMENTS (RAG) ===
{truncate_to_tokens(rag_context, 1000)}

Based on all the above information, provide a comprehensive structured research output.
Include ALL sources consulted. Identify gaps that need further investigation."""

    try:
        research_output: ResearchOutput = llm_structured.invoke([
            SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
            HumanMessage(content=research_prompt),
        ])
        logger.info(
            f"   → Research complete: {len(research_output.web_findings)} web findings, "
            f"{len(research_output.rag_findings)} RAG findings"
        )
    except Exception as e:
        logger.error(f"Researcher LLM call failed: {e}")
        # Create a minimal fallback output
        research_output = ResearchOutput(
            query_understood=query,
            web_findings=[r.get("content", "")[:200] for r in all_web_results[:5]],
            rag_findings=[doc.page_content[:200] for doc in rag_docs[:3]],
            sources=web_sources + rag_sources,
            research_gaps=["LLM synthesis failed — raw results provided"],
            raw_content=web_content[:3000],
        )

    # Merge all sources
    all_sources = web_sources + rag_sources
    # Deduplicate research_output.sources with collected sources
    for src in research_output.sources:
        key = src.url or src.title
        if not any((s.url or s.title) == key for s in all_sources):
            all_sources.append(src)

    return {
        "research_output": research_output.model_dump(),
        "all_sources": all_sources,
        "current_agent": "researcher",
        "agent_status": {**state.agent_status, "researcher": "done"},
    }