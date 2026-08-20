"""
tools/search_tool.py
────────────────────
Tavily-powered web search tool.
Returns structured results with title, url, content, and score.
"""

from __future__ import annotations
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from typing import Any
import os

from core.config import settings
from schemas.outputs import Source
from utils.logger import get_logger

logger = get_logger(__name__)


def get_tavily_tool(max_results: int | None = None) -> TavilySearchResults:
    """Return a configured TavilySearchResults tool."""
    os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
    return TavilySearchResults(
        max_results=max_results or settings.max_search_results,
        include_answer=True,
        include_raw_content=True,
        include_images=False,
        search_depth="advanced",
    )


def search_web(query: str, max_results: int | None = None) -> list[dict[str, Any]]:
    """
    Execute a web search and return raw results.
    
    Returns:
        List of dicts with keys: title, url, content, score
    """
    logger.info(f"🔍 Web search: '{query[:80]}...' " if len(query) > 80 else f"🔍 Web search: '{query}'")
    
    try:
        tavily = get_tavily_tool(max_results)
        results = tavily.invoke(query)
        
        if isinstance(results, str):
            # Tavily sometimes returns a string answer
            return [{"title": "Tavily Answer", "url": "", "content": results, "score": 1.0}]
        
        logger.info(f"   → {len(results)} results returned")
        return results
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def search_to_sources(results: list[dict]) -> list[Source]:
    """Convert raw Tavily results to Source objects."""
    sources = []
    for r in results:
        if not r.get("content"):
            continue
        sources.append(Source(
            title=r.get("title", "Untitled"),
            url=r.get("url", ""),
            snippet=r.get("content", "")[:300],
            source_type="web",
        ))
    return sources


def multi_query_search(
    queries: list[str], max_results_per_query: int = 4
) -> tuple[list[dict], list[Source]]:
    """
    Run multiple search queries and aggregate results.
    Useful for multi-hop reasoning where sub-questions need answers.
    """
    all_results: list[dict] = []
    all_sources: list[Source] = []
    seen_urls: set[str] = set()

    for q in queries:
        results = search_web(q, max_results=max_results_per_query)
        for r in results:
            url = r.get("url", "")
            if url not in seen_urls:
                all_results.append(r)
                seen_urls.add(url)

        all_sources.extend(search_to_sources(results))

    logger.info(f"Multi-query search: {len(queries)} queries → {len(all_results)} unique results")
    return all_results, all_sources