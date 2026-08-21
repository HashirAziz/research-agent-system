"""
tools/search_tool.py
────────────────────
Tavily-powered web search using the tavily-python client directly.
"""

from __future__ import annotations
from typing import Any
import os

from core.config import settings
from schemas.outputs import Source
from utils.logger import get_logger

logger = get_logger(__name__)


def search_web(query: str, max_results: int | None = None) -> list[dict[str, Any]]:
    """Execute a web search and return normalized list of dicts."""
    logger.info(
        f"🔍 Web search: '{query[:80]}...'"
        if len(query) > 80
        else f"🔍 Web search: '{query}'"
    )
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query=query,
            max_results=max_results or settings.max_search_results,
            include_answer=True,
            include_raw_content=False,
            search_depth="advanced",
        )

        results = []

        if response.get("answer"):
            results.append({
                "title": "Tavily Direct Answer",
                "url": "",
                "content": response["answer"],
                "score": 1.0,
            })

        for r in response.get("results", []):
            if isinstance(r, dict):
                results.append({
                    "title": r.get("title", "Untitled"),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0.5),
                })

        logger.info(f"   → {len(results)} results returned")
        return results

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def search_to_sources(results: list[dict]) -> list[Source]:
    """Convert raw search results to Source objects."""
    sources = []
    for r in results:
        if not isinstance(r, dict) or not r.get("content"):
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
    """Run multiple search queries and aggregate deduplicated results."""
    all_results: list[dict] = []
    all_sources: list[Source] = []
    seen_urls: set[str] = set()

    for q in queries:
        results = search_web(q, max_results=max_results_per_query)
        for r in results:
            if not isinstance(r, dict):
                continue
            url = r.get("url", "")
            if url not in seen_urls:
                all_results.append(r)
                seen_urls.add(url)
        all_sources.extend(search_to_sources(results))

    logger.info(
        f"Multi-query search: {len(queries)} queries → {len(all_results)} unique results"
    )
    return all_results, all_sources