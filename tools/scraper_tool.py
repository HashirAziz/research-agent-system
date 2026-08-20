"""
tools/scraper_tool.py
─────────────────────
Lightweight URL content scraper using requests + BeautifulSoup.
Used by Researcher to get full content beyond Tavily snippets.
"""

from __future__ import annotations
import re
import requests
from bs4 import BeautifulSoup
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def scrape_url(url: str, max_chars: int = 3000) -> Optional[str]:
    """
    Scrape and clean text content from a URL.
    
    Returns:
        Cleaned text content, or None on failure.
    """
    if not url or not url.startswith("http"):
        return None

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "iframe", "noscript", "form"]):
            tag.decompose()

        # Extract main content — prefer semantic tags
        main = (
            soup.find("article") or
            soup.find("main") or
            soup.find("div", {"id": re.compile(r"content|article|post", re.I)}) or
            soup.find("body")
        )

        text = main.get_text(separator="\n", strip=True) if main else ""

        # Clean whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = text.strip()

        logger.info(f"🌐 Scraped {len(text):,} chars from {url[:60]}...")
        return text[:max_chars]

    except requests.exceptions.Timeout:
        logger.warning(f"Scrape timeout: {url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Scrape failed ({url}): {e}")
        return None


def scrape_multiple(urls: list[str], max_per_url: int = 2000) -> dict[str, str]:
    """Scrape multiple URLs, returning {url: content} dict."""
    results = {}
    for url in urls[:5]:  # Limit to 5 URLs to avoid delays
        content = scrape_url(url, max_chars=max_per_url)
        if content:
            results[url] = content
    return results