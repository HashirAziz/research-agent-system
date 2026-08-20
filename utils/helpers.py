"""
utils/helpers.py
────────────────
Utility functions: citation formatting, text chunking, token counting, etc.
"""

from __future__ import annotations
import re
import hashlib
from datetime import datetime
from typing import Any
import tiktoken
from schemas.outputs import Source


def format_citation(source: Source, index: int) -> str:
    """Return a Markdown footnote-style citation string."""
    if source.url:
        return f"[{index}] [{source.title}]({source.url})"
    return f"[{index}] {source.title} (private document)"


def build_bibliography(sources: list[Source]) -> str:
    """Build a numbered Markdown bibliography from a list of sources."""
    lines = ["## References\n"]
    for i, src in enumerate(sources, 1):
        lines.append(format_citation(src, i))
        if src.snippet:
            lines.append(f"   > {src.snippet[:200]}...")
        lines.append("")
    return "\n".join(lines)


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens for a given text using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        # Fallback: rough estimate
        return len(text.split()) * 4 // 3


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4o") -> str:
    """Truncate text to a maximum token count."""
    try:
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens]) + "\n\n[... truncated ...]"
    except Exception:
        # Fallback character-based truncation
        approx_chars = max_tokens * 3
        return text[:approx_chars] + "\n\n[... truncated ...]"


def sanitize_filename(name: str) -> str:
    """Convert a query string into a safe filename."""
    name = re.sub(r"[^\w\s-]", "", name.lower())
    name = re.sub(r"[\s_]+", "_", name).strip("_")
    return name[:80]


def generate_report_id(query: str) -> str:
    """Generate a short unique ID for a report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_hash = hashlib.md5(query.encode()).hexdigest()[:6]
    return f"{timestamp}_{query_hash}"


def format_agent_status(status_dict: dict[str, str]) -> str:
    """Format the agent status dict into a readable string."""
    icons = {
        "pending": "⏳",
        "running": "🔄",
        "done":    "✅",
        "error":   "❌",
        "skipped": "⏭️",
    }
    lines = []
    for agent, status in status_dict.items():
        icon = icons.get(status, "❓")
        lines.append(f"{icon} **{agent.replace('_', ' ').title()}**: {status}")
    return "\n".join(lines)


def combine_findings(web: list[str], rag: list[str]) -> str:
    """Combine web and RAG findings into a single formatted block."""
    sections = []
    if web:
        sections.append("### Web Research Findings\n" + "\n".join(f"- {f}" for f in web))
    if rag:
        sections.append("### Private Document Findings\n" + "\n".join(f"- {f}" for f in rag))
    return "\n\n".join(sections)