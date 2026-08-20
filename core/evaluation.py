"""
core/evaluation.py
──────────────────
Basic evaluation and quality scoring utilities.

Evaluation dimensions:
1. Source diversity score  — are sources varied?
2. Coverage score          — from Critic
3. Fact verification rate  — % of claims verified
4. Report completeness     — sections present?
5. Overall composite score
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationResult:
    """Aggregated evaluation result for a research run."""
    overall_score: float          # 0-10
    quality_score: float          # From critic (0-10)
    coverage_score: float         # From critic (0-10)
    accuracy_score: float         # From critic (0-10)
    depth_score: float            # From critic (0-10)
    source_diversity: float       # Computed (0-10)
    fact_verification_rate: float # % of facts verified (0-1)
    critique_loops: int           # How many loops were needed
    source_count: int             # Total unique sources
    word_count: int               # Report word count
    grade: str                    # Letter grade
    summary: str                  # Human-readable evaluation summary

    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "quality_score": self.quality_score,
            "coverage_score": self.coverage_score,
            "accuracy_score": self.accuracy_score,
            "depth_score": self.depth_score,
            "source_diversity": self.source_diversity,
            "fact_verification_rate": self.fact_verification_rate,
            "critique_loops": self.critique_loops,
            "source_count": self.source_count,
            "word_count": self.word_count,
            "grade": self.grade,
            "summary": self.summary,
        }


def score_to_grade(score: float) -> str:
    """Convert a 0-10 score to letter grade."""
    if score >= 9.0: return "A+"
    if score >= 8.5: return "A"
    if score >= 8.0: return "A-"
    if score >= 7.5: return "B+"
    if score >= 7.0: return "B"
    if score >= 6.5: return "B-"
    if score >= 6.0: return "C+"
    if score >= 5.0: return "C"
    return "D"


def compute_source_diversity(sources: list[dict]) -> float:
    """
    Score source diversity 0-10.
    Rewards: multiple unique domains, mix of web + RAG, variety.
    """
    if not sources:
        return 0.0

    total = len(sources)
    web_sources = [s for s in sources if s.get("source_type") == "web"]
    rag_sources = [s for s in sources if s.get("source_type") == "rag"]

    # Unique domains
    domains = set()
    for s in web_sources:
        url = s.get("url", "")
        if url:
            from urllib.parse import urlparse
            try:
                domains.add(urlparse(url).netloc)
            except Exception:
                pass

    domain_score = min(len(domains) / 5.0, 1.0)  # 5+ unique domains = full score
    volume_score = min(total / 8.0, 1.0)           # 8+ sources = full score
    mix_bonus = 0.5 if (web_sources and rag_sources) else 0.0

    raw = (domain_score * 5 + volume_score * 3 + mix_bonus * 2)
    return min(raw, 10.0)


def compute_fact_verification_rate(fact_check: dict | None) -> float:
    """Compute percentage of claims that were verified."""
    if not fact_check:
        return 0.0

    checks = fact_check.get("checks", [])
    if not checks:
        return fact_check.get("overall_reliability", 0.5)

    verified = sum(1 for c in checks if c.get("verdict") == "verified")
    return verified / len(checks)


def evaluate_run(final_state: dict) -> EvaluationResult:
    """
    Compute a comprehensive evaluation from the final graph state.
    
    Args:
        final_state: The final state dict from graph.invoke()
    
    Returns:
        EvaluationResult with all scores
    """
    critic = final_state.get("critic_output") or {}
    fact_check = final_state.get("fact_check_output") or {}
    report = final_state.get("report_output") or {}
    sources = final_state.get("all_sources", [])
    loops = final_state.get("critique_loop_count", 0)

    # Scores from critic
    quality = critic.get("quality_score", 5.0)
    coverage = critic.get("coverage_score", 5.0)
    accuracy = critic.get("accuracy_score", 5.0)
    depth = critic.get("depth_score", 5.0)

    # Computed scores
    source_diversity = compute_source_diversity(
        [s.model_dump() if hasattr(s, "model_dump") else s for s in sources]
    )
    fact_rate = compute_fact_verification_rate(fact_check)

    # Loop penalty: each extra loop beyond 1 reduces score slightly
    loop_penalty = max(0.0, (loops - 1) * 0.3)

    # Composite score
    composite = (
        quality * 0.35 +
        coverage * 0.20 +
        accuracy * 0.20 +
        depth * 0.15 +
        source_diversity * 0.10
    ) - loop_penalty

    composite = max(0.0, min(10.0, composite))
    grade = score_to_grade(composite)

    summary = (
        f"Research scored {composite:.1f}/10 ({grade}). "
        f"Coverage: {coverage:.1f}/10 | Accuracy: {accuracy:.1f}/10 | "
        f"Depth: {depth:.1f}/10. "
        f"Used {len(sources)} sources across {loops} critique loop(s). "
        f"Fact verification rate: {fact_rate:.0%}."
    )

    return EvaluationResult(
        overall_score=round(composite, 2),
        quality_score=quality,
        coverage_score=coverage,
        accuracy_score=accuracy,
        depth_score=depth,
        source_diversity=round(source_diversity, 2),
        fact_verification_rate=round(fact_rate, 3),
        critique_loops=loops,
        source_count=len(sources),
        word_count=report.get("word_count", 0),
        grade=grade,
        summary=summary,
    )