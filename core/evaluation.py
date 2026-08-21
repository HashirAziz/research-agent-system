"""
core/evaluation.py
──────────────────
Quality scoring for a completed research run.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    overall_score: float
    quality_score: float
    coverage_score: float
    accuracy_score: float
    depth_score: float
    source_diversity: float
    fact_verification_rate: float
    critique_loops: int
    source_count: int
    word_count: int
    grade: str
    summary: str

    def to_dict(self) -> dict:
        return self.__dict__


def score_to_grade(score: float) -> str:
    if score >= 9.0: return "A+"
    if score >= 8.5: return "A"
    if score >= 8.0: return "A-"
    if score >= 7.5: return "B+"
    if score >= 7.0: return "B"
    if score >= 6.5: return "B-"
    if score >= 6.0: return "C+"
    if score >= 5.0: return "C"
    return "D"


def compute_source_diversity(sources: list) -> float:
    if not sources:
        return 0.0
    total = len(sources)
    web_sources = [s for s in sources if (s.get("source_type") if isinstance(s, dict) else getattr(s, "source_type", "")) == "web"]
    rag_sources = [s for s in sources if (s.get("source_type") if isinstance(s, dict) else getattr(s, "source_type", "")) == "rag"]

    domains = set()
    for s in web_sources:
        url = s.get("url", "") if isinstance(s, dict) else getattr(s, "url", "")
        if url:
            try:
                from urllib.parse import urlparse
                domains.add(urlparse(url).netloc)
            except Exception:
                pass

    domain_score = min(len(domains) / 5.0, 1.0)
    volume_score = min(total / 8.0, 1.0)
    mix_bonus = 0.5 if (web_sources and rag_sources) else 0.0
    return min(domain_score * 5 + volume_score * 3 + mix_bonus * 2, 10.0)


def compute_fact_verification_rate(fact_check: dict | None) -> float:
    if not fact_check:
        return 0.0
    checks = fact_check.get("checks", [])
    if not checks:
        return fact_check.get("overall_reliability", 0.5)
    verified = sum(1 for c in checks if c.get("verdict") == "verified")
    return verified / len(checks)


def evaluate_run(final_state: dict) -> EvaluationResult:
    critic = final_state.get("critic_output") or {}
    fact_check = final_state.get("fact_check_output") or {}
    report = final_state.get("report_output") or {}
    sources = final_state.get("all_sources", [])
    loops = final_state.get("critique_loop_count", 0)

    quality = float(critic.get("quality_score", 5.0))
    coverage = float(critic.get("coverage_score", 5.0))
    accuracy = float(critic.get("accuracy_score", 5.0))
    depth = float(critic.get("depth_score", 5.0))

    source_diversity = compute_source_diversity(sources)
    fact_rate = compute_fact_verification_rate(fact_check)
    loop_penalty = max(0.0, (loops - 1) * 0.3)

    composite = (
        quality * 0.35 +
        coverage * 0.20 +
        accuracy * 0.20 +
        depth * 0.15 +
        source_diversity * 0.10
    ) - loop_penalty
    composite = max(0.0, min(10.0, composite))

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
        grade=score_to_grade(composite),
        summary=(
            f"Research scored {composite:.1f}/10 ({score_to_grade(composite)}). "
            f"Coverage: {coverage:.1f} | Accuracy: {accuracy:.1f} | Depth: {depth:.1f}. "
            f"{len(sources)} sources, {loops} critique loop(s). "
            f"Fact verification: {fact_rate:.0%}."
        ),
    )