"""
schemas/outputs.py
──────────────────
Pydantic v2 structured output models for every agent.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class Source(BaseModel):
    title: str = Field(description="Title of the source document or webpage")
    url: Optional[str] = Field(None, description="URL if it's a web source")
    snippet: str = Field(description="Relevant excerpt from the source (max 300 chars)")
    source_type: str = Field(description="'web' | 'document' | 'rag'")


class ResearchOutput(BaseModel):
    query_understood: str = Field(description="Researcher's interpretation of the query")
    web_findings: list[str] = Field(description="Key findings from web search")
    rag_findings: list[str] = Field(description="Key findings from private documents")
    sources: list[Source] = Field(description="All sources consulted")
    research_gaps: list[str] = Field(description="Topics needing further investigation")
    raw_content: str = Field(description="Full combined raw research content")


class FactCheckItem(BaseModel):
    claim: str = Field(description="The specific claim being checked")
    verdict: str = Field(description="'verified' | 'disputed' | 'unverifiable'")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0-1")
    evidence: str = Field(description="Evidence supporting the verdict")
    correction: Optional[str] = Field(None, description="Corrected version if disputed")


class FactCheckOutput(BaseModel):
    checks: list[FactCheckItem] = Field(description="Individual fact checks")
    overall_reliability: float = Field(
        ge=0.0, le=1.0,
        description="Overall reliability score of the research (0-1)"
    )
    flagged_issues: list[str] = Field(description="Major reliability concerns")
    verified_facts: list[str] = Field(description="Confirmed accurate statements")


class SummarizerOutput(BaseModel):
    executive_summary: str = Field(description="3-5 sentence high-level summary")
    key_points: list[str] = Field(description="7-10 most important findings")
    themes: list[str] = Field(description="Major recurring themes identified")
    contradictions: list[str] = Field(description="Contradictions found in the research")
    data_points: list[str] = Field(description="Key statistics and quantitative data")


class ReasoningChain(BaseModel):
    step: int
    premise: str
    inference: str
    confidence: float = Field(ge=0.0, le=1.0)


class AnalystOutput(BaseModel):
    multi_hop_chains: list[ReasoningChain] = Field(
        description="Step-by-step reasoning chains connecting facts"
    )
    root_causes: list[str] = Field(description="Identified root causes or drivers")
    implications: list[str] = Field(description="Short and long-term implications")
    connections: list[str] = Field(description="Non-obvious connections between findings")
    confidence_assessment: str = Field(description="Overall confidence in the analysis")
    recommendations: list[str] = Field(description="Actionable recommendations")


class CriticOutput(BaseModel):
    quality_score: float = Field(
        ge=0.0, le=10.0,
        description="Overall quality score (0-10)."
    )
    coverage_score: float = Field(ge=0.0, le=10.0, description="Research coverage 0-10")
    accuracy_score: float = Field(ge=0.0, le=10.0, description="Factual accuracy 0-10")
    depth_score: float = Field(ge=0.0, le=10.0, description="Analytical depth 0-10")
    approved: bool = Field(description="True if quality meets the threshold")
    critical_gaps: list[str] = Field(description="Missing topics that must be addressed")
    improvement_instructions: str = Field(
        description="Specific instructions for re-research if not approved"
    )
    strengths: list[str] = Field(description="What the research did well")
    weaknesses: list[str] = Field(description="Specific weaknesses identified")


class ReportSection(BaseModel):
    title: str
    content: str
    citations: list[str] = Field(description="Citation keys used in this section")


class ReportOutput(BaseModel):
    title: str = Field(description="Professional report title")
    abstract: str = Field(description="150-200 word abstract")
    sections: list[ReportSection] = Field(description="All report sections")
    conclusion: str = Field(description="Concluding paragraphs")
    bibliography: list[str] = Field(description="Formatted bibliography entries")
    quality_score: float = Field(default=0.0, description="Overall quality score from Critic")
    coverage_score: float = Field(default=0.0, description="Coverage score from Critic")
    accuracy_score: float = Field(default=0.0, description="Accuracy score from Critic")
    depth_score: float = Field(default=0.0, description="Depth score from Critic")
    word_count: int = Field(description="Approximate word count")
    full_markdown: str = Field(description="Complete report as a single Markdown string")