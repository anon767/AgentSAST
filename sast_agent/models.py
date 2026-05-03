"""Core data models for the SAST agent system."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class HypothesisStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    CONFIRMED = "confirmed"
    DISPROVEN = "disproven"
    UNCERTAIN = "uncertain"


class FindingSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Planner models
# ---------------------------------------------------------------------------

class Hypothesis(BaseModel):
    """A single security hypothesis produced by the planner."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    hypothesis: str = Field(description="Human-readable description of the potential vulnerability")
    risk_area: str = Field(description="Category: sqli, xss, ssrf, idor, auth_bypass, rce, path_traversal, etc.")
    priority: RiskLevel = RiskLevel.MEDIUM
    entrypoints: list[str] = Field(default_factory=list, description="Routes, handlers, RPC methods")
    files: list[str] = Field(default_factory=list, description="Files to inspect")
    suggested_queries: list[str] = Field(default_factory=list, description="CodeQL queries or search patterns")
    token_budget: int = Field(default=4000, description="Max tokens the analyzer should spend")
    status: HypothesisStatus = HypothesisStatus.PENDING


class PlannerOutput(BaseModel):
    """Structured output from the planner agent."""

    repo_summary: str = Field(description="Brief summary of the repository / PR")
    attack_surface: list[str] = Field(default_factory=list, description="Identified attack surface areas")
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Analyzer models
# ---------------------------------------------------------------------------

class Evidence(BaseModel):
    """A piece of evidence supporting or refuting a hypothesis."""

    description: str
    file: str = ""
    line_range: str = ""
    snippet: str = ""
    tool_used: str = ""


class AnalyzerOutput(BaseModel):
    """Structured output from an analyzer agent run."""

    hypothesis_id: str
    status: HypothesisStatus
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
    counter_evidence_checked: list[str] = Field(default_factory=list)
    exploitability: str = ""
    minimal_fix: str = ""
    test_or_poc: str = Field(default="", description="Minimal repro test, not a weaponized exploit")
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Verifier / final report models
# ---------------------------------------------------------------------------

class Finding(BaseModel):
    """A deduplicated, verified security finding."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str
    severity: FindingSeverity
    confidence: float = Field(ge=0.0, le=1.0)
    description: str
    affected_files: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    exploitability: str = ""
    minimal_fix: str = ""
    test_or_poc: str = ""
    hypothesis_ids: list[str] = Field(default_factory=list, description="Source hypotheses")


class ScanReport(BaseModel):
    """Final output of a full SAST scan."""

    scan_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    repo: str = ""
    ref: str = ""
    findings: list[Finding] = Field(default_factory=list)
    hypotheses_total: int = 0
    hypotheses_confirmed: int = 0
    hypotheses_disproven: int = 0
    hypotheses_uncertain: int = 0
    summary: str = ""
