"""Verifier agent: deduplicates, challenges, and ranks findings.

The verifier takes all analyzer outputs, deduplicates overlapping findings,
challenges weak evidence, and produces a final ranked report.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sast_agent.providers import LLMProvider
from sast_agent.config import ScanConfig
from sast_agent.utils import find_largest_json_object
from sast_agent.models import (
    AnalyzerOutput,
    Evidence,
    Finding,
    FindingSeverity,
    Hypothesis,
    HypothesisStatus,
    PlannerOutput,
    ScanReport,
)

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM_PROMPT = """\
You are a security findings verifier. You receive the results from multiple \
analyzer agents, each of which investigated a specific security hypothesis.

## Your job
1. DEDUPLICATE: Merge findings that describe the same underlying vulnerability.
2. CHALLENGE: Question weak evidence. If an analyzer marked something as \
   "confirmed" with low confidence or thin evidence, downgrade it.
3. RANK: Order findings by severity and confidence.
4. SUMMARIZE: Produce a clear executive summary.

## Input
You will receive a JSON array of analyzer results and the original hypotheses.

## Output format
Output a single JSON object:
```json
{{
  "findings": [
    {{
      "title": "SQL Injection in /api/search endpoint",
      "severity": "critical|high|medium|low",
      "confidence": 0.0 to 1.0,
      "description": "Clear description of the vulnerability",
      "affected_files": ["routes/search.ts", "db/query.ts"],
      "evidence": [
        {{
          "description": "What was found",
          "file": "path/to/file",
          "line_range": "42-58",
          "snippet": "code snippet",
          "tool_used": "tool name"
        }}
      ],
      "exploitability": "How this could be exploited",
      "minimal_fix": "Recommended fix",
      "test_or_poc": "PoC or test description",
      "hypothesis_ids": ["abc123"]
    }}
  ],
  "summary": "Executive summary of the scan results"
}}
```

## Rules
- Drop "disproven" hypotheses entirely.
- Merge duplicate findings.
- Be skeptical: if evidence is weak, lower the confidence.
- Severity mapping: critical (RCE, auth bypass, SQLi with data exfil), \
  high (XSS, SSRF, path traversal), medium (IDOR, info disclosure), \
  low (missing headers, minor config issues).
"""


def run_verifier(
    planner_output: PlannerOutput,
    analyzer_outputs: list[AnalyzerOutput],
    config: ScanConfig,
    bedrock: LLMProvider,
) -> ScanReport:
    """Run the verifier to produce the final scan report."""

    # Build the input for the verifier.
    hypotheses_map = {h.id: h for h in planner_output.hypotheses}

    results_for_verifier = []
    for ao in analyzer_outputs:
        h = hypotheses_map.get(ao.hypothesis_id)
        results_for_verifier.append({
            "hypothesis_id": ao.hypothesis_id,
            "hypothesis": h.hypothesis if h else "unknown",
            "risk_area": h.risk_area if h else "unknown",
            "status": ao.status.value,
            "confidence": ao.confidence,
            "evidence": [e.model_dump() for e in ao.evidence],
            "counter_evidence_checked": ao.counter_evidence_checked,
            "exploitability": ao.exploitability,
            "minimal_fix": ao.minimal_fix,
            "test_or_poc": ao.test_or_poc,
            "reasoning": ao.reasoning,
        })

    user_msg = (
        f"Repository: {config.repo_path}\n"
        f"Repo summary: {planner_output.repo_summary}\n\n"
        f"## Analyzer Results\n\n"
        f"```json\n{json.dumps(results_for_verifier, indent=2)}\n```\n\n"
        f"Deduplicate, challenge weak evidence, rank by severity, "
        f"and produce the final findings report."
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"text": user_msg}]}
    ]

    logger.info("Starting verifier with %d analyzer results...", len(analyzer_outputs))

    # Verifier doesn't need tools — it's a pure reasoning step.
    response = bedrock.converse(
        messages=messages,
        system=VERIFIER_SYSTEM_PROMPT,
        max_tokens=config.llm.max_tokens,
    )

    output_message = response.get("output", {}).get("message", {})
    text_parts = [
        block["text"]
        for block in output_message.get("content", [])
        if "text" in block
    ]
    response_text = "\n".join(text_parts)

    return _parse_verifier_output(
        response_text, planner_output, analyzer_outputs, config
    )


def _parse_verifier_output(
    text: str,
    planner_output: PlannerOutput,
    analyzer_outputs: list[AnalyzerOutput],
    config: ScanConfig,
) -> ScanReport:
    """Parse verifier output into a ScanReport."""
    json_str = find_largest_json_object(text, discriminator_key="findings")

    # Count hypothesis statuses.
    confirmed = sum(1 for ao in analyzer_outputs if ao.status == HypothesisStatus.CONFIRMED)
    disproven = sum(1 for ao in analyzer_outputs if ao.status == HypothesisStatus.DISPROVEN)
    uncertain = sum(1 for ao in analyzer_outputs if ao.status == HypothesisStatus.UNCERTAIN)

    if not json_str:
        logger.warning("Could not find JSON in verifier output")
        return ScanReport(
            repo=config.repo_path,
            summary="Verifier did not produce structured output.",
            hypotheses_total=len(planner_output.hypotheses),
            hypotheses_confirmed=confirmed,
            hypotheses_disproven=disproven,
            hypotheses_uncertain=uncertain,
        )

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse verifier JSON: %s", exc)
        return ScanReport(
            repo=config.repo_path,
            summary=f"JSON parse error: {exc}",
            hypotheses_total=len(planner_output.hypotheses),
            hypotheses_confirmed=confirmed,
            hypotheses_disproven=disproven,
            hypotheses_uncertain=uncertain,
        )

    findings: list[Finding] = []
    for f in data.get("findings", []):
        try:
            severity = FindingSeverity(f.get("severity", "medium"))
        except ValueError:
            severity = FindingSeverity.MEDIUM

        evidence = []
        for e in f.get("evidence", []):
            evidence.append(Evidence(
                description=e.get("description", ""),
                file=e.get("file", ""),
                line_range=e.get("line_range", ""),
                snippet=e.get("snippet", ""),
                tool_used=e.get("tool_used", ""),
            ))

        findings.append(Finding(
            title=f.get("title", "Untitled finding"),
            severity=severity,
            confidence=float(f.get("confidence", 0.5)),
            description=f.get("description", ""),
            affected_files=f.get("affected_files", []),
            evidence=evidence,
            exploitability=f.get("exploitability", ""),
            minimal_fix=f.get("minimal_fix", ""),
            test_or_poc=f.get("test_or_poc", ""),
            hypothesis_ids=f.get("hypothesis_ids", []),
        ))

    return ScanReport(
        repo=config.repo_path,
        findings=findings,
        hypotheses_total=len(planner_output.hypotheses),
        hypotheses_confirmed=confirmed,
        hypotheses_disproven=disproven,
        hypotheses_uncertain=uncertain,
        summary=data.get("summary", ""),
    )
