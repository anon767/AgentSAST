"""Analyzer agent: narrow, deep, evidence-driven investigation of a single hypothesis.

Each analyzer instance receives ONE hypothesis and a tight budget. Its job is to
prove it, disprove it, or mark it uncertain — with evidence.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sast_agent.providers import LLMProvider
from sast_agent.config import ScanConfig
from sast_agent.utils import find_largest_json_object, extract_all_text_from_messages
from sast_agent.models import (
    AnalyzerOutput,
    Evidence,
    Hypothesis,
    HypothesisStatus,
)
from sast_agent.tool_registry import TOOL_DEFINITIONS, ToolHandler

logger = logging.getLogger(__name__)

ANALYZER_SYSTEM_PROMPT = """\
You are a security analyzer agent. You have been assigned ONE specific \
hypothesis to investigate. Your job is to PROVE it, DISPROVE it, or mark \
it UNCERTAIN — with concrete evidence.

## Your hypothesis
{hypothesis_json}

## Rules
1. Stay focused on THIS hypothesis only. Do not wander.
2. Use tools to gather evidence: read files, grep for patterns, run CodeQL \
   queries (built-in or custom via codeql_query_raw), use semantic search.
3. Check for counter-evidence: sanitization, validation, parameterized queries, \
   auth checks, etc.
4. If the vulnerability is real, determine exploitability and suggest a minimal fix.
5. If appropriate, write a minimal PoC using python_exec that demonstrates \
   reachability (NOT a weaponized exploit — just a unit-test-style repro).
6. PoC rules: target local test fixtures only, no network calls, no destructive \
   actions, no real system attacks.

## CRITICAL OUTPUT REQUIREMENT
Your FINAL message MUST contain EXACTLY ONE JSON code block with your findings. \
Do NOT end with just prose. Use this format:

```json
{{
  "hypothesis_id": "{hypothesis_id}",
  "status": "confirmed|disproven|uncertain",
  "confidence": 0.85,
  "evidence": [
    {{
      "description": "What was found",
      "file": "path/to/file.ts",
      "line_range": "42-58",
      "snippet": "relevant code snippet",
      "tool_used": "grep_code|read_file|codeql_run|semantic_search|etc"
    }}
  ],
  "counter_evidence_checked": [
    "Checked for parameterized queries in db/query.ts — none found",
    "Checked for input validation middleware — not applied to this route"
  ],
  "exploitability": "Description of how this can be exploited",
  "minimal_fix": "Use parameterized queries instead of string concatenation",
  "test_or_poc": "Minimal repro test code or description",
  "reasoning": "Step-by-step reasoning for the conclusion"
}}
```

## Strategy
1. First, read the files mentioned in the hypothesis.
2. Use semantic_search to find related code (sinks, sources, sanitizers).
3. Trace the data flow from entry point to sink.
4. Check for existing protections (validation, sanitization, auth).
5. If confirmed, assess exploitability and write a minimal PoC.
6. Output your structured conclusion as a JSON code block.
"""


def run_analyzer(
    hypothesis: Hypothesis,
    config: ScanConfig,
    bedrock: LLMProvider,
    tool_handler: ToolHandler | None = None,
) -> AnalyzerOutput:
    """Execute an analyzer agent for a single hypothesis."""
    if tool_handler is None:
        tool_handler = ToolHandler(config)

    hypothesis_json = hypothesis.model_dump_json(indent=2)
    system = ANALYZER_SYSTEM_PROMPT.format(
        hypothesis_json=hypothesis_json,
        hypothesis_id=hypothesis.id,
    )

    # Add DAST context if available.
    if config.tools.dast_enabled and config.tools.target_url:
        system += (
            "\n\n## DAST Mode — Live Application Available\n"
            f"The application is running at: {config.tools.target_url}\n"
            "You can use python_exec with `import requests` to send HTTP requests "
            "to this target. Use this to CONFIRM vulnerabilities by:\n"
            "- Sending crafted payloads to the vulnerable endpoint\n"
            "- Checking response codes, headers, and body for evidence of exploitation\n"
            "- Verifying that injected values appear in responses (XSS, SQLi error messages)\n"
            "- Testing auth bypass by accessing protected resources without credentials\n"
            "- Confirming SSRF by requesting internal URLs through the vulnerable endpoint\n\n"
            "RULES for DAST probing:\n"
            "- Only target the configured URL — requests to other hosts are blocked\n"
            "- Do NOT perform destructive actions (DROP TABLE, rm -rf, etc.)\n"
            "- Do NOT attempt denial of service\n"
            "- Use read-only payloads that demonstrate the vulnerability without causing damage\n"
            "- A successful DAST confirmation significantly increases confidence\n"
        )

    user_msg = (
        f"Investigate this hypothesis:\n\n"
        f"**{hypothesis.hypothesis}**\n\n"
        f"Risk area: {hypothesis.risk_area}\n"
        f"Priority: {hypothesis.priority.value}\n"
        f"Entry points: {', '.join(hypothesis.entrypoints) or 'unknown'}\n"
        f"Files to inspect: {', '.join(hypothesis.files) or 'unknown'}\n"
        f"Suggested queries: {', '.join(hypothesis.suggested_queries) or 'none'}\n\n"
        f"Start by reading the relevant files, then trace the data flow. "
        f"Prove or disprove this hypothesis with evidence. "
        f"End with a JSON code block containing your structured conclusion."
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"text": user_msg}]}
    ]

    logger.info("Starting analyzer for hypothesis %s: %s",
                hypothesis.id, hypothesis.hypothesis[:80])

    response_text, messages = bedrock.converse_with_tools(
        messages=messages,
        system=system,
        tools=TOOL_DEFINITIONS,
        tool_handler=tool_handler,
        max_turns=15,
    )

    # Try parsing the final response first.
    output = _parse_analyzer_output(response_text, hypothesis.id)

    if output.status == HypothesisStatus.UNCERTAIN and output.confidence <= 0.3:
        # JSON parsing likely failed — scan the full conversation history.
        logger.info("Scanning conversation history for analyzer %s JSON...", hypothesis.id)
        all_text = extract_all_text_from_messages(messages)
        output2 = _parse_analyzer_output(all_text, hypothesis.id)
        if output2.confidence > output.confidence:
            output = output2

    # Still no good output — ask for the JSON explicitly.
    if output.status == HypothesisStatus.UNCERTAIN and output.confidence <= 0.3:
        logger.info("Sending follow-up to analyzer %s for JSON output...", hypothesis.id)
        messages.append({
            "role": "user",
            "content": [{
                "text": (
                    "You've completed your investigation. Now output your conclusion. "
                    "Respond with ONLY a JSON code block containing hypothesis_id, status, "
                    "confidence, evidence, counter_evidence_checked, exploitability, "
                    "minimal_fix, test_or_poc, and reasoning. Do NOT call any more tools."
                ),
            }],
        })
        followup_text, messages = bedrock.converse_with_tools(
            messages=messages,
            system=system,
            tools=TOOL_DEFINITIONS,
            tool_handler=tool_handler,
            max_turns=2,
        )
        output3 = _parse_analyzer_output(followup_text, hypothesis.id)
        if output3.confidence > output.confidence:
            output = output3

        # Last resort — scan everything.
        if output.confidence <= 0.3:
            all_text = extract_all_text_from_messages(messages)
            output4 = _parse_analyzer_output(all_text, hypothesis.id)
            if output4.confidence > output.confidence:
                output = output4

    return output


def _parse_analyzer_output(text: str, hypothesis_id: str) -> AnalyzerOutput:
    """Extract structured output from the analyzer's response."""
    json_str = find_largest_json_object(text, discriminator_key="status")

    if not json_str:
        logger.warning("Could not find JSON in analyzer output for %s", hypothesis_id)
        return AnalyzerOutput(
            hypothesis_id=hypothesis_id,
            status=HypothesisStatus.UNCERTAIN,
            confidence=0.3,
            reasoning="Analyzer did not produce structured output.",
        )

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse analyzer JSON for %s: %s", hypothesis_id, exc)
        return AnalyzerOutput(
            hypothesis_id=hypothesis_id,
            status=HypothesisStatus.UNCERTAIN,
            confidence=0.2,
            reasoning=f"JSON parse error: {exc}",
        )

    status_map = {
        "confirmed": HypothesisStatus.CONFIRMED,
        "disproven": HypothesisStatus.DISPROVEN,
        "uncertain": HypothesisStatus.UNCERTAIN,
    }
    status = status_map.get(data.get("status", "uncertain"), HypothesisStatus.UNCERTAIN)

    evidence = []
    for e in data.get("evidence", []):
        evidence.append(Evidence(
            description=e.get("description", ""),
            file=e.get("file", ""),
            line_range=e.get("line_range", ""),
            snippet=e.get("snippet", ""),
            tool_used=e.get("tool_used", ""),
        ))

    return AnalyzerOutput(
        hypothesis_id=hypothesis_id,
        status=status,
        confidence=float(data.get("confidence", 0.5)),
        evidence=evidence,
        counter_evidence_checked=data.get("counter_evidence_checked", []),
        exploitability=data.get("exploitability", ""),
        minimal_fix=data.get("minimal_fix", ""),
        test_or_poc=data.get("test_or_poc", ""),
        reasoning=data.get("reasoning", ""),
    )
