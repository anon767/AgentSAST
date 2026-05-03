"""Planner agent: broad, shallow analysis to produce ranked hypotheses.

The planner's job is to understand the attack surface and produce structured
hypotheses for the analyzer agents to investigate. It stays shallow — it reads
file listings, diffs, route definitions, and import graphs, but does NOT dive
deep into any single code path.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sast_agent.providers import LLMProvider
from sast_agent.config import ScanConfig
from sast_agent.models import Hypothesis, HypothesisStatus, PlannerOutput, RiskLevel
from sast_agent.tool_registry import TOOL_DEFINITIONS, ToolHandler
from sast_agent.utils import find_largest_json_object, extract_all_text_from_messages

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """\
You are a security-focused planner agent performing the first phase of a \
static application security test (SAST). Your job is to stay BROAD and SHALLOW.

## Your mission
1. Understand the repository structure, tech stack, and attack surface.
2. Identify entry points: HTTP routes, API handlers, CLI commands, message \
   consumers, cron jobs, RPC methods.
3. Map auth boundaries, input sources, and sensitive sinks (DB queries, \
   file I/O, command execution, template rendering, deserialization).
4. If a git diff base is provided, focus on CHANGED code paths.
5. Produce a ranked list of security hypotheses.

## Tools available
You have shell access (ls, cat, grep, find, git), semantic code search, \
CodeQL query listing, file reading, and the ability to write and run \
arbitrary CodeQL queries. Use them to explore the codebase.

## CRITICAL OUTPUT REQUIREMENT
When you have finished your investigation, your FINAL message MUST contain \
EXACTLY ONE JSON code block with your findings. Use this format:

```json
{{
  "repo_summary": "Description of the repo, tech stack, purpose",
  "attack_surface": ["area1", "area2"],
  "hypotheses": [
    {{
      "hypothesis": "Description of the potential vulnerability",
      "risk_area": "sqli|xss|ssrf|idor|auth_bypass|rce|path_traversal|insecure_deserialization|hardcoded_credentials|open_redirect|csrf|info_disclosure|...",
      "priority": "critical|high|medium|low|info",
      "entrypoints": ["GET /search", "POST /api/users"],
      "files": ["routes/search.ts", "db/queryBuilder.ts"],
      "suggested_queries": ["CodeQL taint query for request param to SQL sink"],
      "token_budget": 4000
    }}
  ]
}}
```

## Rules
- Do NOT dive deep into any single hypothesis. That's the analyzer's job.
- Rank hypotheses by risk: critical > high > medium > low.
- Limit to at most {max_hypotheses} hypotheses.
- Focus on REAL, exploitable patterns — not theoretical style issues.
- If there's a git diff, prioritize changed code paths.
- Use semantic_search to find security-relevant patterns quickly.
- Your LAST message MUST contain the JSON block. Do NOT end with just text.
"""


def run_planner(config: ScanConfig, bedrock: LLMProvider) -> PlannerOutput:
    """Execute the planner agent and return structured hypotheses."""
    tool_handler = ToolHandler(config)
    system = PLANNER_SYSTEM_PROMPT.format(max_hypotheses=config.max_hypotheses)

    # Add DAST context if a target URL is configured.
    if config.tools.dast_enabled and config.tools.target_url:
        system += (
            "\n\n## DAST Mode — Live Application Available\n"
            f"The application is running at: {config.tools.target_url}\n"
            "You can use python_exec with `import requests` to send HTTP requests "
            "to this target (and ONLY this target). Use this to:\n"
            "- Discover live endpoints and their response behavior\n"
            "- Check which routes require authentication\n"
            "- Probe for error messages that reveal stack traces or internals\n"
            "- Verify that hypothesized entry points are actually reachable\n"
            "Keep DAST probing lightweight during planning — save deep exploitation for analyzers.\n"
        )

    # Build the initial user message.
    user_msg = "Analyze this repository for security vulnerabilities.\n\n"
    user_msg += f"Repository path: {config.repo_path}\n"
    if config.git_diff_base:
        user_msg += f"Git diff base: {config.git_diff_base} (focus on changed files)\n"
    if config.target_files:
        user_msg += f"Target files: {', '.join(config.target_files)}\n"
    if config.tools.dast_enabled:
        user_msg += f"Live target: {config.tools.target_url} (use python_exec with requests to probe)\n"
    user_msg += (
        "\nStart by listing the file structure, then identify entry points, "
        "auth boundaries, and sensitive sinks. Use semantic search to find "
        "security-relevant patterns. When done investigating, output your "
        "ranked hypotheses as a JSON code block."
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"text": user_msg}]}
    ]

    logger.info("Starting planner agent...")
    response_text, messages = bedrock.converse_with_tools(
        messages=messages,
        system=system,
        tools=TOOL_DEFINITIONS,
        tool_handler=tool_handler,
        max_turns=25,
    )

    # Try to parse JSON from the response.
    output = _parse_planner_output(response_text, config)

    # If the planner didn't produce hypotheses, try extracting JSON from the
    # full conversation history first — the model may have emitted it in a
    # turn that preceded more tool calls.
    if not output.hypotheses:
        logger.info("No JSON in final response, scanning conversation history...")
        all_text = extract_all_text_from_messages(messages)
        output = _parse_planner_output(all_text, config)

    # Still nothing — send a follow-up explicitly asking for the JSON.
    if not output.hypotheses:
        logger.info("No JSON found anywhere, sending follow-up request...")
        messages.append({
            "role": "user",
            "content": [{
                "text": (
                    "You've completed your investigation. Now output your findings. "
                    "Respond with ONLY a JSON code block containing repo_summary, "
                    "attack_surface, and hypotheses arrays. No other text. "
                    "Do NOT call any more tools."
                ),
            }],
        })

        # Must pass tools= because the message history contains toolUse blocks.
        followup_text, messages = bedrock.converse_with_tools(
            messages=messages,
            system=system,
            tools=TOOL_DEFINITIONS,
            tool_handler=tool_handler,
            max_turns=3,
        )
        output = _parse_planner_output(followup_text, config)

        if not output.hypotheses:
            # Last resort: scan everything again.
            all_text = extract_all_text_from_messages(messages)
            output = _parse_planner_output(all_text, config)

    return output


def _parse_planner_output(text: str, config: ScanConfig) -> PlannerOutput:
    """Extract the JSON output from the planner's response text."""
    json_str = find_largest_json_object(text, discriminator_key="hypotheses")

    if not json_str:
        logger.warning("Could not find JSON in planner output")
        return PlannerOutput(
            repo_summary="Planner did not produce structured output.",
            hypotheses=[],
        )

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse planner JSON: %s", exc)
        return PlannerOutput(
            repo_summary=f"JSON parse error: {exc}",
            hypotheses=[],
        )

    # Build hypotheses from the parsed data.
    hypotheses: list[Hypothesis] = []
    for h in data.get("hypotheses", []):
        try:
            priority = RiskLevel(h.get("priority", "medium"))
        except ValueError:
            priority = RiskLevel.MEDIUM

        hypotheses.append(Hypothesis(
            hypothesis=h.get("hypothesis", ""),
            risk_area=h.get("risk_area", "other"),
            priority=priority,
            entrypoints=h.get("entrypoints", []),
            files=h.get("files", []),
            suggested_queries=h.get("suggested_queries", []),
            token_budget=h.get("token_budget", config.analyzer_token_budget),
        ))

    return PlannerOutput(
        repo_summary=data.get("repo_summary", ""),
        attack_surface=data.get("attack_surface", []),
        hypotheses=hypotheses,
        metadata=data.get("metadata", {}),
    )
