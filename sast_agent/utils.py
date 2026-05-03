"""Shared utilities for JSON extraction from LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any, List


def find_largest_json_object(text: str, discriminator_key: str = "") -> str:
    """Find the largest valid JSON object in a text string.

    Args:
        text: The text to search for JSON objects.
        discriminator_key: If set, prefer JSON objects that contain this key
            (e.g. "hypotheses" for planner output, "status" for analyzer output).

    Returns:
        The JSON string, or "" if none found.
    """
    candidates: list[str] = []

    # Strategy 1: ```json blocks.
    for match in re.finditer(r"```json\s*\n?(.*?)```", text, re.DOTALL):
        candidates.append(match.group(1).strip())

    # Strategy 2: ``` blocks that start with {.
    if not candidates:
        for match in re.finditer(r"```\s*\n?(.*?)```", text, re.DOTALL):
            content = match.group(1).strip()
            if content.startswith("{"):
                candidates.append(content)

    # Strategy 3: raw { } pairs.
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[i:j + 1])
                        break
            i = j + 1 if depth == 0 else i + 1
        else:
            i += 1

    # Prefer candidates that have the discriminator key.
    if discriminator_key:
        best = ""
        for candidate in candidates:
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and discriminator_key in data:
                    if len(candidate) > len(best):
                        best = candidate
            except (json.JSONDecodeError, ValueError):
                continue
        if best:
            return best

    # Fallback: return the largest valid JSON object.
    for candidate in sorted(candidates, key=len, reverse=True):
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            continue

    return ""


def extract_all_text_from_messages(messages: List[dict]) -> str:
    """Pull all text blocks out of a Bedrock-format conversation history."""
    parts: list[str] = []
    for msg in messages:
        for block in msg.get("content", []):
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
    return "\n".join(parts)
