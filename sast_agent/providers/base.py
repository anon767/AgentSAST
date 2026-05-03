"""Abstract base classes for LLM and embedding providers."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract interface for LLM conversation with tool use.

    All providers use Bedrock's message format internally:
    - Messages: list of {role, content: [{text: ...}, {toolUse: ...}]}
    - Tools: list of {toolSpec: {name, description, inputSchema}}
    - Response: {stopReason, output: {message: {role, content}}}
    """

    @abstractmethod
    def converse(
        self,
        messages: List[dict],
        system: str = "",
        tools: List[dict] | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Single conversation turn."""
        ...

    def converse_with_tools(
        self,
        messages: List[dict],
        system: str,
        tools: List[dict],
        tool_handler: Callable[[str, dict], str],
        max_turns: int = 15,
    ) -> Tuple[str, List[dict]]:
        """Multi-turn conversation with automatic tool execution.

        Returns (final_text_response, full_message_history).
        """
        for turn in range(max_turns):
            response = self.converse(messages, system=system, tools=tools)
            stop_reason = response.get("stopReason", "")
            output_message = response.get("output", {}).get("message", {})

            messages.append(output_message)

            if stop_reason == "end_turn":
                text_parts = [
                    block["text"]
                    for block in output_message.get("content", [])
                    if "text" in block
                ]
                return "\n".join(text_parts), messages

            if stop_reason == "tool_use":
                tool_results = []
                for block in output_message.get("content", []):
                    if "toolUse" not in block:
                        continue
                    tool_use = block["toolUse"]
                    tool_name = tool_use["name"]
                    tool_input = tool_use["input"]
                    tool_use_id = tool_use["toolUseId"]

                    logger.info("Tool call: %s(%s)", tool_name, json.dumps(tool_input)[:200])

                    try:
                        result = tool_handler(tool_name, tool_input)
                        result_str = str(result)[:30_000]
                        logger.debug("Tool result: %s → %s", tool_name, result_str[:500])
                        tool_results.append({
                            "toolResult": {
                                "toolUseId": tool_use_id,
                                "content": [{"text": result_str}],
                            }
                        })
                    except Exception as exc:
                        logger.error("Tool error: %s — %s", tool_name, exc)
                        tool_results.append({
                            "toolResult": {
                                "toolUseId": tool_use_id,
                                "content": [{"text": f"ERROR: {exc}"}],
                                "status": "error",
                            }
                        })

                messages.append({"role": "user", "content": tool_results})
            else:
                logger.warning("Unexpected stop reason: %s", stop_reason)
                text_parts = [
                    block.get("text", "")
                    for block in output_message.get("content", [])
                    if "text" in block
                ]
                return "\n".join(text_parts), messages

        return "Max turns reached without final response.", messages


class EmbeddingProvider(ABC):
    """Abstract interface for text embeddings."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text. Returns a 1-D float32 array."""
        ...

    def embed_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Embed multiple texts. Returns (N, dim) float32 array."""
        embeddings: List[np.ndarray] = []
        iterator = tqdm(texts, desc="Embedding chunks", unit="chunk",
                        disable=not show_progress)
        for text in iterator:
            embeddings.append(self.embed_text(text))
        return np.vstack(embeddings)
