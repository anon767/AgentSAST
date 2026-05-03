"""OpenAI provider: GPT models + text-embedding-3.

Translates between Bedrock's internal message format and OpenAI's
chat completion format transparently.
"""

from __future__ import annotations

import json

import numpy as np

from sast_agent.providers.base import EmbeddingProvider, LLMProvider


class OpenAILLM(LLMProvider):
    """OpenAI GPT models with tool use."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str = "",
        base_url: str = "",
        max_tokens: int = 8192,
        temperature: float = 0.2,
    ) -> None:
        from openai import OpenAI
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def converse(self, messages, system="", tools=None, max_tokens=None):
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})

        for msg in messages:
            oai_msg = self._bedrock_msg_to_openai(msg)
            if oai_msg:
                if isinstance(oai_msg, list):
                    oai_messages.extend(oai_msg)
                else:
                    oai_messages.append(oai_msg)

        oai_tools = None
        if tools:
            oai_tools = []
            for t in tools:
                spec = t.get("toolSpec", {})
                oai_tools.append({
                    "type": "function",
                    "function": {
                        "name": spec["name"],
                        "description": spec.get("description", ""),
                        "parameters": spec.get("inputSchema", {}).get("json", {}),
                    },
                })

        kwargs: dict = {
            "model": self.model,
            "messages": oai_messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature,
        }
        if oai_tools:
            kwargs["tools"] = oai_tools

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        return self._openai_response_to_bedrock(choice)

    def _bedrock_msg_to_openai(self, msg: dict):
        role = msg.get("role", "user")
        content = msg.get("content", [])

        tool_results = [b for b in content if isinstance(b, dict) and "toolResult" in b]
        if tool_results:
            result_msgs = []
            for tr in tool_results:
                result = tr["toolResult"]
                text = ""
                for c in result.get("content", []):
                    if "text" in c:
                        text += c["text"]
                result_msgs.append({
                    "role": "tool",
                    "tool_call_id": result["toolUseId"],
                    "content": text,
                })
            return result_msgs

        text_parts = []
        tool_calls = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    text_parts.append(block["text"])
                elif "toolUse" in block:
                    tu = block["toolUse"]
                    tool_calls.append({
                        "id": tu["toolUseId"],
                        "type": "function",
                        "function": {
                            "name": tu["name"],
                            "arguments": json.dumps(tu["input"]),
                        },
                    })

        oai_msg: dict = {"role": role, "content": "\n".join(text_parts) if text_parts else None}
        if tool_calls:
            oai_msg["role"] = "assistant"
            oai_msg["tool_calls"] = tool_calls
        return oai_msg

    def _openai_response_to_bedrock(self, choice) -> dict:
        content_blocks = []

        if choice.message.content:
            content_blocks.append({"text": choice.message.content})

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                content_blocks.append({
                    "toolUse": {
                        "toolUseId": tc.id,
                        "name": tc.function.name,
                        "input": args,
                    }
                })

        stop_reason = "end_turn"
        if choice.finish_reason == "tool_calls":
            stop_reason = "tool_use"

        return {
            "stopReason": stop_reason,
            "output": {
                "message": {
                    "role": "assistant",
                    "content": content_blocks,
                }
            },
        }


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI text-embedding-3 models."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        base_url: str = "",
        dimensions: int = 1024,
    ) -> None:
        from openai import OpenAI
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self._dim = dimensions

    @property
    def dim(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model,
            input=text[:8000],
            dimensions=self._dim,
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Native batch embedding — OpenAI accepts a list of inputs per call."""
        from tqdm import tqdm
        all_embeddings: List[np.ndarray] = []
        batch_size = 100  # OpenAI limit is 2048 inputs, but 100 keeps payloads sane
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        for batch in tqdm(batches, desc="Embedding chunks", unit="batch",
                          disable=not show_progress):
            truncated = [t[:8000] for t in batch]
            response = self.client.embeddings.create(
                model=self.model,
                input=truncated,
                dimensions=self._dim,
            )
            for item in sorted(response.data, key=lambda x: x.index):
                all_embeddings.append(np.array(item.embedding, dtype=np.float32))
        return np.vstack(all_embeddings)
