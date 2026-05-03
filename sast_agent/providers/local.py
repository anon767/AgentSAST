"""Local model provider via OpenAI-compatible API.

Works with Ollama, llama.cpp server, vLLM, LM Studio, or any server
that exposes an OpenAI-compatible /v1/chat/completions endpoint.
"""

from __future__ import annotations

from sast_agent.providers.openai_provider import OpenAIEmbedding, OpenAILLM


class LocalLLM(OpenAILLM):
    """Local model via OpenAI-compatible API.

    Examples:
        LocalLLM(model="qwen2.5-coder:32b", base_url="http://localhost:11434/v1")  # Ollama
        LocalLLM(model="local", base_url="http://localhost:8080/v1")                # llama.cpp
        LocalLLM(model="Qwen/Qwen2.5-Coder-32B-Instruct", base_url="http://localhost:8000/v1")  # vLLM
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:32b",
        base_url: str = "http://localhost:11434/v1",
        max_tokens: int = 8192,
        temperature: float = 0.2,
    ) -> None:
        super().__init__(
            model=model,
            api_key="not-needed",
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
        )


class LocalEmbedding(OpenAIEmbedding):
    """Local embedding model via OpenAI-compatible API.

    Examples:
        LocalEmbedding(model="nomic-embed-text", base_url="http://localhost:11434/v1")  # Ollama
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434/v1",
        dimensions: int = 768,
    ) -> None:
        super().__init__(
            model=model,
            api_key="not-needed",
            base_url=base_url,
            dimensions=dimensions,
        )
