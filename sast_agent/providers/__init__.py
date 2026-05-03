"""LLM and embedding provider abstractions.

Usage:
    from sast_agent.providers import LLMProvider, EmbeddingProvider
    from sast_agent.providers import create_llm_from_config, create_embedding_from_config
"""

from __future__ import annotations

from sast_agent.providers.base import EmbeddingProvider, LLMProvider

__all__ = [
    "LLMProvider",
    "EmbeddingProvider",
    "create_llm",
    "create_llm_from_config",
    "create_embedding",
    "create_embedding_from_config",
]


def create_llm(provider: str = "bedrock", **kwargs) -> LLMProvider:
    """Create an LLM provider by name."""
    if provider == "bedrock":
        from sast_agent.providers.bedrock import BedrockLLM
        return BedrockLLM(**kwargs)
    elif provider == "openai":
        from sast_agent.providers.openai_provider import OpenAILLM
        return OpenAILLM(**kwargs)
    elif provider == "local":
        from sast_agent.providers.local import LocalLLM
        return LocalLLM(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def create_embedding(provider: str = "bedrock", **kwargs) -> EmbeddingProvider:
    """Create an embedding provider by name."""
    if provider == "bedrock":
        from sast_agent.providers.bedrock import BedrockEmbedding
        return BedrockEmbedding(**kwargs)
    elif provider == "openai":
        from sast_agent.providers.openai_provider import OpenAIEmbedding
        return OpenAIEmbedding(**kwargs)
    elif provider == "local":
        from sast_agent.providers.local import LocalEmbedding
        return LocalEmbedding(**kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def create_llm_from_config(cfg) -> LLMProvider:
    """Create an LLM provider from an LLMConfig object."""
    p = cfg.provider
    if p == "bedrock":
        from sast_agent.providers.bedrock import BedrockLLM
        return BedrockLLM(
            model_id=cfg.model_id, region=cfg.region,
            max_tokens=cfg.max_tokens, temperature=cfg.temperature,
        )
    elif p == "openai":
        from sast_agent.providers.openai_provider import OpenAILLM
        return OpenAILLM(
            model=cfg.model_id, api_key=cfg.api_key, base_url=cfg.base_url,
            max_tokens=cfg.max_tokens, temperature=cfg.temperature,
        )
    elif p == "local":
        from sast_agent.providers.local import LocalLLM
        return LocalLLM(
            model=cfg.model_id, base_url=cfg.base_url,
            max_tokens=cfg.max_tokens, temperature=cfg.temperature,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {p}")


def create_embedding_from_config(cfg) -> EmbeddingProvider:
    """Create an embedding provider from an EmbeddingConfig object."""
    p = cfg.provider
    if p == "bedrock":
        from sast_agent.providers.bedrock import BedrockEmbedding
        return BedrockEmbedding(
            model_id=cfg.model_id, region=cfg.region, dimensions=cfg.dimensions,
        )
    elif p == "openai":
        from sast_agent.providers.openai_provider import OpenAIEmbedding
        return OpenAIEmbedding(
            model=cfg.model_id, api_key=cfg.api_key, base_url=cfg.base_url,
            dimensions=cfg.dimensions,
        )
    elif p == "local":
        from sast_agent.providers.local import LocalEmbedding
        return LocalEmbedding(
            model=cfg.model_id, base_url=cfg.base_url, dimensions=cfg.dimensions,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {p}")
