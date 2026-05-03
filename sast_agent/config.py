"""Configuration for the SAST agent system."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = "bedrock"  # "bedrock", "openai", "local"
    model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    max_tokens: int = 8192
    temperature: float = 0.2
    # Bedrock-specific
    region: str = "us-east-1"
    # OpenAI-specific
    api_key: str = ""
    # Local-specific (also used by OpenAI when overriding)
    base_url: str = ""


class EmbeddingConfig(BaseModel):
    """Embedding provider configuration."""

    provider: str = "bedrock"  # "bedrock", "openai", "local"
    model_id: str = "amazon.titan-embed-text-v2:0"
    dimensions: int = 1024
    # Bedrock-specific
    region: str = "us-east-1"
    # OpenAI-specific
    api_key: str = ""
    # Local-specific
    base_url: str = ""


# Keep BedrockConfig as an alias for backward compatibility.
BedrockConfig = LLMConfig


class ToolConfig(BaseModel):
    """Configuration for available tools."""

    codeql_enabled: bool = True
    codeql_db_path: str = ""
    python_repl_enabled: bool = True
    poc_generation_enabled: bool = True
    poc_sandbox_only: bool = True
    semantic_search_enabled: bool = True
    semantic_index_path: str = "/tmp/sast-agent-index"
    dast_enabled: bool = False
    target_url: str = ""
    dast_allowed_hosts: list[str] = Field(default_factory=list)


class ScanConfig(BaseModel):
    """Top-level scan configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    bedrock: Optional[LLMConfig] = Field(default=None, exclude=True)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    max_hypotheses: int = 20
    analyzer_token_budget: int = 4000
    max_concurrent_analyzers: int = 4
    repo_path: str = "."
    git_diff_base: str = ""
    target_files: list[str] = Field(default_factory=list)

    def __init__(self, **data):
        if "bedrock" in data and "llm" not in data:
            data["llm"] = data.pop("bedrock")
        else:
            data.pop("bedrock", None)
        super().__init__(**data)
