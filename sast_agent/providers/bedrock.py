"""AWS Bedrock provider: Claude LLM + Titan Embeddings."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np
from tqdm import tqdm

from sast_agent.providers.base import EmbeddingProvider, LLMProvider


class BedrockLLM(LLMProvider):
    """Claude on AWS Bedrock via the Converse API."""

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0",
        region: str = "us-east-1",
        max_tokens: int = 8192,
        temperature: float = 0.2,
    ) -> None:
        import boto3
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def converse(self, messages, system="", tools=None, max_tokens=None):
        kwargs: dict = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens or self.max_tokens,
                "temperature": self.temperature,
            },
        }
        if system:
            kwargs["system"] = [{"text": system}]
        if tools:
            kwargs["toolConfig"] = {"tools": tools}
        return self.client.converse(**kwargs)


class BedrockEmbedding(EmbeddingProvider):
    """Amazon Titan Embeddings V2 on Bedrock."""

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = "us-east-1",
        dimensions: int = 1024,
    ) -> None:
        import boto3
        self.model_id = model_id
        self._dim = dimensions
        self.client = boto3.client("bedrock-runtime", region_name=region)

    @property
    def dim(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> np.ndarray:
        body = json.dumps({
            "inputText": text[:8000],
            "dimensions": self._dim,
            "normalize": True,
        })
        response = self.client.invoke_model(
            modelId=self.model_id, body=body,
            contentType="application/json", accept="application/json",
        )
        result = json.loads(response["body"].read())
        return np.array(result["embedding"], dtype=np.float32)

    def embed_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Parallel embedding — Titan doesn't support native batching,
        so we use threads to overlap the API latency."""
        results: List[tuple] = [None] * len(texts)  # type: ignore[list-item]

        def _embed_one(idx: int, text: str):
            vec = self.embed_text(text)
            return idx, vec

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {
                pool.submit(_embed_one, i, t): i
                for i, t in enumerate(texts)
            }
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Embedding chunks", unit="chunk",
                disable=not show_progress,
            ):
                idx, vec = future.result()
                results[idx] = (idx, vec)

        return np.vstack([vec for _, vec in results])
