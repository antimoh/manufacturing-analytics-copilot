"""
LLM Client for Manufacturing Analytics Copilot.

Provides a unified interface for multiple LLM providers:
- Groq (GPT-OSS 20B) - Free tier, default
- AWS Bedrock (Claude 3 Haiku) - Production deployment
- Ollama - Local development, zero cost

Note: Llama models on Groq are now Enterprise-only (as of 2026).

Author: Anton Timokhin
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str = ""
    provider: str = ""


@dataclass
class TokenUsage:
    """Tracks cumulative token usage across a session."""
    
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    
    def add(self, response: LLMResponse) -> None:
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_calls += 1
    
    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, system_prompt: str, user_message: str, max_tokens: int = 1024, temperature: float = 0.0) -> LLMResponse:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError


class GroqClient(BaseLLMClient):
    """Groq API client for GPT-OSS models. Free tier: 1K RPM, 250K TPM."""
    
    DEFAULT_MODEL = "openai/gpt-oss-20b"
    
    def __init__(self, model: Optional[str] = None):
        self._model = model or self.DEFAULT_MODEL
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
            except ImportError:
                raise ImportError("groq package not installed. Run: pip install groq")
            
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set.")
            
            self._client = Groq(api_key=api_key)
        return self._client
    
    def generate(self, system_prompt: str, user_message: str, max_tokens: int = 1024, temperature: float = 0.0) -> LLMResponse:
        client = self._get_client()
        start_time = time.perf_counter()
        
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return LLMResponse(
            content=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            latency_ms=latency_ms,
            model=self._model,
            provider="groq",
        )
    
    @property
    def provider_name(self) -> str:
        return "Groq"
    
    @property
    def model_name(self) -> str:
        return self._model


class BedrockClient(BaseLLMClient):
    """AWS Bedrock client for Claude 3 Haiku."""
    
    DEFAULT_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
    
    def __init__(self, model: Optional[str] = None, region: Optional[str] = None):
        self._model = model or self.DEFAULT_MODEL
        self._region = region or os.environ.get("AWS_REGION", "us-east-1")
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError("boto3 package not installed. Run: pip install boto3")
            self._client = boto3.client("bedrock-runtime", region_name=self._region)
        return self._client
    
    def generate(self, system_prompt: str, user_message: str, max_tokens: int = 1024, temperature: float = 0.0) -> LLMResponse:
        import json
        client = self._get_client()
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }
        
        start_time = time.perf_counter()
        response = client.invoke_model(
            modelId=self._model,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        response_body = json.loads(response["body"].read())
        
        return LLMResponse(
            content=response_body["content"][0]["text"],
            input_tokens=response_body.get("usage", {}).get("input_tokens", 0),
            output_tokens=response_body.get("usage", {}).get("output_tokens", 0),
            latency_ms=latency_ms,
            model=self._model,
            provider="bedrock",
        )
    
    @property
    def provider_name(self) -> str:
        return "AWS Bedrock"
    
    @property
    def model_name(self) -> str:
        return self._model


class OllamaClient(BaseLLMClient):
    """Ollama client for local model inference."""
    
    DEFAULT_MODEL = "llama3.2"
    DEFAULT_HOST = "http://localhost:11434"
    
    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        self._model = model or os.environ.get("OLLAMA_MODEL", self.DEFAULT_MODEL)
        self._host = host or os.environ.get("OLLAMA_HOST", self.DEFAULT_HOST)
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import ollama
            except ImportError:
                raise ImportError("ollama package not installed. Run: pip install ollama")
            self._client = ollama.Client(host=self._host)
        return self._client
    
    def generate(self, system_prompt: str, user_message: str, max_tokens: int = 1024, temperature: float = 0.0) -> LLMResponse:
        client = self._get_client()
        start_time = time.perf_counter()
        
        response = client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            options={"num_predict": max_tokens, "temperature": temperature},
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return LLMResponse(
            content=response["message"]["content"],
            input_tokens=response.get("prompt_eval_count", 0),
            output_tokens=response.get("eval_count", 0),
            latency_ms=latency_ms,
            model=self._model,
            provider="ollama",
        )
    
    @property
    def provider_name(self) -> str:
        return "Ollama"
    
    @property
    def model_name(self) -> str:
        return self._model


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing."""
    
    def __init__(self, responses: Optional[list[str]] = None):
        self._responses = responses or ["SELECT 1"]
        self._call_count = 0
        self._calls: list[dict] = []
    
    def generate(self, system_prompt: str, user_message: str, max_tokens: int = 1024, temperature: float = 0.0) -> LLMResponse:
        self._calls.append({"system_prompt": system_prompt, "user_message": user_message})
        response_text = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        
        return LLMResponse(
            content=response_text,
            input_tokens=(len(system_prompt) + len(user_message)) // 4,
            output_tokens=len(response_text) // 4,
            latency_ms=10.0,
            model="mock-model",
            provider="mock",
        )
    
    @property
    def provider_name(self) -> str:
        return "Mock"
    
    @property
    def model_name(self) -> str:
        return "mock-model"
    
    @property
    def call_count(self) -> int:
        return self._call_count
    
    @property
    def calls(self) -> list[dict]:
        return self._calls


class LLMClientFactory:
    """Factory for creating LLM clients based on configuration."""
    
    PROVIDERS = {
        "groq": GroqClient,
        "bedrock": BedrockClient,
        "ollama": OllamaClient,
        "mock": MockLLMClient,
    }
    
    @classmethod
    def create(cls, provider: Optional[str] = None, **kwargs) -> BaseLLMClient:
        if provider is None:
            provider = os.environ.get("LLM_PROVIDER", "groq")
        
        provider = provider.lower()
        
        if provider not in cls.PROVIDERS:
            valid = ", ".join(cls.PROVIDERS.keys())
            raise ValueError(f"Unknown LLM provider: {provider}. Valid: {valid}")
        
        return cls.PROVIDERS[provider](**kwargs)
    
    @classmethod
    def get_available_providers(cls) -> list[str]:
        return list(cls.PROVIDERS.keys())
