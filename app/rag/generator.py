import os
import logging
from abc import ABC, abstractmethod
from typing import Generator, Optional
from app.config.settings import settings

logger = logging.getLogger("ecommerce_rag")


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def stream_generate(self, prompt: str) -> Generator[str, None, None]:
        pass


class MockGenerator(BaseGenerator):
    """Offline generator for automated testing and local development."""
    def generate(self, prompt: str) -> str:
        return (
            "Based on our store catalog and official policies [Source 1]:\n\n"
            "Here is the verified information matching your request. Our products are fully tested "
            "and covered under the ShopWise standard warranty."
        )

    def stream_generate(self, prompt: str) -> Generator[str, None, None]:
        tokens = self.generate(prompt).split(" ")
        for t in tokens:
            yield t + " "


class GroqGenerator(BaseGenerator):
    """Production Groq LLM generator with dynamic model fallback."""
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.client = None
        self._init_client()

    def _init_client(self):
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            self.model = self._resolve_best_model()
            logger.info(f"Initialized Groq generator with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            self.client = None

    def _resolve_best_model(self) -> str:
        if not self.client:
            return self.model
        try:
            available_models = [m.id for m in self.client.models.list().data if getattr(m, "active", True)]
            priority_models = [
                self.model,
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
                "qwen/qwen3.8-27b",
                "llama-3.1-8b-instant"
            ]
            for p in priority_models:
                if p in available_models:
                    return p
            # Filter non-audio models
            text_models = [m for m in available_models if "whisper" not in m and "orpheus" not in m]
            if text_models:
                return text_models[0]
        except Exception as e:
            logger.warning(f"Error resolving Groq model list ({e}); keeping configured default: {self.model}")
        return self.model

    def generate(self, prompt: str) -> str:
        if not self.client:
            return MockGenerator().generate(prompt)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional e-commerce shopping and policy assistant. Strictly adhere to context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq generation failed ({e}); falling back to mock response.")
            return f"I apologize, but our AI assistant encountered a temporary connectivity issue ({e}). Here is our policy overview [Source 1]."

    def stream_generate(self, prompt: str) -> Generator[str, None, None]:
        if not self.client:
            yield from MockGenerator().stream_generate(prompt)
            return
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional e-commerce shopping and policy assistant. Strictly adhere to context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
        except Exception as e:
            logger.error(f"Groq streaming failed ({e})")
            yield f"Error in stream: {e}"


def get_generator() -> BaseGenerator:
    """Factory creating configured LLM provider."""
    provider = settings.LLM_PROVIDER.lower()
    if provider == "groq" and settings.GROQ_API_KEY:
        return GroqGenerator(settings.GROQ_API_KEY, settings.LLM_MODEL)
    return MockGenerator()
