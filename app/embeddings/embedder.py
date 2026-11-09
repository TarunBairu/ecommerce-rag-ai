from abc import ABC, abstractmethod
from typing import List
import hashlib
import math
import logging
from app.config.settings import settings

logger = logging.getLogger("ecommerce_rag")


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of texts."""
        pass


class MockEmbedder(BaseEmbedder):
    """
    Deterministic mock embedder for rapid local development and unit tests.
    Generates consistent normalized vectors from text hashes.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _hash_to_vector(self, text: str) -> List[float]:
        vec = []
        for i in range(self.dimension):
            h = hashlib.sha256(f"{text}_{i}".encode("utf-8")).hexdigest()
            val = (int(h[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0
            vec.append(val)
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_text(self, text: str) -> List[float]:
        return self._hash_to_vector(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_to_vector(t) for t in texts]


class SentenceTransformersEmbedder(BaseEmbedder):
    """Local transformer embedding using sentence-transformers."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            logger.warning("sentence-transformers not installed; falling back to MockEmbedder.")
            self.model = None

    def embed_text(self, text: str) -> List[float]:
        if not self.model:
            return MockEmbedder(settings.EMBEDDING_DIMENSION).embed_text(text)
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self.model:
            return MockEmbedder(settings.EMBEDDING_DIMENSION).embed_batch(texts)
        return self.model.encode(texts).tolist()


def get_embedder() -> BaseEmbedder:
    """Factory function for embedding provider."""
    provider = settings.EMBEDDING_PROVIDER.lower()
    if provider == "sentence-transformers":
        return SentenceTransformersEmbedder(settings.EMBEDDING_MODEL)
    return MockEmbedder(settings.EMBEDDING_DIMENSION)
