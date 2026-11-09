import pytest
from app.ingestion.chunker import TextChunker
from app.embeddings.embedder import MockEmbedder
from app.vectorstore.vector_db import LocalVectorCollection
from app.rag.prompt_builder import EcommercePromptBuilder
from app.rag.pipeline import EcommerceRAGPipeline
import tempfile
import os


def test_text_chunker():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    text = "Section 1: Returns. Customers may return unopened products within 30 days of delivery. Opened products have 14 days."
    chunks = chunker.chunk_text(text, metadata={"section": "Returns"})

    assert len(chunks) >= 1
    assert "text" in chunks[0]
    assert chunks[0]["section"] == "Returns"


def test_mock_embedder():
    embedder = MockEmbedder(dimension=64)
    v1 = embedder.embed_text("Sony noise canceling headphones")
    v2 = embedder.embed_text("Sony noise canceling headphones")
    v3 = embedder.embed_text("Nike running shoes")

    assert len(v1) == 64
    assert v1 == v2  # Deterministic
    assert v1 != v3


def test_local_vector_collection():
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "test_store.json")
        col = LocalVectorCollection(store_path)

        embedder = MockEmbedder(dimension=32)
        v1 = embedder.embed_text("Sony headphones")
        v2 = embedder.embed_text("Nike shoes")

        col.add(
            ids=["prod_1", "prod_2"],
            embeddings=[v1, v2],
            documents=["Sony WH-1000XM5", "Nike Pegasus 40"],
            metadatas=[{"category": "Audio"}, {"category": "Footwear"}]
        )

        assert col.count() == 2

        # Query
        res = col.query([v1], n_results=1)
        assert res["ids"][0][0] == "prod_1"
        assert res["documents"][0][0] == "Sony WH-1000XM5"


def test_ecommerce_prompt_builder():
    builder = EcommercePromptBuilder()
    chunks = [
        {
            "text": "Sony XM5 price is $349.99 with 30hr battery.",
            "metadata": {"type": "product", "sku": "AUD-SONY-001"}
        }
    ]
    prompt = builder.build_prompt("Recommend headphones", chunks, intent="product_recommendation")

    assert "ShopWise AI" in prompt
    assert "[Source 1: PRODUCT - AUD-SONY-001]" in prompt
    assert "AUD-SONY-001" in prompt


def test_intent_classification():
    pipeline = EcommerceRAGPipeline()

    assert pipeline.classify_intent("Recommend lightweight running shoes under $120") == "product_recommendation"
    assert pipeline.classify_intent("What is your refund policy on returned clothes?") == "policy_support"
    assert pipeline.classify_intent("Can I return the Sony headphones if I open them?") == "hybrid"
