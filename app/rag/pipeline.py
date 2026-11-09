import time
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from app.rag.retriever import EcommerceRetriever
from app.rag.prompt_builder import EcommercePromptBuilder
from app.rag.generator import get_generator, BaseGenerator
from app.db.models import QueryLog

logger = logging.getLogger("ecommerce_rag")


class EcommerceRAGPipeline:
    """End-to-end RAG orchestrator for E-Commerce shopping and customer support."""
    def __init__(self, db_session: Optional[Session] = None):
        self.db = db_session
        self.retriever = EcommerceRetriever()
        self.prompt_builder = EcommercePromptBuilder()
        self.generator: BaseGenerator = get_generator()

    def classify_intent(self, query: str) -> str:
        """Classify shopper query into product discovery, policy inquiry, or hybrid."""
        q = query.lower()

        policy_keywords = [
            "return", "refund", "exchange", "shipping", "deliver", "warranty",
            "guarantee", "price match", "cancel", "fee", "international", "restocking", "payment"
        ]
        product_keywords = [
            "recommend", "suggest", "find", "looking for", "best", "compare",
            "headphone", "shoe", "laptop", "watch", "air fryer", "vacuum", "coffee",
            "sony", "bose", "apple", "nike", "asics", "dyson", "ninja", "under $"
        ]

        has_policy = any(k in q for k in policy_keywords)
        has_product = any(k in q for k in product_keywords)

        if has_policy and has_product:
            return "hybrid"
        elif has_policy:
            return "policy_support"
        elif has_product:
            return "product_recommendation"
        return "general"

    def run(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        start_time = time.time()
        intent = self.classify_intent(query)
        logger.info(f"Classified query intent: {intent} for '{query}'")

        # In hybrid or general, retrieve across both. If purely product or policy, prioritize corresponding types
        retrieved_chunks = self.retriever.retrieve(query, top_k=top_k)

        # Build grounded prompt
        prompt = self.prompt_builder.build_prompt(query, retrieved_chunks, intent=intent)

        # Generate LLM response
        answer = self.generator.generate(prompt)
        latency_ms = round((time.time() - start_time) * 1000, 2)

        # Extract formatted source citations
        sources = []
        for i, chunk in enumerate(retrieved_chunks):
            meta = chunk.get("metadata", {})
            sources.append({
                "source_id": f"Source {i+1}",
                "type": meta.get("type", "source"),
                "identifier": meta.get("sku") or meta.get("section") or meta.get("title") or f"Item-{i+1}",
                "title": meta.get("title", ""),
                "price": meta.get("price"),
                "similarity_score": chunk.get("similarity_score", 0.0),
                "snippet": chunk.get("text", "")[:280] + "..." if len(chunk.get("text", "")) > 280 else chunk.get("text", "")
            })

        # Save audit record to DB
        model_name = getattr(self.generator, "model", "mock")
        if self.db:
            try:
                log_record = QueryLog(
                    query_text=query,
                    intent=intent,
                    retrieved_chunks=len(retrieved_chunks),
                    response_text=answer,
                    llm_model=str(model_name),
                    latency_ms=latency_ms
                )
                self.db.add(log_record)
                self.db.commit()
            except Exception as e:
                logger.error(f"Failed to record query log in DB: {e}")

        return {
            "query": query,
            "intent": intent,
            "answer": answer,
            "llm_model": model_name,
            "latency_ms": latency_ms,
            "sources": sources
        }

    def stream_run(self, query: str, top_k: int = 4):
        """Streaming response generator for Server-Sent Events."""
        intent = self.classify_intent(query)
        retrieved_chunks = self.retriever.retrieve(query, top_k=top_k)
        prompt = self.prompt_builder.build_prompt(query, retrieved_chunks, intent=intent)

        for token in self.generator.stream_generate(prompt):
            yield token
