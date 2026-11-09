import logging
from typing import List, Dict, Any, Optional
from app.embeddings.embedder import get_embedder
from app.vectorstore.vector_db import vector_db
from app.config.settings import settings

logger = logging.getLogger("ecommerce_rag")


class EcommerceRetriever:
    """Semantic and metadata-aware vector retriever for e-commerce catalog and policies."""
    def __init__(self):
        self.embedder = get_embedder()
        self.vector_store = vector_db

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        k = top_k or settings.TOP_K
        th = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD

        # Generate query vector
        query_vector = self.embedder.embed_text(query)

        # Query vector store with wide candidate pool for hybrid re-ranking
        raw_results = self.vector_store.query(
            query_embedding=query_vector,
            n_results=max(50, k * 5),
            where=filter_metadata
        )

        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]
        ids = raw_results.get("ids", [[]])[0]

        all_candidates = []
        # Precompute query tokens for hybrid lexical scoring
        import re
        query_tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
        stop_words = {"the", "and", "for", "with", "what", "how", "can", "you", "are", "this", "that"}
        key_tokens = [t for t in query_tokens if t not in stop_words]

        for i in range(len(documents)):
            dist = distances[i] if i < len(distances) else 1.0
            vector_sim = max(0.0, 1.0 - dist)
            meta = metadatas[i] if i < len(metadatas) else {}
            doc_text = documents[i]

            # Lexical score based on keyword overlap across document, title, brand, category
            doc_search_space = f"{doc_text} {meta.get('title', '')} {meta.get('brand', '')} {meta.get('category', '')} {meta.get('section', '')}".lower()
            if key_tokens:
                token_hits = sum(1 for kt in key_tokens if kt in doc_search_space)
                lexical_score = token_hits / len(key_tokens)
            else:
                lexical_score = 0.0

            # Hybrid score: 40% vector similarity + 60% lexical match
            hybrid_score = round(0.4 * vector_sim + 0.6 * lexical_score, 4)

            item = {
                "id": ids[i] if i < len(ids) else f"doc_{i}",
                "text": doc_text,
                "metadata": meta,
                "similarity_score": hybrid_score,
                "vector_sim": vector_sim,
                "lexical_score": lexical_score,
                "distance": dist
            }
            all_candidates.append(item)

        # Sort descending by hybrid similarity score
        all_candidates.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Filter by threshold, or fallback to top candidates
        filtered = [c for c in all_candidates if c["similarity_score"] >= th]
        if not filtered and all_candidates:
            logger.info("Threshold filter yielded 0 results; falling back to top candidates.")
            return all_candidates[:k]

        return filtered[:k]
