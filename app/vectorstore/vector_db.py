import os
import json
import math
import logging
from typing import List, Dict, Any, Optional
from app.config.settings import settings

logger = logging.getLogger("ecommerce_rag")


class LocalVectorCollection:
    """Persistent JSON fallback vector collection for local testing and lightweight execution."""
    def __init__(self, persist_path: str):
        self.persist_path = persist_path
        self.items: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    self.items = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load vector store from {self.persist_path}: {e}")
                self.items = {}

    def _save(self):
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        try:
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(self.items, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist vector store: {e}")

    def add(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, Any]]):
        for i, doc_id in enumerate(ids):
            self.items[doc_id] = {
                "id": doc_id,
                "embedding": embeddings[i],
                "document": documents[i],
                "metadata": metadatas[i] if metadatas and i < len(metadatas) else {},
            }
        self._save()

    def query(self, query_embeddings: List[List[float]], n_results: int = 4, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        results_ids = []
        results_docs = []
        results_metas = []
        results_dists = []

        for q_emb in query_embeddings:
            scored = []
            for doc_id, item in self.items.items():
                meta = item.get("metadata", {})
                if where:
                    match = True
                    for k, v in where.items():
                        if meta.get(k) != v:
                            match = False
                            break
                    if not match:
                        continue

                d_emb = item["embedding"]
                # Cosine distance = 1 - cosine_similarity
                dot = sum(a * b for a, b in zip(q_emb, d_emb))
                norm_q = math.sqrt(sum(a * a for a, b in zip(q_emb, q_emb))) or 1.0
                norm_d = math.sqrt(sum(b * b for a, b in zip(d_emb, d_emb))) or 1.0
                sim = max(-1.0, min(1.0, dot / (norm_q * norm_d)))
                dist = 1.0 - sim
                scored.append((dist, item))

            scored.sort(key=lambda x: x[0])
            top = scored[:n_results]

            results_ids.append([x[1]["id"] for x in top])
            results_docs.append([x[1]["document"] for x in top])
            results_metas.append([x[1]["metadata"] for x in top])
            results_dists.append([x[0] for x in top])

        return {
            "ids": results_ids,
            "documents": results_docs,
            "metadatas": results_metas,
            "distances": results_dists,
        }

    def count(self) -> int:
        return len(self.items)

    def delete(self, ids: List[str]):
        for doc_id in ids:
            self.items.pop(doc_id, None)
        self._save()

    def reset(self):
        self.items = {}
        self._save()


class VectorDBManager:
    def __init__(self):
        self.persist_dir = settings.VECTOR_STORE_DIR
        self.collection_name = settings.VECTOR_COLLECTION_NAME
        self.collection = None
        self._init_client()

    def _init_client(self):
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            os.makedirs(self.persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Initialized ChromaDB collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"ChromaDB initialization failed ({e}); using LocalVectorCollection fallback.")
            fallback_file = os.path.join(self.persist_dir, f"{self.collection_name}.json")
            self.collection = LocalVectorCollection(fallback_file)

    def add_documents(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, Any]]):
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def query(self, query_embedding: List[float], n_results: int = 4, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

    def count(self) -> int:
        return self.collection.count()

    def delete_by_ids(self, ids: List[str]):
        self.collection.delete(ids=ids)

    def reset(self):
        if hasattr(self.collection, "reset"):
            self.collection.reset()
        elif hasattr(self, "client") and self.client:
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                pass


vector_db = VectorDBManager()
