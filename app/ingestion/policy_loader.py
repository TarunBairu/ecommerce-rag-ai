import os
import re
import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.db.models import StorePolicy
from app.embeddings.embedder import get_embedder
from app.vectorstore.vector_db import vector_db
from app.ingestion.chunker import TextChunker

logger = logging.getLogger("ecommerce_rag")


class PolicyLoader:
    """Ingests corporate store policies, terms, and FAQs into vector and SQL stores."""
    def __init__(self, db_session: Session):
        self.db = db_session
        self.embedder = get_embedder()
        self.chunker = TextChunker(chunk_size=600, chunk_overlap=50)

    def load_from_markdown(self, file_path: str) -> List[StorePolicy]:
        if not os.path.exists(file_path):
            logger.warning(f"Policy file not found: {file_path}")
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return self.ingest_policy_markdown(content, source_name=os.path.basename(file_path))

    def ingest_policy_markdown(self, markdown_text: str, source_name: str = "Store Policies") -> List[StorePolicy]:
        # Split by level 2 or level 3 headings to preserve granular subsections
        sections = re.split(r"(?m)^#{2,3}\s+", markdown_text)
        created_policies = []
        vector_ids = []
        vector_texts = []
        vector_metas = []

        for sec_idx, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue

            lines = section.split("\n")
            title_line = lines[0].strip()
            body_text = "\n".join(lines[1:]).strip()

            if not body_text:
                continue

            # Generate chunks for this section
            chunks = self.chunker.chunk_text(body_text, metadata={"section": title_line})

            for ch_idx, chunk in enumerate(chunks):
                chunk_text = f"Policy Section: {title_line}\n\n{chunk['text']}"
                policy_code = f"POL-{sec_idx+1}-{ch_idx+1}"

                # Store in SQL
                policy_obj = StorePolicy(
                    policy_code=policy_code,
                    title=f"{source_name} - {title_line}",
                    category="Store Policy",
                    section=title_line,
                    content=chunk_text
                )
                self.db.add(policy_obj)
                created_policies.append(policy_obj)

                # Vector indexing
                v_id = f"pol_{sec_idx+1}_{ch_idx+1}"
                vector_ids.append(v_id)
                vector_texts.append(chunk_text)
                vector_metas.append({
                    "type": "policy",
                    "policy_code": policy_code,
                    "title": title_line,
                    "section": title_line,
                    "source": source_name
                })

        self.db.commit()

        if vector_texts:
            logger.info(f"Indexing {len(vector_texts)} policy chunks into ChromaDB...")
            embeddings = self.embedder.embed_batch(vector_texts)
            vector_db.add_documents(
                ids=vector_ids,
                embeddings=embeddings,
                documents=vector_texts,
                metadatas=vector_metas
            )
            logger.info(f"Indexed {len(vector_texts)} policy chunks.")

        return created_policies
