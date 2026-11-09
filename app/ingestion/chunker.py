from typing import List, Dict, Any
import re


class TextChunker:
    """Recursive character chunker that preserves semantic boundaries."""
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        # Preferred split boundaries: paragraphs, sentences, words
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 1 <= self.chunk_size:
                current_chunk = f"{current_chunk}\n\n{para}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # If paragraph itself is larger than chunk_size, split by sentences
                if len(para) > self.chunk_size:
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    sub_chunk = ""
                    for s in sentences:
                        if len(sub_chunk) + len(s) + 1 <= self.chunk_size:
                            sub_chunk = f"{sub_chunk} {s}".strip()
                        else:
                            if sub_chunk:
                                chunks.append(sub_chunk)
                            sub_chunk = s
                    current_chunk = sub_chunk
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        # Apply overlap & construct chunk dictionaries
        result = []
        for i, ch in enumerate(chunks):
            chunk_dict = {
                "chunk_index": i,
                "text": ch,
                "token_estimate": max(1, len(ch) // 4),
                **metadata
            }
            result.append(chunk_dict)

        return result
