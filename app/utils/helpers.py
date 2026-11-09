import hashlib
import re
from typing import Optional


def hash_content(content: str) -> str:
    """Generate SHA-256 hash of text content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def format_currency(amount: Optional[float]) -> str:
    """Format floating point currency to USD string."""
    if amount is None:
        return "$0.00"
    return f"${amount:,.2f}"


def estimate_tokens(text: str) -> int:
    """Heuristic approximation: 1 token ≈ 4 characters."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def normalize_text(text: str) -> str:
    """Clean extra spaces and standardize line endings."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()
