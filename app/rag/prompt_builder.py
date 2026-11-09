from typing import List, Dict, Any


class EcommercePromptBuilder:
    """Constructs grounded e-commerce prompts with strict commercial guardrails."""

    SYSTEM_PROMPT = """You are ShopWise AI, the intelligent shopping and customer support copilot for ShopWise E-Commerce.

YOUR RESPONSIBILITIES:
1. Product Discovery & Recommendations: Help shoppers find ideal products based on their budget, features, specifications, and brand preferences.
2. Customer Service & Policies: Accurately explain return windows, shipping options, warranty terms, price matches, and order procedures.

CRITICAL COMMERCIAL GUARDRAILS:
- Answer ONLY using the factual product details and store policies provided in the context below.
- Never hallucinate non-existent products, fake discount codes, or untrue inventory statuses.
- For product recommendations, clearly specify: Product Name, Brand, Price, Rating, and Why it matches the user's needs.
- If a product is out of stock, explicitly alert the shopper that it is currently unavailable.
- For policy inquiries (returns, shipping, warranties), cite the exact rules, timeframes (e.g., 30 days unopened / 14 days opened electronics), and any applicable fees.
- Ground all facts with clear citations corresponding to the sources provided, e.g. [Source 1], [Source 2].
- If the question cannot be answered using the provided sources, reply: "I do not have enough verified catalog or policy information to answer this question. Please contact our 24/7 customer care team."
"""

    def build_prompt(self, query: str, context_chunks: List[Dict[str, Any]], intent: str = "general") -> str:
        formatted_sources = []
        for i, chunk in enumerate(context_chunks):
            meta = chunk.get("metadata", {})
            src_type = meta.get("type", "source").upper()
            identifier = meta.get("sku") or meta.get("section") or meta.get("title") or f"Item-{i+1}"
            text = chunk.get("text", "")
            formatted_sources.append(f"--- [Source {i+1}: {src_type} - {identifier}] ---\n{text}")

        context_str = "\n\n".join(formatted_sources) if formatted_sources else "No matching catalog or policy records found."

        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"=== VERIFIED STORE & CATALOG CONTEXT ===\n"
            f"{context_str}\n\n"
            f"=== SHOPPER QUERY ===\n"
            f"User Query (Intent: {intent}): {query}\n\n"
            f"=== ASSISTANT RESPONSE ==="
        )
        return prompt
