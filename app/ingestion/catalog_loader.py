import json
import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.db.models import Product
from app.embeddings.embedder import get_embedder
from app.vectorstore.vector_db import vector_db

logger = logging.getLogger("ecommerce_rag")


class CatalogLoader:
    """Ingests e-commerce product catalogs into relational DB and semantic vector store."""
    def __init__(self, db_session: Session):
        self.db = db_session
        self.embedder = get_embedder()

    def format_product_semantic_text(self, item: Dict[str, Any]) -> str:
        """Create a rich, descriptive text block optimized for semantic vector search."""
        features = item.get("features", [])
        features_str = "\n".join([f"  * {f}" for f in features]) if features else "  * None listed"
        stock_str = "In Stock" if item.get("in_stock", True) else "Out of Stock"

        return (
            f"Product: {item.get('title')}\n"
            f"SKU: {item.get('sku')} | Brand: {item.get('brand')} | Category: {item.get('category')}\n"
            f"Price: ${item.get('price', 0.0):.2f} | Customer Rating: {item.get('rating', 0.0)}/5.0 ({item.get('review_count', 0)} reviews)\n"
            f"Availability: {stock_str} (Quantity: {item.get('stock_quantity', 0)})\n"
            f"Warranty: {item.get('warranty_months', 12)} Months\n"
            f"Key Highlights & Features:\n{features_str}\n"
            f"Description: {item.get('description', '')}"
        )

    def load_from_json(self, file_path: str) -> List[Product]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.ingest_products(data)

    def ingest_products(self, product_list: List[Dict[str, Any]]) -> List[Product]:
        created_products = []
        vector_ids = []
        vector_texts = []
        vector_metas = []

        for p_data in product_list:
            sku = p_data.get("sku")
            if not sku:
                continue

            # Check if product exists in SQL
            existing = self.db.query(Product).filter(Product.sku == sku).first()
            if existing:
                existing.title = p_data.get("title", existing.title)
                existing.brand = p_data.get("brand", existing.brand)
                existing.category = p_data.get("category", existing.category)
                existing.price = float(p_data.get("price", existing.price))
                existing.rating = float(p_data.get("rating", existing.rating))
                existing.review_count = int(p_data.get("review_count", existing.review_count))
                existing.in_stock = bool(p_data.get("in_stock", existing.in_stock))
                existing.stock_quantity = int(p_data.get("stock_quantity", existing.stock_quantity))
                existing.features_json = json.dumps(p_data.get("features", []))
                existing.description = p_data.get("description", existing.description)
                existing.warranty_months = int(p_data.get("warranty_months", existing.warranty_months))
                product_obj = existing
            else:
                product_obj = Product(
                    sku=sku,
                    title=p_data.get("title", "Untitled Product"),
                    brand=p_data.get("brand", "Generic"),
                    category=p_data.get("category", "General"),
                    price=float(p_data.get("price", 0.0)),
                    rating=float(p_data.get("rating", 0.0)),
                    review_count=int(p_data.get("review_count", 0)),
                    in_stock=bool(p_data.get("in_stock", True)),
                    stock_quantity=int(p_data.get("stock_quantity", 0)),
                    features_json=json.dumps(p_data.get("features", [])),
                    description=p_data.get("description", ""),
                    warranty_months=int(p_data.get("warranty_months", 12))
                )
                self.db.add(product_obj)

            created_products.append(product_obj)

            # Prepare semantic vector document
            semantic_text = self.format_product_semantic_text(p_data)
            v_id = f"prod_{sku}"
            vector_ids.append(v_id)
            vector_texts.append(semantic_text)
            vector_metas.append({
                "type": "product",
                "sku": sku,
                "title": p_data.get("title", ""),
                "brand": p_data.get("brand", ""),
                "category": p_data.get("category", ""),
                "price": float(p_data.get("price", 0.0)),
                "rating": float(p_data.get("rating", 0.0)),
                "in_stock": bool(p_data.get("in_stock", True)),
                "warranty_months": int(p_data.get("warranty_months", 12))
            })

        self.db.commit()

        # Embed and insert into vector store
        if vector_texts:
            logger.info(f"Generating embeddings for {len(vector_texts)} products...")
            embeddings = self.embedder.embed_batch(vector_texts)
            vector_db.add_documents(
                ids=vector_ids,
                embeddings=embeddings,
                documents=vector_texts,
                metadatas=vector_metas
            )
            logger.info(f"Successfully indexed {len(vector_texts)} products into ChromaDB.")

        return created_products
