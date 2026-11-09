from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    sku = Column(String(64), unique=True, index=True, nullable=False)
    title = Column(String(255), nullable=False, index=True)
    brand = Column(String(128), index=True)
    category = Column(String(128), index=True)
    price = Column(Float, nullable=False)
    rating = Column(Float, default=0.0)
    review_count = Column(Integer, default=0)
    in_stock = Column(Boolean, default=True)
    stock_quantity = Column(Integer, default=0)
    features_json = Column(Text, default="[]")
    description = Column(Text)
    warranty_months = Column(Integer, default=12)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        import json
        return {
            "id": self.id,
            "sku": self.sku,
            "title": self.title,
            "brand": self.brand,
            "category": self.category,
            "price": self.price,
            "rating": self.rating,
            "review_count": self.review_count,
            "in_stock": self.in_stock,
            "stock_quantity": self.stock_quantity,
            "features": json.loads(self.features_json) if self.features_json else [],
            "description": self.description,
            "warranty_months": self.warranty_months,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class StorePolicy(Base):
    __tablename__ = "store_policies"

    id = Column(Integer, primary_key=True, index=True)
    policy_code = Column(String(64), index=True)
    title = Column(String(255), nullable=False)
    category = Column(String(128), index=True)
    section = Column(String(128))
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    intent = Column(String(64), default="general")
    retrieved_chunks = Column(Integer, default=0)
    response_text = Column(Text, nullable=False)
    llm_model = Column(String(128))
    latency_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
