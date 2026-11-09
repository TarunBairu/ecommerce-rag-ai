import os
import json
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.db.postgres import get_db
from app.db.models import Product
from app.ingestion.catalog_loader import CatalogLoader

router = APIRouter(prefix="/api/v1/catalog", tags=["Product Catalog"])


@router.get("/products", summary="List products with optional search and category filters")
def list_products(
    category: Optional[str] = Query(None, description="Filter by category"),
    search: Optional[str] = Query(None, description="Search by title or brand"),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    query = db.query(Product)
    if category:
        query = query.filter(Product.category.ilike(f"%{category}%"))
    if search:
        query = query.filter(
            (Product.title.ilike(f"%{search}%")) | (Product.brand.ilike(f"%{search}%"))
        )
    products = query.limit(limit).all()
    return {
        "count": len(products),
        "products": [p.to_dict() for p in products]
    }


@router.get("/products/{sku}", summary="Get product details by SKU")
def get_product(sku: str, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.sku == sku).first()
    if not product:
        raise HTTPException(status_code=404, detail=f"Product with SKU '{sku}' not found.")
    return product.to_dict()


@router.post("/upload", summary="Upload and index JSON product catalog")
async def upload_catalog(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only JSON catalog files are supported.")

    content = await file.read()
    try:
        data = json.loads(content.decode("utf-8"))
        if not isinstance(data, list):
            raise HTTPException(status_code=400, detail="JSON must be an array of product objects.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {e}")

    loader = CatalogLoader(db)
    created = loader.ingest_products(data)
    return {
        "message": f"Successfully ingested and indexed {len(created)} products.",
        "imported_count": len(created)
    }


@router.post("/reindex", summary="Re-index bundled sample product catalog")
def reindex_sample_catalog(db: Session = Depends(get_db)):
    catalog_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "catalog", "products.json"
    )
    if not os.path.exists(catalog_path):
        raise HTTPException(status_code=404, detail="Default products.json not found on disk.")

    loader = CatalogLoader(db)
    created = loader.load_from_json(catalog_path)
    return {
        "message": "Sample catalog indexed successfully.",
        "count": len(created)
    }
