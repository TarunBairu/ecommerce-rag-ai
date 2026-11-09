import os
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from app.db.postgres import get_db
from app.db.models import StorePolicy
from app.ingestion.policy_loader import PolicyLoader

router = APIRouter(prefix="/api/v1/policies", tags=["Store Policies"])


@router.get("", summary="List all indexed store policies")
def list_policies(db: Session = Depends(get_db)):
    policies = db.query(StorePolicy).all()
    return {
        "count": len(policies),
        "policies": [
            {
                "id": p.id,
                "policy_code": p.policy_code,
                "title": p.title,
                "category": p.category,
                "section": p.section,
                "content_preview": p.content[:200] + "..." if len(p.content) > 200 else p.content
            }
            for p in policies
        ]
    }


@router.post("/upload", summary="Upload and index Markdown store policy")
async def upload_policy(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not (file.filename.endswith(".md") or file.filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only Markdown (.md) or Text (.txt) policy files are supported.")

    content = await file.read()
    try:
        text_str = content.decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode text: {e}")

    loader = PolicyLoader(db)
    created = loader.ingest_policy_markdown(text_str, source_name=file.filename)
    return {
        "message": f"Successfully parsed and indexed {len(created)} policy chunks.",
        "chunks_indexed": len(created)
    }


@router.post("/reindex", summary="Re-index bundled sample store policies")
def reindex_sample_policies(db: Session = Depends(get_db)):
    policy_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "policies", "store_policies.md"
    )
    if not os.path.exists(policy_path):
        raise HTTPException(status_code=404, detail="Default store_policies.md not found on disk.")

    loader = PolicyLoader(db)
    created = loader.load_from_markdown(policy_path)
    return {
        "message": "Sample store policies indexed successfully.",
        "count": len(created)
    }
