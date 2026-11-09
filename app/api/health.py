from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.postgres import get_db
from app.vectorstore.vector_db import vector_db
from app.config.settings import settings

router = APIRouter(prefix="/health", tags=["Health & Diagnostics"])


@router.get("", summary="Liveness probe")
def liveness():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "env": settings.APP_ENV,
        "port": settings.APP_PORT
    }


@router.get("/ready", summary="Readiness probe verifying DB and Vector Store")
def readiness(db: Session = Depends(get_db)):
    db_status = "ok"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {e}"

    vector_count = 0
    vector_status = "ok"
    try:
        vector_count = vector_db.count()
    except Exception as e:
        vector_status = f"unhealthy: {e}"

    all_ready = (db_status == "ok" and vector_status == "ok")
    return {
        "ready": all_ready,
        "database": db_status,
        "vector_store": vector_status,
        "indexed_vector_count": vector_count,
        "llm_provider": settings.LLM_PROVIDER,
        "embedding_provider": settings.EMBEDDING_PROVIDER
    }
