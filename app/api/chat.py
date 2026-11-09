from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.db.postgres import get_db
from app.rag.pipeline import EcommerceRAGPipeline

router = APIRouter(prefix="/api/v1/chat", tags=["E-Commerce Copilot Chat"])


class ChatQueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000, description="Shopper query or customer service question")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of catalog/policy sources to retrieve")


@router.post("/query", summary="Ask ShopWise Copilot for product recommendations or policy advice")
def query_copilot(
    request: ChatQueryRequest,
    db: Session = Depends(get_db)
):
    pipeline = EcommerceRAGPipeline(db_session=db)
    try:
        result = pipeline.run(query=request.query, top_k=request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@router.post("/query/stream", summary="Stream response tokens via Server-Sent Events (SSE)")
def stream_query_copilot(
    request: ChatQueryRequest,
    db: Session = Depends(get_db)
):
    pipeline = EcommerceRAGPipeline(db_session=db)

    def event_stream():
        for token in pipeline.stream_run(query=request.query, top_k=request.top_k):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
