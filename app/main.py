import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.db.postgres import init_db, SessionLocal
from app.db.models import Product, StorePolicy
from app.ingestion.catalog_loader import CatalogLoader
from app.ingestion.policy_loader import PolicyLoader

from app.middleware.rate_limiter import RateLimiterMiddleware
from app.middleware.logger import RequestLoggerMiddleware
from app.middleware.error_handler import ErrorHandlerMiddleware

from app.api.health import router as health_router
from app.api.catalog import router as catalog_router
from app.api.policies import router as policies_router
from app.api.chat import router as chat_router

logger = logging.getLogger("ecommerce_rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DB schema and seed initial sample catalog & policies if empty
    logger.info("Initializing E-Commerce RAG application lifespan...")
    init_db()

    db = SessionLocal()
    try:
        # Check if products need seeding
        prod_count = db.query(Product).count()
        if prod_count == 0:
            catalog_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "catalog", "products.json"
            )
            if os.path.exists(catalog_file):
                logger.info("Seeding initial product catalog from products.json...")
                loader = CatalogLoader(db)
                loader.load_from_json(catalog_file)

        # Check if policies need seeding
        policy_count = db.query(StorePolicy).count()
        if policy_count == 0:
            policy_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "policies", "store_policies.md"
            )
            if os.path.exists(policy_file):
                logger.info("Seeding initial store policies from store_policies.md...")
                ploader = PolicyLoader(db)
                ploader.load_from_markdown(policy_file)
    except Exception as e:
        logger.error(f"Error during startup data seeding: {e}")
    finally:
        db.close()

    logger.info(f"ShopWise AI successfully started on port {settings.APP_PORT}")
    yield
    # Shutdown
    logger.info("ShopWise AI shutting down cleanly.")


app = FastAPI(
    title=settings.APP_NAME,
    description="Production E-Commerce Retrieval-Augmented Generation (RAG) Platform for product discovery and customer policy support.",
    version="1.0.0",
    lifespan=lifespan
)

# Attach Middlewares in order: RateLimiter -> RequestLogger -> ErrorHandler -> CORS
app.add_middleware(RateLimiterMiddleware)
app.add_middleware(RequestLoggerMiddleware)
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(health_router)
app.include_router(catalog_router)
app.include_router(policies_router)
app.include_router(chat_router)


# Web Chat UI Endpoints
@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
@app.get("/chat", response_class=HTMLResponse, tags=["Web UI"])
def get_chat_ui():
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=settings.DEBUG)
