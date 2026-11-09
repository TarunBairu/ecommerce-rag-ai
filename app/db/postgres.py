import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config.settings import settings
from app.db.models import Base

logger = logging.getLogger("ecommerce_rag")


def get_engine():
    """Create database engine with SQLite fallback."""
    # Ensure local data directory exists for SQLite
    os.makedirs("./data", exist_ok=True)
    try:
        if settings.DATABASE_URL.startswith("postgresql"):
            logger.info(f"Attempting PostgreSQL connection: {settings.DATABASE_URL}")
            eng = create_engine(
                settings.DATABASE_URL,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                connect_args={"connect_timeout": 3}
            )
            with eng.connect() as conn:
                logger.info("Successfully connected to primary PostgreSQL.")
                return eng
        else:
            return create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
    except Exception as e:
        logger.warning(f"PostgreSQL connection failed ({e}). Falling back to SQLite: {settings.SQLITE_FALLBACK_URL}")
        return create_engine(
            settings.SQLITE_FALLBACK_URL,
            connect_args={"check_same_thread": False}
        )


engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema initialized successfully.")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")


# Auto-initialize tables on module load
init_db()
