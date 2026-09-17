from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from app.api.analytics import router as analytics_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.search import router as search_router
from app.config import get_settings
from app.database.session import init_db
from app.utils.logger import setup_logger

# Load application configuration settings from environment / .env
settings = get_settings()
logger = setup_logger(name=settings.app_name, level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for application startup and shutdown events."""
    logger.info("Starting up %s in '%s' environment...", settings.app_name, settings.environment)
    
    # Initialize database tables
    db_initialized = init_db()
    if db_initialized:
        logger.info("Database connection and tables verified.")
    else:
        logger.warning("Database could not be initialized automatically. Please check your MySQL settings.")

    # Initialize Snowflake analytical tables if configured
    try:
        from app.snowflake.connection import init_snowflake_tables
        init_snowflake_tables()
    except Exception as e:
        logger.warning("Snowflake analytical tables could not be initialized: %s", str(e))

    logger.info("Swagger docs available at http://%s:%s/docs", settings.host, settings.port)
    yield
    logger.info("Shutting down %s...", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI Insurance Document Intelligence Assistant provides automated document "
        "analysis, classification, and question-answering powered by AWS Bedrock, "
        "RAG, and Snowflake integration."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Register API Routers
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(analytics_router)


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to health check."""
    return {"message": f"Welcome to {settings.app_name}. Visit /docs for the API documentation."}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=(settings.environment == "development"),
    )
