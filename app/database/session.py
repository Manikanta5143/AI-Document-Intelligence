from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings
from app.utils.logger import setup_logger

settings = get_settings()
logger = setup_logger(name="database", level=settings.log_level)


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy database models."""
    pass


# Configure SQLAlchemy Engine
engine_kwargs = {
    "pool_pre_ping": True,
    "echo": False,
}
if settings.database_url.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs["pool_recycle"] = 3600

engine = create_engine(
    settings.database_url,
    **engine_kwargs,
)

# Session factory for generating database sessions
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a transactional SQLAlchemy session per request.

    Yields:
        Session: Database session for query execution.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> bool:
    """Initialize database tables defined in SQLAlchemy models.

    Returns:
        bool: True if tables were successfully created/verified, False otherwise.
    """
    try:
        # Import models here to ensure all models are registered with Base.metadata
        from app import models  # noqa: F401

        logger.info("Initializing database tables using Base.metadata...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified successfully.")
        return True
    except OperationalError as e:
        logger.warning(
            "Could not connect to MySQL database at %s:%s. "
            "Please ensure MySQL is running, the database '%s' exists, "
            "and credentials in .env are correct. Detail: %s",
            settings.mysql_host,
            settings.mysql_port,
            settings.mysql_database,
            e,
        )
        return False
    except SQLAlchemyError as e:
        logger.error("An error occurred during database initialization: %s", e)
        return False
