"""Dedicated Snowflake connection and session lifecycle module.

Provides production-ready connection management, context-managed cursors,
health checks, and schema initialization adhering to the project's
architectural separation between MySQL (OLTP) and Snowflake (OLAP).
"""

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional
import snowflake.connector
from snowflake.connector.errors import (
    DatabaseError,
    Error as SnowflakeError,
    InterfaceError,
    OperationalError,
    ProgrammingError,
)

from app.config import Settings, get_settings
from app.utils.logger import setup_logger

logger = setup_logger("snowflake_connection")


class SnowflakeConnectionManager:
    """Manages connections to the Snowflake Data Cloud warehouse."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def is_configured(self) -> bool:
        """Check whether the minimum required credentials for Snowflake are present.

        Returns:
            bool: True if account, user, and password are all populated.
        """
        return bool(
            self.settings.snowflake_account
            and self.settings.snowflake_account.strip()
            and self.settings.snowflake_user
            and self.settings.snowflake_user.strip()
            and self.settings.snowflake_password
            and self.settings.snowflake_password.strip()
        )

    def get_connection(self) -> snowflake.connector.SnowflakeConnection:
        """Establish and return a new connection to Snowflake using configured credentials.

        Raises:
            ValueError: If required Snowflake credentials are not configured.
            RuntimeError: If connection cannot be established.

        Returns:
            snowflake.connector.SnowflakeConnection: Active Snowflake connection.
        """
        if not self.is_configured():
            missing = []
            if not self.settings.snowflake_account:
                missing.append("SNOWFLAKE_ACCOUNT")
            if not self.settings.snowflake_user:
                missing.append("SNOWFLAKE_USER")
            if not self.settings.snowflake_password:
                missing.append("SNOWFLAKE_PASSWORD")
            msg = f"Snowflake credentials incomplete. Missing: {', '.join(missing)}"
            logger.warning(msg)
            raise ValueError(msg)

        logger.info(
            "Connecting to Snowflake (account='%s', user='%s', warehouse='%s', db='%s', schema='%s', role='%s')",
            self.settings.snowflake_account,
            self.settings.snowflake_user,
            self.settings.snowflake_warehouse,
            self.settings.snowflake_database,
            self.settings.snowflake_schema,
            self.settings.snowflake_role,
        )

        try:
            conn = snowflake.connector.connect(
                account=self.settings.snowflake_account.strip(),
                user=self.settings.snowflake_user.strip(),
                password=self.settings.snowflake_password.strip(),
                warehouse=self.settings.snowflake_warehouse.strip() if self.settings.snowflake_warehouse else None,
                database=self.settings.snowflake_database.strip() if self.settings.snowflake_database else None,
                schema=self.settings.snowflake_schema.strip() if self.settings.snowflake_schema else None,
                role=self.settings.snowflake_role.strip() if self.settings.snowflake_role else None,
                client_session_keep_alive=True,
                autocommit=True,
            )
            return conn
        except (ProgrammingError, OperationalError, InterfaceError, DatabaseError) as e:
            logger.error("Snowflake connection failed [%s]: %s", type(e).__name__, str(e))
            raise RuntimeError(f"Snowflake connection failed: [{type(e).__name__}] {str(e)}") from e
        except Exception as e:
            logger.error("Unexpected error connecting to Snowflake: %s", str(e))
            raise RuntimeError(f"Could not connect to Snowflake: {str(e)}") from e

    @contextmanager
    def get_cursor(self) -> Generator[snowflake.connector.cursor.SnowflakeCursor, None, None]:
        """Context manager providing a safe, managed cursor with automatic cleanup.

        Yields:
            SnowflakeCursor: Active cursor for executing SQL statements.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
        finally:
            try:
                cursor.close()
            except Exception as e:
                logger.debug("Error closing Snowflake cursor: %s", str(e))
            try:
                conn.close()
            except Exception as e:
                logger.debug("Error closing Snowflake connection: %s", str(e))

    def verify_connection(self) -> Dict[str, Any]:
        """Execute diagnostic queries to verify live Snowflake connectivity.

        Returns:
            dict: Diagnostic metadata including version, role, warehouse, and database.
        """
        if not self.is_configured():
            return {
                "status": "UNCONFIGURED",
                "message": "Snowflake credentials (SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD) not configured in .env",
            }

        try:
            with self.get_cursor() as cur:
                cur.execute("""
                    SELECT
                        CURRENT_VERSION() AS version,
                        CURRENT_ACCOUNT() AS account,
                        CURRENT_ROLE() AS role,
                        CURRENT_DATABASE() AS database,
                        CURRENT_SCHEMA() AS schema,
                        CURRENT_WAREHOUSE() AS warehouse,
                        CURRENT_TIMESTAMP() AS server_time;
                """)
                row = cur.fetchone()
                columns = [col[0].lower() for col in cur.description]
                result = dict(zip(columns, row))
                result["status"] = "CONNECTED"
                logger.info("Snowflake connection verified: version=%s, role=%s", result.get("version"), result.get("role"))
                return result
        except Exception as e:
            logger.error("Snowflake connection verification failed: %s", str(e))
            return {
                "status": "ERROR",
                "error_type": type(e).__name__,
                "error_message": str(e),
            }

    def init_tables(self) -> bool:
        """Create the necessary analytical tables in Snowflake if they do not exist.

        Tables created:
        1. DOCUMENT_ANALYTICS: OLAP document summary metrics (processing duration, chunks, queries, risk).
        2. POLICY_METRICS: Structured insurance policy parameters extracted for analytical reporting.
        3. QUERY_ANALYTICS: Query engagement and retrieval analytics.

        Returns:
            bool: True if tables were created or verified successfully.
        """
        if not self.is_configured():
            logger.warning("Skipping Snowflake table init: credentials not configured.")
            return False

        ddl_document_analytics = """
        CREATE TABLE IF NOT EXISTS document_analytics (
            document_id INTEGER PRIMARY KEY,
            document_type VARCHAR(100),
            file_name VARCHAR(255),
            processing_time FLOAT,
            chunk_count INTEGER,
            query_count INTEGER DEFAULT 0,
            risk_category VARCHAR(50) DEFAULT 'LOW',
            status VARCHAR(50) DEFAULT 'PROCESSED',
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            synced_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        );
        """

        ddl_policy_metrics = """
        CREATE TABLE IF NOT EXISTS policy_metrics (
            policy_id VARCHAR(100) PRIMARY KEY,
            document_id INTEGER,
            policy_type VARCHAR(100),
            deductible NUMBER(12, 2),
            coverage_amount NUMBER(14, 2),
            premium NUMBER(12, 2),
            risk_level VARCHAR(50),
            extracted_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        );
        """

        ddl_query_analytics = """
        CREATE TABLE IF NOT EXISTS query_analytics (
            query_id VARCHAR(64) PRIMARY KEY,
            document_id INTEGER,
            question_text VARCHAR(2000),
            question_category VARCHAR(100),
            answer_length INTEGER,
            retrieved_chunks_count INTEGER,
            queried_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        );
        """

        try:
            logger.info("Verifying and initializing Snowflake analytical tables...")
            with self.get_cursor() as cur:
                cur.execute(ddl_document_analytics)
                cur.execute(ddl_policy_metrics)
                cur.execute(ddl_query_analytics)
            logger.info("Snowflake analytical tables (document_analytics, policy_metrics, query_analytics) verified.")
            return True
        except Exception as e:
            logger.error("Failed to initialize Snowflake tables: %s", str(e))
            raise


# Global singleton manager
_manager: Optional[SnowflakeConnectionManager] = None


def get_snowflake_manager(settings: Optional[Settings] = None) -> SnowflakeConnectionManager:
    """Return a singleton instance of SnowflakeConnectionManager."""
    global _manager
    if _manager is None or settings is not None:
        _manager = SnowflakeConnectionManager(settings=settings or get_settings())
    return _manager


def get_snowflake_connection() -> snowflake.connector.SnowflakeConnection:
    """Convenience accessor to get a direct Snowflake connection."""
    return get_snowflake_manager().get_connection()


@contextmanager
def get_snowflake_cursor() -> Generator[snowflake.connector.cursor.SnowflakeCursor, None, None]:
    """Convenience context manager for executing Snowflake queries."""
    with get_snowflake_manager().get_cursor() as cur:
        yield cur


def verify_snowflake_connection() -> Dict[str, Any]:
    """Convenience function to test and verify Snowflake connectivity."""
    return get_snowflake_manager().verify_connection()


def init_snowflake_tables() -> bool:
    """Convenience function to initialize Snowflake tables."""
    return get_snowflake_manager().init_tables()
