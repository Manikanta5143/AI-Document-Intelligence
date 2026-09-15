from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.config import Settings, get_settings
from app.schemas.analytics import (
    AnalyticsSummaryResponse,
    DocumentAnalyticsRecord,
    DocumentsByTypeItem,
    ProcessingTimeMetrics,
    QuestionsByDocumentItem,
    RiskCategoryDistributionItem,
)
from app.snowflake.connection import (
    SnowflakeConnectionManager,
    get_snowflake_manager,
)
from app.utils.logger import setup_logger

logger = setup_logger("snowflake_analytics")


class BaseAnalyticsService(ABC):
    """Abstract base class defining the analytical interface for insurance document intelligence.

    Adheres strictly to the architectural separation:
    - MySQL: Operational / Transactional data
    - Snowflake / Analytics Service: OLAP reporting and aggregations
    """

    @abstractmethod
    def init_schema(self) -> bool:
        """Create or verify analytical tables and schemas."""
        pass

    @abstractmethod
    def sync_document_analytics(self, record: DocumentAnalyticsRecord) -> bool:
        """Insert or upsert an analytical record for a document."""
        pass

    @abstractmethod
    def record_query_event(self, document_id: int) -> bool:
        """Increment query count engagement metric for a document."""
        pass

    @abstractmethod
    def get_analytics_summary(self) -> AnalyticsSummaryResponse:
        """Execute the 5 core analytical queries and return consolidated metrics:
        1. Number of documents by type
        2. Number of questions by document
        3. Number of processed documents
        4. Risk-category distribution
        5. Average processing time
        """
        pass


class SnowflakeAnalyticsService(BaseAnalyticsService):
    """Integrates with Snowflake data warehouse via the centralized SnowflakeConnectionManager.

    Reads credentials strictly from application environment settings without hard-coding,
    manages secure connections, and executes analytical OLAP queries.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        connection_manager: Optional[SnowflakeConnectionManager] = None,
    ):
        self.settings = settings or get_settings()
        self.connection_manager = connection_manager or get_snowflake_manager(settings=self.settings)

    def init_schema(self) -> bool:
        """Initialize required analytical tables in Snowflake."""
        return self.connection_manager.init_tables()

    def sync_document_analytics(self, record: DocumentAnalyticsRecord) -> bool:
        """Upsert document analytical metrics into Snowflake via MERGE."""
        logger.info(
            "Syncing document ID %d analytics to Snowflake (type='%s', chunks=%d, time=%.2fs, risk='%s')",
            record.document_id,
            record.document_type,
            record.chunk_count,
            record.processing_time,
            record.risk_category,
        )
        merge_sql = """
        MERGE INTO document_analytics AS target
        USING (SELECT %(doc_id)s AS document_id, %(doc_type)s AS document_type,
                      %(proc_time)s AS processing_time, %(chunks)s AS chunk_count,
                      %(queries)s AS query_count, %(risk)s AS risk_category,
                      %(created)s AS created_at, CURRENT_TIMESTAMP() AS synced_at) AS src
        ON target.document_id = src.document_id
        WHEN MATCHED THEN
            UPDATE SET target.document_type = src.document_type,
                       target.processing_time = src.processing_time,
                       target.chunk_count = src.chunk_count,
                       target.risk_category = src.risk_category,
                       target.synced_at = src.synced_at
        WHEN NOT MATCHED THEN
            INSERT (document_id, document_type, processing_time, chunk_count, query_count, risk_category, created_at, synced_at)
            VALUES (src.document_id, src.document_type, src.processing_time, src.chunk_count, src.query_count, src.risk_category, src.created_at, src.synced_at);
        """
        params = {
            "doc_id": record.document_id,
            "doc_type": record.document_type,
            "proc_time": record.processing_time,
            "chunks": record.chunk_count,
            "queries": record.query_count,
            "risk": record.risk_category,
            "created": record.created_at or datetime.utcnow(),
        }
        try:
            with self.connection_manager.get_cursor() as cursor:
                cursor.execute(merge_sql, params)
            logger.info("Successfully merged analytics record for doc ID %d in Snowflake.", record.document_id)
            return True
        except Exception as e:
            logger.error("Failed to sync analytics record to Snowflake: %s", str(e))
            raise

    def record_query_event(self, document_id: int) -> bool:
        """Increment query_count in Snowflake for user engagement analytics."""
        update_sql = """
        UPDATE document_analytics
        SET query_count = query_count + 1,
            synced_at = CURRENT_TIMESTAMP()
        WHERE document_id = %(doc_id)s;
        """
        try:
            with self.connection_manager.get_cursor() as cursor:
                cursor.execute(update_sql, {"doc_id": document_id})
            return True
        except Exception as e:
            logger.warning("Could not increment Snowflake query count for document ID %d: %s", document_id, str(e))
            return False

    def get_analytics_summary(self) -> AnalyticsSummaryResponse:
        """Execute the 5 required analytical queries in Snowflake."""
        logger.info("Executing analytical aggregations in Snowflake...")
        try:
            with self.connection_manager.get_cursor() as cursor:
                # Query 1: Number of documents by type
                cursor.execute("""
                    SELECT document_type, COUNT(*) AS doc_count
                    FROM document_analytics
                    GROUP BY document_type
                    ORDER BY doc_count DESC, document_type ASC;
                """)
                docs_by_type = [
                    DocumentsByTypeItem(document_type=row[0] or "Unknown", doc_count=row[1])
                    for row in cursor.fetchall()
                ]

                # Query 2: Number of questions by document
                cursor.execute("""
                    SELECT document_id, document_type, query_count
                    FROM document_analytics
                    ORDER BY query_count DESC, document_id ASC;
                """)
                questions_by_doc = [
                    QuestionsByDocumentItem(
                        document_id=row[0],
                        document_type=row[1] or "Unknown",
                        query_count=row[2] or 0,
                    )
                    for row in cursor.fetchall()
                ]

                # Query 3: Number of processed documents
                cursor.execute("""
                    SELECT COUNT(*) AS total_processed
                    FROM document_analytics
                    WHERE processing_time > 0;
                """)
                total_processed = cursor.fetchone()[0] or 0

                # Query 4: Risk-category distribution
                cursor.execute("""
                    SELECT risk_category, COUNT(*) AS category_count
                    FROM document_analytics
                    GROUP BY risk_category
                    ORDER BY category_count DESC, risk_category ASC;
                """)
                risk_dist = [
                    RiskCategoryDistributionItem(risk_category=row[0] or "LOW", category_count=row[1])
                    for row in cursor.fetchall()
                ]

                # Query 5: Average processing time
                cursor.execute("""
                    SELECT
                        COALESCE(AVG(processing_time), 0.0) AS avg_time,
                        COALESCE(MIN(processing_time), 0.0) AS min_time,
                        COALESCE(MAX(processing_time), 0.0) AS max_time
                    FROM document_analytics;
                """)
                time_row = cursor.fetchone()
                proc_metrics = ProcessingTimeMetrics(
                    avg_processing_time=round(float(time_row[0] or 0.0), 2),
                    min_processing_time=round(float(time_row[1] or 0.0), 2),
                    max_processing_time=round(float(time_row[2] or 0.0), 2),
                )

            return AnalyticsSummaryResponse(
                engine="Snowflake Live OLAP",
                total_processed_documents=total_processed,
                documents_by_type=docs_by_type,
                questions_by_document=questions_by_doc,
                risk_distribution=risk_dist,
                processing_time_metrics=proc_metrics,
            )
        except Exception as e:
            logger.error("Error executing Snowflake analytics aggregations: %s", str(e))
            raise


class MockAnalyticsService(BaseAnalyticsService):
    """In-memory analytics engine for local development, offline mode, and unit tests.

    Mirrors Snowflake table structure and reproduces exact SQL aggregation semantics.
    """

    def __init__(self):
        # In-memory analytical table keyed by document_id
        self._table: Dict[int, Dict[str, Any]] = {}
        logger.info("Initialized MockAnalyticsService in-memory engine.")

    def init_schema(self) -> bool:
        """Simulate table verification."""
        logger.info("MockAnalyticsService: Verified document_analytics schema.")
        return True

    def sync_document_analytics(self, record: DocumentAnalyticsRecord) -> bool:
        """Upsert record in simulated analytics table."""
        doc_id = record.document_id
        if doc_id in self._table:
            # Update metrics while preserving existing queries if not provided
            existing = self._table[doc_id]
            existing["document_type"] = record.document_type
            existing["processing_time"] = record.processing_time
            existing["chunk_count"] = record.chunk_count
            existing["risk_category"] = record.risk_category
            if record.query_count > 0:
                existing["query_count"] = record.query_count
        else:
            self._table[doc_id] = {
                "document_id": record.document_id,
                "document_type": record.document_type,
                "processing_time": record.processing_time,
                "chunk_count": record.chunk_count,
                "query_count": record.query_count,
                "risk_category": record.risk_category,
                "created_at": record.created_at or datetime.utcnow(),
            }
        logger.info(
            "MockAnalyticsService: Synced doc_id=%d (type='%s', risk='%s', time=%.2fs)",
            doc_id,
            record.document_type,
            record.risk_category,
            record.processing_time,
        )
        return True

    def record_query_event(self, document_id: int) -> bool:
        """Increment question count for a document."""
        if document_id in self._table:
            self._table[document_id]["query_count"] += 1
            logger.info(
                "MockAnalyticsService: Incremented query_count for doc_id=%d to %d",
                document_id,
                self._table[document_id]["query_count"],
            )
            return True
        return False

    def get_analytics_summary(self) -> AnalyticsSummaryResponse:
        """Perform simulated OLAP aggregations across the analytical records."""
        records = list(self._table.values())

        # Query 1: Number of documents by type
        type_counts: Dict[str, int] = {}
        for r in records:
            dtype = r["document_type"]
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
        docs_by_type = [
            DocumentsByTypeItem(document_type=k, doc_count=v)
            for k, v in sorted(type_counts.items(), key=lambda x: (-x[1], x[0]))
        ]

        # Query 2: Number of questions by document
        questions_by_doc = [
            QuestionsByDocumentItem(
                document_id=r["document_id"],
                document_type=r["document_type"],
                query_count=r["query_count"],
            )
            for r in sorted(records, key=lambda x: (-x["query_count"], x["document_id"]))
        ]

        # Query 3: Number of processed documents
        total_processed = sum(1 for r in records if r["processing_time"] > 0)

        # Query 4: Risk-category distribution
        risk_counts: Dict[str, int] = {}
        for r in records:
            rcat = r.get("risk_category", "LOW")
            risk_counts[rcat] = risk_counts.get(rcat, 0) + 1
        risk_dist = [
            RiskCategoryDistributionItem(risk_category=k, category_count=v)
            for k, v in sorted(risk_counts.items(), key=lambda x: (-x[1], x[0]))
        ]

        # Query 5: Average processing time
        times = [r["processing_time"] for r in records if r["processing_time"] is not None]
        if times:
            avg_time = round(sum(times) / len(times), 2)
            min_time = round(min(times), 2)
            max_time = round(max(times), 2)
        else:
            avg_time, min_time, max_time = 0.0, 0.0, 0.0

        proc_metrics = ProcessingTimeMetrics(
            avg_processing_time=avg_time,
            min_processing_time=min_time,
            max_processing_time=max_time,
        )

        return AnalyticsSummaryResponse(
            engine="Snowflake Mock OLAP",
            total_processed_documents=total_processed,
            documents_by_type=docs_by_type,
            questions_by_document=questions_by_doc,
            risk_distribution=risk_dist,
            processing_time_metrics=proc_metrics,
        )


def get_analytics_service() -> BaseAnalyticsService:
    """Dependency provider returning SnowflakeAnalyticsService if credentials are provided,
    or MockAnalyticsService for local/offline execution."""
    mgr = get_snowflake_manager()
    if mgr.is_configured():
        return SnowflakeAnalyticsService(connection_manager=mgr)
    logger.info("Snowflake credentials not fully configured; using MockAnalyticsService.")
    return MockAnalyticsService()
