from datetime import datetime, timezone
from app.config import Settings
from app.schemas.health import HealthResponse
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class HealthService:
    """Service layer handling system health inspection logic."""

    def __init__(self, settings: Settings):
        self._settings = settings

    def get_health_status(self) -> HealthResponse:
        """Inspect and return the application health status.

        Returns:
            HealthResponse: Object containing current health and metadata.
        """
        logger.debug("Health status check requested.")
        return HealthResponse(
            status="healthy",
            app_name=self._settings.app_name,
            environment=self._settings.environment,
            version=self._settings.app_version,
            timestamp=datetime.now(timezone.utc),
        )
