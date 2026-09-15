from fastapi import APIRouter, Depends, status
from app.config import Settings, get_settings
from app.schemas.health import HealthResponse
from app.services.health_service import HealthService

router = APIRouter(tags=["Health"])


def get_health_service(settings: Settings = Depends(get_settings)) -> HealthService:
    """Dependency provider for HealthService."""
    return HealthService(settings=settings)


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Application Health Check",
    description="Returns the operational status, environment, and version of the service.",
)
def check_health(
    service: HealthService = Depends(get_health_service),
) -> HealthResponse:
    """Check health status of the application."""
    return service.get_health_status()
