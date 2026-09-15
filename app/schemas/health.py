from datetime import datetime
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema representing the health status response."""

    status: str = Field(default="healthy", description="Current service health status")
    app_name: str = Field(..., description="Name of the application")
    environment: str = Field(..., description="Current running environment (e.g., development, production)")
    version: str = Field(..., description="Application semantic version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp of the health check")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "app_name": "AI Insurance Document Intelligence Assistant",
                "environment": "development",
                "version": "1.0.0",
                "timestamp": "2026-09-14T09:45:00Z"
            }
        }
    }
