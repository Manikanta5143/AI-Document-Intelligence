import os
import re
import uuid
from typing import BinaryIO, Optional
import boto3
from boto3.exceptions import Boto3Error
from botocore.exceptions import BotoCoreError, ClientError

from app.config import Settings, get_settings
from app.utils.logger import setup_logger

logger = setup_logger("s3_service")


class S3Service:
    """Service encapsulating AWS S3 operations using boto3.

    Handles authentication via AWS CLI profiles, environment variables,
    or IAM roles, while strictly guarding against credential leakage in logs.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._s3_client = None

    @property
    def client(self):
        """Lazy initialization of boto3 S3 client with configured session."""
        if self._s3_client is None:
            self._s3_client = self._create_s3_client()
        return self._s3_client

    def _create_s3_client(self):
        """Build boto3 client using the credential chain or profile configuration."""
        session_kwargs = {}
        if self.settings.aws_profile:
            session_kwargs["profile_name"] = self.settings.aws_profile
        if self.settings.aws_region:
            session_kwargs["region_name"] = self.settings.aws_region

        # Optional direct credentials override (never logged)
        if self.settings.aws_access_key_id and self.settings.aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = self.settings.aws_access_key_id
            session_kwargs["aws_secret_access_key"] = self.settings.aws_secret_access_key
            if self.settings.aws_session_token:
                session_kwargs["aws_session_token"] = self.settings.aws_session_token

        logger.info(
            "Initializing boto3 S3 client (region='%s', profile='%s')",
            self.settings.aws_region,
            self.settings.aws_profile or "default/env",
        )

        try:
            session = boto3.Session(**session_kwargs)
            return session.client("s3")
        except Exception as e:
            logger.error("Failed to initialize boto3 S3 client: %s", str(e))
            raise RuntimeError(f"Could not initialize S3 client: {str(e)}") from e

    def generate_s3_key(self, file_name: str) -> str:
        """Generate a safe, collision-resistant S3 object key.

        Args:
            file_name: Original file name.

        Returns:
            str: S3 key in the format: {prefix}/{uuid}_{sanitized_file_name}
        """
        # Sanitize filename to avoid weird S3 key characters
        base_name = os.path.basename(file_name)
        clean_name = re.sub(r"[^a-zA-Z0-9._-]", "_", base_name)
        unique_id = uuid.uuid4().hex[:12]
        prefix = self.settings.s3_prefix.strip("/")
        return f"{prefix}/{unique_id}_{clean_name}"

    def upload_fileobj(
        self,
        file_obj: BinaryIO,
        s3_key: str,
        content_type: str = "application/pdf",
    ) -> str:
        """Upload a file-like binary object to the configured S3 bucket.

        Args:
            file_obj: Readable binary file object.
            s3_key: Target S3 destination key.
            content_type: MIME type of the document.

        Raises:
            ValueError: If S3 bucket name is not configured.
            RuntimeError: If S3 upload fails.

        Returns:
            str: The S3 key where the object was stored.
        """
        bucket = self.settings.s3_bucket_name
        if not bucket:
            logger.error("S3 upload attempted but 'S3_BUCKET_NAME' is not configured.")
            raise ValueError("S3 bucket name is not configured. Please set S3_BUCKET_NAME in .env.")

        logger.info("Uploading object to S3 bucket='%s', key='%s'", bucket, s3_key)
        try:
            extra_args = {"ContentType": content_type}
            self.client.upload_fileobj(
                Fileobj=file_obj,
                Bucket=bucket,
                Key=s3_key,
                ExtraArgs=extra_args,
            )
            logger.info("Successfully uploaded object to s3://%s/%s", bucket, s3_key)
            return s3_key
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            logger.error("AWS S3 ClientError during upload [Code: %s]: %s", error_code, error_msg)
            raise RuntimeError(f"S3 upload failed: [{error_code}] {error_msg}") from e
        except (Boto3Error, BotoCoreError) as e:
            logger.error("Boto3/BotoCore error during upload: %s", str(e))
            raise RuntimeError(f"Boto error during S3 upload: {str(e)}") from e


def get_s3_service() -> S3Service:
    """Dependency provider for S3Service."""
    return S3Service()
