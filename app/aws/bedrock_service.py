from abc import ABC, abstractmethod
import hashlib
import json
import math
from typing import List, Optional
import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import Settings, get_settings
from app.utils.logger import setup_logger

logger = setup_logger("embedding_service")


class BaseEmbeddingService(ABC):
    """Abstract base class defining the interface for generating text embeddings."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate an embedding vector for a single query or text snippet.

        Args:
            text: Text string to embed.

        Returns:
            List[float]: High-dimensional embedding vector.
        """
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of document chunks.

        Args:
            texts: List of document text strings to embed.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        pass


class BedrockEmbeddingService(BaseEmbeddingService):
    """Generates vector embeddings using Amazon Bedrock Runtime via boto3.

    Supports Amazon Titan Text Embeddings (v1 and v2) and other Bedrock embedding models.
    Respects AWS CLI profiles, environment credentials, and IAM roles without logging secrets.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model_id: Optional[str] = None,
    ):
        self.settings = settings or get_settings()
        self.model_id = model_id or self.settings.bedrock_embedding_model_id
        self._bedrock_client = None

    @property
    def client(self):
        """Lazy initialization of boto3 bedrock-runtime client."""
        if self._bedrock_client is None:
            self._bedrock_client = self._create_bedrock_client()
        return self._bedrock_client

    def _create_bedrock_client(self):
        """Build Bedrock Runtime client with configured session."""
        session_kwargs = {}
        if self.settings.aws_profile:
            session_kwargs["profile_name"] = self.settings.aws_profile
        if self.settings.aws_region:
            session_kwargs["region_name"] = self.settings.aws_region

        if self.settings.aws_access_key_id and self.settings.aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = self.settings.aws_access_key_id
            session_kwargs["aws_secret_access_key"] = self.settings.aws_secret_access_key
            if self.settings.aws_session_token:
                session_kwargs["aws_session_token"] = self.settings.aws_session_token

        logger.info(
            "Initializing Bedrock Runtime client (model='%s', region='%s', profile='%s')",
            self.model_id,
            self.settings.aws_region,
            self.settings.aws_profile or "default/env",
        )

        try:
            session = boto3.Session(**session_kwargs)
            return session.client("bedrock-runtime", region_name=self.settings.aws_region)
        except Exception as e:
            logger.error("Failed to initialize boto3 Bedrock Runtime client: %s", str(e))
            raise RuntimeError(f"Could not initialize Bedrock client: {str(e)}") from e

    def _build_payload(self, text: str) -> dict:
        """Format request payload according to the configured Bedrock embedding model."""
        payload = {"inputText": text}
        # Amazon Titan Text Embeddings v2 supports dimension and normalization parameters
        if "v2" in self.model_id:
            payload["dimensions"] = self.settings.embedding_dimension
            payload["normalize"] = True
        return payload

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single query using Amazon Bedrock."""
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text string.")

        clean_text = text.strip()
        payload = self._build_payload(clean_text)

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload),
            )
            response_body = json.loads(response["body"].read().decode("utf-8"))
            embedding = response_body.get("embedding")
            if not embedding:
                raise ValueError("Response from Bedrock did not contain an 'embedding' field.")
            return embedding
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            logger.error("Bedrock ClientError invoking embedding model [Code: %s]: %s", error_code, error_msg)
            raise RuntimeError(f"Bedrock embedding failed [{error_code}]: {error_msg}") from e
        except BotoCoreError as e:
            logger.error("Bedrock BotoCoreError during embedding: %s", str(e))
            raise RuntimeError(f"Bedrock communication error: {str(e)}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a collection of document chunks."""
        if not texts:
            return []

        logger.info("Generating Bedrock embeddings for %d document chunks...", len(texts))
        embeddings: List[List[float]] = []
        for idx, text in enumerate(texts):
            # Fallback for empty chunk text if any
            content = text.strip() if text else "empty"
            emb = self.embed_text(content)
            embeddings.append(emb)

        logger.info("Successfully generated %d embeddings with Bedrock.", len(embeddings))
        return embeddings


class MockEmbeddingService(BaseEmbeddingService):
    """Deterministic embedding service generator for automated testing without live AWS access."""

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension

    def _generate_vector(self, text: str) -> List[float]:
        """Generate a deterministic normalized float vector using SHA-256 seed."""
        hash_digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = []
        for i in range(self.dimension):
            byte_val = hash_digest[i % len(hash_digest)]
            vector.append((byte_val / 255.0) * 2.0 - 1.0)

        # Normalize to unit length
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_text(self, text: str) -> List[float]:
        return self._generate_vector(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._generate_vector(t) for t in texts]


def get_embedding_service() -> BaseEmbeddingService:
    """Dependency provider for the embedding service."""
    return BedrockEmbeddingService()
