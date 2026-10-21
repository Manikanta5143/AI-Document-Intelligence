from abc import ABC, abstractmethod
import json
from typing import Optional
import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import Settings, get_settings
from app.utils.logger import setup_logger

logger = setup_logger("bedrock_llm")


class BaseLLMService(ABC):
    """Abstract base class defining the text generation interface for LLM operations."""

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text using a language model."""
        pass

    def generate_text_with_metadata(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """Generate text and return dict with text, input_tokens, output_tokens, latency_ms."""
        text = self.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {
            "text": text,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0,
        }


class BedrockLLMService(BaseLLMService):
    """Integrates Amazon Nova Lite (and other Bedrock foundation models) via boto3 Bedrock Runtime.

    Employs the Bedrock Converse API with fallback, respects AWS credentials and profile
    configuration without credential leakage, and supports configurable inference parameters.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model_id: Optional[str] = None,
    ):
        self.settings = settings or get_settings()
        self.model_id = model_id or self.settings.bedrock_model_id
        self._bedrock_client = None

    @property
    def client(self):
        """Lazy initialization of boto3 Bedrock Runtime client."""
        if self._bedrock_client is None:
            self._bedrock_client = self._create_bedrock_client()
        return self._bedrock_client

    def _create_bedrock_client(self):
        """Construct boto3 bedrock-runtime client respecting credential resolution chain."""
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
            "Initializing Bedrock Runtime LLM client (model='%s', region='%s', profile='%s')",
            self.model_id,
            self.settings.aws_region,
            self.settings.aws_profile or "default/env",
        )

        try:
            session = boto3.Session(**session_kwargs)
            return session.client("bedrock-runtime", region_name=self.settings.aws_region)
        except Exception as e:
            logger.error("Failed to initialize boto3 Bedrock client: %s", str(e))
            raise RuntimeError(f"Could not initialize Bedrock client: {str(e)}") from e

    def generate_text_with_metadata(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """Execute text generation against Amazon Nova Lite returning text, usage, and latency metadata."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty for text generation.")

        tokens = max_tokens or self.settings.bedrock_max_tokens
        temp = self.settings.bedrock_temperature if temperature is None else temperature

        messages = [
            {
                "role": "user",
                "content": [{"text": prompt.strip()}],
            }
        ]

        converse_kwargs = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": tokens,
                "temperature": temp,
            },
        }

        if system_prompt and system_prompt.strip():
            converse_kwargs["system"] = [{"text": system_prompt.strip()}]

        logger.info(
            "Invoking Bedrock LLM '%s' (max_tokens=%d, temperature=%.2f)...",
            self.model_id,
            tokens,
            temp,
        )

        try:
            response = self.client.converse(**converse_kwargs)
            output_message = response.get("output", {}).get("message", {})
            content_list = output_message.get("content", [])
            if not content_list:
                raise ValueError("Received empty content array in Bedrock Converse response.")

            completion = content_list[0].get("text", "").strip()
            usage = response.get("usage", {})
            metrics = response.get("metrics", {})
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)
            latency_ms = metrics.get("latencyMs", 0)

            logger.info("Bedrock LLM response generated successfully (%d characters).", len(completion))
            return {
                "text": completion,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": usage.get("totalTokens", input_tokens + output_tokens),
                "latency_ms": latency_ms,
            }

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            logger.error("Bedrock ClientError during LLM generation [Code: %s]: %s", error_code, error_msg)
            raise RuntimeError(f"Bedrock generation failed [{error_code}]: {error_msg}") from e
        except BotoCoreError as e:
            logger.error("BotoCoreError during Bedrock communication: %s", str(e))
            raise RuntimeError(f"Bedrock communication error: {str(e)}") from e

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Execute text generation against Amazon Nova Lite via Bedrock Converse API."""
        result = self.generate_text_with_metadata(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return result["text"]


class MockLLMService(BaseLLMService):
    """Deterministic mock LLM service for testing summarization and structured extraction offline."""

    def __init__(self, model_id: str = "mock.nova-lite-v1:0"):
        self.model_id = model_id

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        prompt_lower = prompt.lower()
        system_lower = (system_prompt or "").lower()

        # If this is a structured extraction request
        if "extract" in prompt_lower or "json" in prompt_lower or "json" in system_lower:
            mock_extraction = {
                "policy_type": "Commercial Property Insurance",
                "coverage": {
                    "building_property": "$2,500,000",
                    "business_personal_property": "$750,000",
                    "business_income_extra_expense": "$300,000",
                },
                "exclusions": [
                    "Flood and surface water",
                    "Earth movement including earthquake and landslide",
                    "Normal wear and tear, rust, and gradual deterioration",
                ],
                "deductible": {
                    "standard_deductible": "$5,000 per occurrence",
                    "windstorm_hail_deductible": "2% of stated building value",
                },
                "important_conditions": [
                    "Prompt notice of claim required within 30 days of loss",
                    "Direct physical loss or damage to covered premises",
                ],
                "risk_indicators": [
                    "High building property limit with percentage-based windstorm deductible",
                    "Complete exclusion of flood risks requiring separate endorsement",
                ],
            }
            return json.dumps(mock_extraction, indent=2)

        # If this is a RAG Q&A question-answering request
        if "user question:" in prompt_lower or "insurance document assistant" in system_lower:
            target_text = prompt_lower
            if "user question:" in prompt_lower:
                target_text = prompt_lower.split("user question:", 1)[1]

            if "cyber" in target_text or "ransomware" in target_text or "unrelated" in target_text:
                return "The document does not provide enough information to answer this question."
            elif "water damage" in target_text or "flood" in target_text:
                return (
                    "Based on the provided policy documents (Page 2), water damage resulting from flood, "
                    "surface water, waves, or sewer backup is explicitly excluded under Exclusion Section 4. "
                    "However, sudden and accidental water discharge from internal plumbing fixtures is covered "
                    "subject to the $5,000 standard deductible."
                )
            elif "deductible" in target_text or "limit" in target_text:
                return (
                    "According to the policy terms (Page 1), the standard deductible is $5,000 per occurrence "
                    "for Building Property losses (limit: $2,500,000), with an endorsement specifying a 2% "
                    "deductible for windstorm or hail occurrences."
                )
            else:
                return (
                    "Based on the retrieved policy context, the policy provides coverage subject to "
                    "the stated policy limits, deductible terms, and exclusion criteria specified in the contract."
                )

        # Otherwise return an executive document summary
        return (
            "This document is a Commercial Property Insurance Policy issued to Acme Global Logistics LLC "
            "for the annual term from January 1, 2026 to January 1, 2027 with an annual premium of $14,500.\n\n"
            "Key Points:\n"
            "- Coverage A provides $2,500,000 for Building Property with a $5,000 standard deductible.\n"
            "- Coverage B provides $750,000 for Business Personal Property.\n"
            "- Standard exclusions apply for Flood, Earth Movement, and Normal Wear and Tear.\n"
            "- Claim notification must be submitted within 30 days of loss occurrence."
        )


def get_llm_service() -> BaseLLMService:
    """Dependency provider for Bedrock LLM service."""
    return BedrockLLMService()
