
import os

import litellm

from app.config import Settings
import instructor

from ragas.llms.base import (
    InstructorLLM,
    InstructorModelArgs
)
from ragas.embeddings.base import embedding_factory


def build_ragas_models():
    """
    Configure Ragas evaluator models using AWS Bedrock.

    Uses the same AWS profile, region, LLM and embedding
    model configured by the existing application.
    """

    settings = Settings()

    # Configure AWS region for LiteLLM
    os.environ["AWS_REGION_NAME"] = settings.aws_region
    os.environ["AWS_DEFAULT_REGION"] = settings.aws_region

    # Use the AWS profile configured in the application.
    # boto3/LiteLLM will resolve credentials through AWS.
    if settings.aws_profile:
        os.environ["AWS_PROFILE"] = settings.aws_profile

    # Configure evaluator LLM
    # Create asynchronous Instructor client using Markdown JSON mode
        async_client = instructor.from_litellm(
            litellm.acompletion,
            mode=instructor.Mode.MD_JSON
        )

        # Configure Ragas evaluator LLM
        evaluator_llm = InstructorLLM(
            client=async_client,
            model=f"bedrock/{settings.bedrock_model_id}",
            provider="litellm",
            model_args=InstructorModelArgs(
                temperature=0.0,
                max_tokens=1024
            )
        )

    # Configure evaluator embeddings
    evaluator_embeddings = embedding_factory(
        provider="litellm",
        model=f"bedrock/{settings.bedrock_embedding_model_id}",
    )

    return settings, evaluator_llm, evaluator_embeddings


if __name__ == "__main__":

    settings, evaluator_llm, evaluator_embeddings = (
        build_ragas_models()
    )

    print("Ragas evaluator configuration successful!")

    print(f"AWS Region: {settings.aws_region}")
    print(f"LLM Model: {settings.bedrock_model_id}")
    print(
        f"Embedding Model: "
        f"{settings.bedrock_embedding_model_id}"
    )

    print("Evaluator LLM initialized:", evaluator_llm is not None)
    print(
        "Evaluator embeddings initialized:",
        evaluator_embeddings is not None
    )
