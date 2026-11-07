
import os

import litellm

from app.config import Settings


def main():
    settings = Settings()

    # Configure AWS region and profile
    os.environ["AWS_REGION_NAME"] = settings.aws_region
    os.environ["AWS_DEFAULT_REGION"] = settings.aws_region

    if settings.aws_profile:
        os.environ["AWS_PROFILE"] = settings.aws_profile

    aws_params = {
        "aws_region_name": settings.aws_region
    }

    if settings.aws_profile:
        aws_params["aws_profile_name"] = settings.aws_profile

    # ----------------------------------
    # 1. Test evaluator LLM
    # ----------------------------------

    print("\nTesting Bedrock evaluator LLM...")

    llm_response = litellm.completion(
        model=f"bedrock/{settings.bedrock_model_id}",
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: RAGAS_LLM_OK"
            }
        ],
        temperature=0.0,
        max_tokens=40,
        **aws_params
    )

    answer = llm_response.choices[0].message.content

    if not answer:
        raise RuntimeError("LLM returned an empty response")

    print("[PASS] Bedrock LLM invocation successful")
    print("Response:", answer)

    # ----------------------------------
    # 2. Test evaluator embeddings
    # ----------------------------------

    print("\nTesting Bedrock evaluator embeddings...")

    embedding_response = litellm.embedding(
        model=f"bedrock/{settings.bedrock_embedding_model_id}",
        input=["Insurance policy waiting period."],
        **aws_params
    )

    item = embedding_response.data[0]

    vector = (
        item["embedding"]
        if isinstance(item, dict)
        else item.embedding
    )

    if not vector:
        raise RuntimeError("Embedding generation returned empty vector")

    print("[PASS] Bedrock embedding generation successful")
    print("Embedding dimensions:", len(vector))

    print("\nAll Bedrock smoke tests passed!")


if __name__ == "__main__":
    main()
