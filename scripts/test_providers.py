import boto3
import json

bedrock = boto3.client('bedrock', region_name='ap-south-1')
rt = boto3.client('bedrock-runtime', region_name='ap-south-1')

models = bedrock.list_foundation_models()['modelSummaries']

# Sample different providers: Amazon, Anthropic, Meta, Mistral, Google, AI21, Cohere
tested_providers = {}
for m in models:
    p = m['providerName']
    mid = m['modelId']
    if p not in tested_providers and 'ON_DEMAND' in m.get('inferenceTypesSupported', []):
        tested_providers[p] = mid

print(f"Testing one model per provider in ap-south-1:")
for p, mid in tested_providers.items():
    try:
        # Try converse
        resp = rt.converse(
            modelId=mid,
            messages=[{'role': 'user', 'content': [{'text': 'hi'}]}]
        )
        print(f"[{p}] {mid}: SUCCESS")
    except Exception as e:
        print(f"[{p}] {mid}: {type(e).__name__} - {e}")
