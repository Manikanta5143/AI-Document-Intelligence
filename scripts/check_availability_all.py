import boto3
import json

bedrock = boto3.client('bedrock', region_name='ap-south-1')

models_to_check = [
    'google.gemma-3-4b-it',
    'google.gemma-3-12b-it',
    'amazon.titan-embed-text-v2:0',
    'amazon.titan-embed-text-v1',
    'amazon.titan-text-lite-v1',
    'meta.llama3-8b-instruct-v1:0',
    'anthropic.claude-3-haiku-20240307-v1:0'
]

for m in models_to_check:
    try:
        res = bedrock.get_foundation_model_availability(modelId=m)
        print(f"{m}:")
        print(f"  authStatus: {res.get('authorizationStatus')}")
        print(f"  agreement: {res.get('agreementAvailability')}")
        print(f"  entitlement: {res.get('entitlementAvailability')}")
        print(f"  region: {res.get('regionAvailability')}")
    except Exception as e:
        print(f"{m}: Error {type(e).__name__} - {e}")
