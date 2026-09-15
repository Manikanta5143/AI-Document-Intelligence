import boto3
import json

bedrock = boto3.client('bedrock', region_name='ap-south-1')
model_id = 'google.gemma-3-4b-it'

print("=== 1. get_foundation_model_availability ===")
try:
    res = bedrock.get_foundation_model_availability(modelId=model_id)
    print(json.dumps(res, indent=2, default=str))
except Exception as e:
    print(f"Error: {type(e).__name__} - {e}")

print("\n=== 2. list_foundation_model_agreement_offers ===")
try:
    res = bedrock.list_foundation_model_agreement_offers(modelId=model_id)
    print(json.dumps(res, indent=2, default=str))
except Exception as e:
    print(f"Error: {type(e).__name__} - {e}")

print("\n=== 3. get_use_case_for_model_access ===")
try:
    res = bedrock.get_use_case_for_model_access()
    print(json.dumps(res, indent=2, default=str))
except Exception as e:
    print(f"Error: {type(e).__name__} - {e}")

print("\n=== 4. list_marketplace_model_endpoints ===")
try:
    res = bedrock.list_marketplace_model_endpoints()
    print(json.dumps(res, indent=2, default=str))
except Exception as e:
    print(f"Error: {type(e).__name__} - {e}")
