import boto3
import json

bedrock = boto3.client('bedrock', region_name='ap-south-1')
try:
    res = bedrock.list_inference_profiles()
    profiles = res.get('inferenceProfileSummaries', [])
    print(f"Inference profiles in ap-south-1: {len(profiles)}")
    for p in profiles:
        print(f"ID: {p.get('inferenceProfileId')} - Name: {p.get('inferenceProfileName')} - Type: {p.get('type')}")
except Exception as e:
    print("Error:", e)
