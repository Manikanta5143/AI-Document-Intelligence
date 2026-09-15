import boto3

bedrock = boto3.client('bedrock', region_name='ap-south-1')
models = bedrock.list_foundation_models()['modelSummaries']

print(f"Checking agreement offers across {len(models)} models...")
for m in models:
    mid = m['modelId']
    try:
        offers = bedrock.list_foundation_model_agreement_offers(modelId=mid)
        print(f"FOUND OFFERS for {mid}: {offers}")
    except Exception as e:
        # Most might say 'Agreement not supported for this model'
        pass
print("Done checking.")
