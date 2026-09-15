import boto3

bedrock = boto3.client('bedrock', region_name='ap-south-1')
models = bedrock.list_foundation_models()['modelSummaries']

print(f"Checking authorizationStatus for all {len(models)} models in ap-south-1...")
authorized = []
not_authorized = []
errors = []

for m in models:
    mid = m['modelId']
    try:
        avail = bedrock.get_foundation_model_availability(modelId=mid)
        status = avail.get('authorizationStatus')
        if status == 'AUTHORIZED':
            authorized.append(mid)
            print(f"  [AUTHORIZED] {mid}")
        else:
            not_authorized.append((mid, status))
    except Exception as e:
        errors.append((mid, str(e)))

print(f"\nSummary:")
print(f"  AUTHORIZED: {len(authorized)}")
print(f"  NOT_AUTHORIZED: {len(not_authorized)}")
print(f"  Errors: {len(errors)}")
if authorized:
    print("Authorized models:", authorized)
