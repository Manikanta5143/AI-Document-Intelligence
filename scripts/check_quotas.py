import boto3
from botocore.exceptions import ClientError

session = boto3.Session(region_name="ap-south-1")
sq = session.client("service-quotas", region_name="ap-south-1")

print("=== Checking Bedrock Service Quotas in ap-south-1 ===")
try:
    paginator = sq.get_paginator("list_service_quotas")
    found_any = False
    for page in paginator.paginate(ServiceCode="bedrock"):
        for q in page.get("Quotas", []):
            found_any = True
            print(f"{q.get('QuotaName')}: {q.get('Value')}")
    if not found_any:
        print("No specific quotas returned for bedrock.")
except ClientError as e:
    print(f"Service Quotas ClientError: {e}")
except Exception as e:
    print(f"Service Quotas Error: {e}")
