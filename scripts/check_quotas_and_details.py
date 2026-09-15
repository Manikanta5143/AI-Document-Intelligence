import boto3
import json

print("=== CHECKING BEDROCK SERVICE QUOTAS (ap-south-1) ===")
sq = boto3.client('service-quotas', region_name='ap-south-1')
try:
    paginator = sq.get_paginator('list_service_quotas')
    quotas = []
    for page in paginator.paginate(ServiceCode='bedrock'):
        quotas.extend(page.get('Quotas', []))
    print(f"Total Bedrock quotas in ap-south-1: {len(quotas)}")
    for q in quotas:
        name = q.get('QuotaName')
        val = q.get('Value')
        adj = q.get('Adjustable')
        code = q.get('QuotaCode')
        if any(term in name.lower() for term in ['gemma', 'invoke', 'model', 'request', 'token']):
            print(f"Quota: {name} ({code}) -> Value: {val}, Adjustable: {adj}")
except Exception as e:
    print(f"Service quotas Bedrock error: {type(e)} {e}")

print("\n=== CHECKING AWS MARKETPLACE AGREEMENTS & SUBSCRIPTIONS ===")
for client_name in ['marketplace-agreement', 'marketplace-entitlement', 'marketplace-catalog']:
    try:
        c = boto3.client(client_name, region_name='us-east-1')
        print(f"Client {client_name} initialized successfully.")
        if client_name == 'marketplace-agreement':
            # search agreements
            try:
                res = c.search_agreements(catalog='AWSMarketplace', maxResults=10)
                print("Agreements:", res.get('agreementViewSummaries', []))
            except Exception as e:
                print(f"search_agreements error: {type(e)} {e}")
    except Exception as e:
        print(f"Error for {client_name}: {type(e)} {e}")
