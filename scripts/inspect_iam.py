import boto3
import json

iam = boto3.client('iam')
user_name = 'insurance-ai-dev'

user = iam.get_user(UserName=user_name)['User']
print('User:', user['UserName'])
print('ARN:', user['Arn'])
print('Permissions Boundary:', user.get('PermissionsBoundary'))

groups = iam.list_groups_for_user(UserName=user_name)['Groups']
print('Groups:', [g['GroupName'] for g in groups])

attached = iam.list_attached_user_policies(UserName=user_name)['AttachedPolicies']
print('Attached Policies:', json.dumps(attached, indent=2))

inline = iam.list_user_policies(UserName=user_name)['PolicyNames']
print('Inline Policies:', inline)
for p in inline:
    pol = iam.get_user_policy(UserName=user_name, PolicyName=p)
    print(f'Inline Policy [{p}]:', json.dumps(pol['PolicyDocument'], indent=2))

for g in groups:
    gname = g['GroupName']
    gp_attached = iam.list_attached_group_policies(GroupName=gname)['AttachedPolicies']
    print(f'Group {gname} attached policies:', json.dumps(gp_attached, indent=2))
    gp_inline = iam.list_group_policies(GroupName=gname)['PolicyNames']
    print(f'Group {gname} inline policies:', gp_inline)
    for gp in gp_inline:
        gpol = iam.get_group_policy(GroupName=gname, PolicyName=gp)
        print(f'Group [{gname}] inline policy [{gp}]:', json.dumps(gpol['PolicyDocument'], indent=2))
