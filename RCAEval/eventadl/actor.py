import re


def get_actor(event):
    user_identity = event.get("userIdentity")
    if isinstance(user_identity, dict):
        if user_identity.get('arn'):
            return user_identity.get('arn')
        elif user_identity.get('ARN'):
            return user_identity.get('ARN')
        elif user_identity.get('principalId'):
            return user_identity.get('principalId')
        elif user_identity.get('invokedBy'):
            return user_identity.get('invokedBy')
        else:
            raise ValueError(f"No valid actor found in {user_identity}")
    elif isinstance(user_identity, str):
        # Try to extract invokedby
        invokedby_match = re.search(r'invokedby=([^,}]+)', user_identity)
        if invokedby_match and invokedby_match.group(1) != 'null':
            return f"{invokedby_match.group(1)}"
        
        # Try to extract ARN
        arn_match = re.search(r'arn=([^,}]+)', user_identity)
        if arn_match and arn_match.group(1) != 'null':
            return arn_match.group(1)
        
        # Try to extract type and principalid
        type_match = re.search(r'type=([^,}]+)', user_identity)
        principal_match = re.search(r'principalid=([^,}]+)', user_identity)
        if type_match and principal_match and principal_match.group(1) != 'null':
            return f"{principal_match.group(1)}"
        
        # Try to extract just type
        if type_match and type_match.group(1) != 'null':
            return f"{type_match.group(1)}"

        raise ValueError(f"No valid actor found in {user_identity}") 
    else:
        raise ValueError(f"Unexpected userIdentity type: {user_identity}")
