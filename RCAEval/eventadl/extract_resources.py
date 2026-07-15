#!/usr/bin/env python3
"""
CloudTrail Resource Extractor

This tool extracts impacted resources from CloudTrail events.
It identifies resources affected by different AWS API calls and returns their type and ARN.
"""

import json
import csv
import re
from typing import List, Dict, Any, Optional


def extract_resources(event: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract impacted resources from a CloudTrail event.
    
    Args:
        event: A CloudTrail event as a dictionary
        
    Returns:
        A list of dictionaries, each containing 'type' and 'arn' of an impacted resource
    """
    resources = []
    
    # Strategy 1: Direct Resources Array Extraction
    if "resources" in event and isinstance(event["resources"], list):
        for resource in event["resources"]:
            if "type" in resource and "ARN" in resource:
                resources.append({
                    "type": resource["type"],
                    "arn": resource["ARN"]
                })
        # If we found resources using this method, return them
        if resources:
            return resources
    
    event_source = event.get("eventSource", "")
    event_name = event.get("eventName", "")
    
    # Strategy 2-6: Service-specific extraction based on eventSource and eventName
    if event_source == "iam.amazonaws.com":
        resources.extend(extract_iam_resources(event, event_name))
    elif event_source == "s3.amazonaws.com":
        resources.extend(extract_s3_resources(event, event_name))
    elif event_source == "dynamodb.amazonaws.com":
        resources.extend(extract_dynamodb_resources(event, event_name))
    elif event_source == "sts.amazonaws.com":
        resources.extend(extract_sts_resources(event, event_name))
    elif event_source == "cloudformation.amazonaws.com":
        resources.extend(extract_cloudformation_resources(event, event_name))
    elif event_source == "events.amazonaws.com":
        resources.extend(extract_events_resources(event, event_name))
    elif event_source == "guardduty.amazonaws.com":
        resources.extend(extract_guardduty_resources(event, event_name))
    elif event_source == "kms.amazonaws.com":
        resources.extend(extract_kms_resources(event, event_name))
    elif event_source == "logs.amazonaws.com":
        resources.extend(extract_logs_resources(event, event_name))
    elif event_source == "lambda.amazonaws.com":
        resources.extend(extract_lambda_resources(event, event_name))
    elif event_source == "ec2.amazonaws.com":
        resources.extend(extract_ec2_resources(event, event_name))
    
    # Strategy 7: Fallback extraction - if no resources found yet, try generic extraction
    if not resources:
        resources.extend(extract_generic_resources(event))
    
    return resources


def extract_iam_resources(event: Dict[str, Any], event_name: str) -> List[Dict[str, str]]:
    """Extract resources from IAM events"""
    resources = []
    request_params = event.get("requestParameters", {}) or {}
    response_elements = event.get("responseElements", {}) or {}
    
    if event_name == "CreateRole":
        if response_elements and "role" in response_elements:
            role = response_elements["role"]
            if "arn" in role:
                resources.append({
                    "type": "AWS::IAM::Role",
                    "arn": role["arn"]
                })
        elif request_params and "roleName" in request_params:
            # Construct ARN if possible
            account_id = event.get("recipientAccountId", "")
            if account_id:
                arn = f"arn:aws:iam::{account_id}:role/{request_params['roleName']}"
                resources.append({
                    "type": "AWS::IAM::Role",
                    "arn": arn
                })
    
    elif event_name == "DeleteRole":
        if request_params and "roleName" in request_params:
            account_id = event.get("recipientAccountId", "")
            if account_id:
                arn = f"arn:aws:iam::{account_id}:role/{request_params['roleName']}"
                resources.append({
                    "type": "AWS::IAM::Role",
                    "arn": arn
                })
    
    elif event_name == "CreatePolicy":
        if response_elements and "policy" in response_elements:
            policy = response_elements["policy"]
            if "arn" in policy:
                resources.append({
                    "type": "AWS::IAM::Policy",
                    "arn": policy["arn"]
                })
    
    elif event_name == "DeletePolicy":
        if request_params and "policyArn" in request_params:
            resources.append({
                "type": "AWS::IAM::Policy",
                "arn": request_params["policyArn"]
            })
    
    elif event_name == "DeleteRolePolicy":
        if request_params:
            if "roleName" in request_params:
                account_id = event.get("recipientAccountId", "")
                role_name = request_params["roleName"]
                policy_name = request_params.get("policyName", "")
                
                if account_id and role_name:
                    # The role is impacted
                    resources.append({
                        "type": "AWS::IAM::Role",
                        "arn": f"arn:aws:iam::{account_id}:role/{role_name}"
                    })
                    
                    # The inline policy is impacted
                    if policy_name:
                        resources.append({
                            "type": "AWS::IAM::RolePolicy",
                            "arn": f"arn:aws:iam::{account_id}:role/{role_name}/policy/{policy_name}"
                        })
    
    elif event_name == "AttachRolePolicy":
        if request_params:
            if "roleName" in request_params:
                account_id = event.get("recipientAccountId", "")
                if account_id:
                    arn = f"arn:aws:iam::{account_id}:role/{request_params['roleName']}"
                    resources.append({
                        "type": "AWS::IAM::Role",
                        "arn": arn
                    })
            if "policyArn" in request_params:
                resources.append({
                    "type": "AWS::IAM::Policy",
                    "arn": request_params["policyArn"]
                })
    
    elif event_name == "DetachRolePolicy":
        if request_params:
            if "roleName" in request_params:
                account_id = event.get("recipientAccountId", "")
                if account_id:
                    arn = f"arn:aws:iam::{account_id}:role/{request_params['roleName']}"
                    resources.append({
                        "type": "AWS::IAM::Role",
                        "arn": arn
                    })
            if "policyArn" in request_params:
                resources.append({
                    "type": "AWS::IAM::Policy",
                    "arn": request_params["policyArn"]
                })
    elif event_name == "GetRole":
        if request_params and "roleName" in request_params:
            roleName = request_params["roleName"]
            account_id = event.get("recipientAccountId", "")
            arn = f"arn:aws:iam::{account_id}:role/{request_params['roleName']}"
            resources.append({
                "type": "AWS::IAM::Role",
                "arn": arn
            })
    return resources


def extract_s3_resources(event: Dict[str, Any], event_name: str) -> List[Dict[str, str]]:
    """Extract resources from S3 events"""
    resources = []
    request_params = event.get("requestParameters", {}) or {}
    
    if event_name in ["PutObject", "GetObject", "DeleteObject"]:
        bucket_name = request_params.get("bucketName", "")
        key = request_params.get("key", "")
        if bucket_name:
            # Add the bucket as a resource
            resources.append({
                "type": "AWS::S3::Bucket",
                "arn": f"arn:aws:s3:::{bucket_name}"
            })
            # Add the object as a resource if we have a key
            if key:
                resources.append({
                    "type": "AWS::S3::Object",
                    "arn": f"arn:aws:s3:::{bucket_name}/{key}"
                })
    
    elif event_name in ["CreateBucket", "DeleteBucket", "HeadBucket"]:
        bucket_name = request_params.get("bucketName", "")
        if bucket_name:
            resources.append({
                "type": "AWS::S3::Bucket",
                "arn": f"arn:aws:s3:::{bucket_name}"
            })
    
    return resources


def extract_dynamodb_resources(event: Dict[str, Any], event_name: str) -> List[Dict[str, str]]:
    """Extract resources from DynamoDB events"""
    resources = []
    request_params = event.get("requestParameters", {}) or {}
    region = event.get("awsRegion", "")
    account_id = event.get("recipientAccountId", "")
    
    if event_name in ["DescribeTable", "CreateTable", "UpdateTable", "DeleteTable", "GetItem", "PutItem", "DeleteItem", "Query", "Scan"]:
        table_name = request_params.get("tableName", "")
        if table_name and region and account_id:
            resources.append({
                "type": "AWS::DynamoDB::Table",
                "arn": f"arn:aws:dynamodb:{region}:{account_id}:table/{table_name}"
            })
    
    return resources


def extract_sts_resources(event: Dict[str, Any], event_name: str) -> List[Dict[str, str]]:
    """Extract resources from STS events"""
    resources = []
    request_params = event.get("requestParameters", {}) or {}
    
    if event_name == "AssumeRole":
        role_arn = request_params.get("roleArn", "")
        if role_arn:
            resources.append({
                "type": "AWS::IAM::Role",
                "arn": role_arn
            })
    
    return resources


def extract_cloudformation_resources(event: Dict[str, Any], event_name: str) -> List[Dict[str, str]]:
    """Extract resources from CloudFormation events"""
    resources = []
    request_params = event.get("requestParameters", {}) or {}
    region = event.get("awsRegion", "")
    account_id = event.get("recipientAccountId", "")
    
    if event_name in ["CreateStack", "UpdateStack", "DeleteStack"]:
        stack_name = request_params.get("stackName", "")
        if stack_name and region and account_id:
            resources.append({
                "type": "AWS::CloudFormation::Stack",
                "arn": f"arn:aws:cloudformation:{region}:{account_id}:stack/{stack_name}"
            })
    
    elif event_name == "CreateChangeSet":
        # Handle error case where requestParameters might be null
        if not request_params and "errorCode" in event:
            # Try to extract from error message if available
            error_message = event.get("errorMessage", "")
            if "do not exist in the template" in error_message:
                # This is likely a CloudFormation template validation error
                # We can't extract specific resources, but we can add a generic CloudFormation resource
                resources.append({
                    "type": "AWS::CloudFormation::ChangeSet",
                    "arn": f"arn:aws:cloudformation:{region}:{account_id}:changeSet/unknown"
                })
        else:
            stack_name = request_params.get("stackName", "")
            change_set_name = request_params.get("changeSetName", "")
            if stack_name and change_set_name and region and account_id:
                resources.append({
                    "type": "AWS::CloudFormation::Stack",
                    "arn": f"arn:aws:cloudformation:{region}:{account_id}:stack/{stack_name}"
                })
                resources.append({
                    "type": "AWS::CloudFormation::ChangeSet",
                    "arn": f"arn:aws:cloudformation:{region}:{account_id}:changeSet/{change_set_name}"
                })
    
    return resources


def extract_events_resources(event: Dict[str, Any], event_name: str) -> List[Dict[str, str]]:
    """Extract resources from CloudWatch Events"""
    resources = []
    request_params = event.get("requestParameters", {}) or {}
    region = event.get("awsRegion", "")
    account_id = event.get("recipientAccountId", "")
    
    if event_name in ["DescribeRule", "PutRule", "DeleteRule"]:
        rule_name = request_params.get("name", "")
        if rule_name and region and account_id:
            resources.append({
                "type": "AWS::Events::Rule",
                "arn": f"arn:aws:events:{region}:{account_id}:rule/{rule_name}"
            })
    
    elif event_name == "ListTargetsByRule":
        rule_name = request_params.get("rule", "")
        if rule_name and region and account_id:
            resources.append({
                "type": "AWS::Events::Rule",
                "arn": f"arn:aws:events:{region}:{account_id}:rule/{rule_name}"
            })
    
    return resources


def extract_guardduty_resources(event: Dict[str, Any], event_name: str) -> List[Dict[str, str]]:
    """Extract resources from GuardDuty events"""
    resources = []
    request_params = event.get("requestParameters", {}) or {}
    region = event.get("awsRegion", "")
    account_id = event.get("recipientAccountId", "")
    
    if "detectorId" in request_params:
        detector_id = request_params["detectorId"]
        if detector_id and region and account_id:
            resources.append({
                "type": "AWS::GuardDuty::Detector",
                "arn": f"arn:aws:guardduty:{region}:{account_id}:detector/{detector_id}"
            })
    
    return resources


def extract_kms_resources(event: Dict[str, Any], event_name: str) -> List[Dict[str, str]]:
    """Extract resources from KMS events"""
    resources = []
    request_params = event.get("requestParameters", {}) or {}
    region = event.get("awsRegion", "")
    account_id = event.get("recipientAccountId", "")
    
    # Extract key ID from various fields
    key_id = None
    if "keyId" in request_params:
        key_id = request_params["keyId"]
    elif "encryptionContext" in request_params and "aws:lambda:FunctionArn" in request_params["encryptionContext"]:
        # For Lambda-related KMS operations, extract the Lambda function as a resource
        lambda_arn = request_params["encryptionContext"]["aws:lambda:FunctionArn"]
        resources.append({
            "type": "AWS::Lambda::Function",
            "arn": lambda_arn
        })
    
    if key_id and region and account_id:
        # If key_id is already an ARN, use it directly
        if key_id.startswith("arn:aws:kms:"):
            resources.append({
                "type": "AWS::KMS::Key",
                "arn": key_id
            })
        else:
            # Otherwise, construct the ARN
            resources.append({
                "type": "AWS::KMS::Key",
                "arn": f"arn:aws:kms:{region}:{account_id}:key/{key_id}"
            })
    
    return resources


def extract_logs_resources(event: Dict[str, Any], event_name: str) -> List[Dict[str, str]]:
    """Extract resources from CloudWatch Logs events"""
    resources = []
    request_params = event.get("requestParameters", {}) or {}
    region = event.get("awsRegion", "")
    account_id = event.get("recipientAccountId", "")
    
    # Extract log group name
    log_group_name = None
    if "logGroupName" in request_params:
        log_group_name = request_params["logGroupName"]
    elif "logGroupIdentifier" in request_params:
        log_group_name = request_params["logGroupIdentifier"]
    
    if log_group_name and region and account_id:
        resources.append({
            "type": "AWS::Logs::LogGroup",
            "arn": f"arn:aws:logs:{region}:{account_id}:log-group:{log_group_name}"
        })
    
    # Extract log stream name if available
    if "logStreamName" in request_params and log_group_name:
        log_stream_name = request_params["logStreamName"]
        resources.append({
            "type": "AWS::Logs::LogStream",
            "arn": f"arn:aws:logs:{region}:{account_id}:log-group:{log_group_name}:log-stream:{log_stream_name}"
        })
    
    return resources


def extract_lambda_resources(event: Dict[str, Any], event_name: str) -> List[Dict[str, str]]:
    """Extract resources from Lambda events"""
    resources = []
    request_params = event.get("requestParameters", {}) or {}
    region = event.get("awsRegion", "")
    account_id = event.get("recipientAccountId", "")
    
    # Extract function name or ARN
    function_name = None
    if "functionName" in request_params:
        function_name = request_params["functionName"]
    
    if function_name:
        # If function_name is already an ARN, use it directly
        if function_name.startswith("arn:aws:lambda:"):
            resources.append({
                "type": "AWS::Lambda::Function",
                "arn": function_name
            })
        elif region and account_id:
            # Otherwise, construct the ARN
            resources.append({
                "type": "AWS::Lambda::Function",
                "arn": f"arn:aws:lambda:{region}:{account_id}:function:{function_name}"
            })
    
    return resources


def extract_ec2_resources(event: Dict[str, Any], event_name: str) -> List[Dict[str, str]]:
    """Extract resources from EC2 events"""
    resources = []
    request_params = event.get("requestParameters", {}) or {}
    response_elements = event.get("responseElements", {}) or {}
    region = event.get("awsRegion", "")
    account_id = event.get("recipientAccountId", "")
    
    # Extract instance IDs
    if event_name in ["RunInstances", "TerminateInstances", "StartInstances", "StopInstances"]:
        instance_ids = []
        
        # Check response elements for instance IDs
        if response_elements and "instancesSet" in response_elements:
            instances = response_elements["instancesSet"].get("items", [])
            for instance in instances:
                if "instanceId" in instance:
                    instance_ids.append(instance["instanceId"])
        
        # Check request parameters for instance IDs
        if "instancesSet" in request_params:
            instances = request_params["instancesSet"].get("items", [])
            for instance in instances:
                if "instanceId" in instance:
                    instance_ids.append(instance["instanceId"])
        
        # Add each instance as a resource
        for instance_id in instance_ids:
            if region and account_id:
                resources.append({
                    "type": "AWS::EC2::Instance",
                    "arn": f"arn:aws:ec2:{region}:{account_id}:instance/{instance_id}"
                })

        # For un-authorized events, we might not have instance IDs, let's take the unknown instance
        if not resources:
            if region and account_id:
                resources.append({
                    "type": "AWS::EC2::Instance",
                    "arn": f"arn:aws:ec2:{region}:{account_id}:instance/unknown"
                })

    
    # Extract security group IDs
    if event_name in ["CreateSecurityGroup", "DeleteSecurityGroup", "AuthorizeSecurityGroupIngress", "RevokeSecurityGroupIngress"]:
        group_id = None
        
        if "groupId" in request_params:
            group_id = request_params["groupId"]
        elif response_elements and "groupId" in response_elements:
            group_id = response_elements["groupId"]
        
        if group_id and region and account_id:
            resources.append({
                "type": "AWS::EC2::SecurityGroup",
                "arn": f"arn:aws:ec2:{region}:{account_id}:security-group/{group_id}"
            })
    
    # Extract VPC IDs
    if event_name in ["CreateVpc", "DeleteVpc"]:
        vpc_id = None
        
        if "vpcId" in request_params:
            vpc_id = request_params["vpcId"]
        elif response_elements and "vpc" in response_elements and "vpcId" in response_elements["vpc"]:
            vpc_id = response_elements["vpc"]["vpcId"]
        
        if vpc_id and region and account_id:
            resources.append({
                "type": "AWS::EC2::VPC",
                "arn": f"arn:aws:ec2:{region}:{account_id}:vpc/{vpc_id}"
            })
    
    return resources


def extract_generic_resources(event: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generic resource extraction for events without specific handlers.
    Attempts to find ARNs in the event and determine their resource type.
    """
    resources = []
    
    # Convert event to string for regex search
    event_str = json.dumps(event)
    
    # Find all ARNs in the event
    arn_pattern = r'arn:aws:[a-zA-Z0-9\-]+:[a-zA-Z0-9\-]*:[0-9]*:[a-zA-Z0-9\-]*:[a-zA-Z0-9\-\/\._]+'
    arns = re.findall(arn_pattern, event_str)
    
    # Process found ARNs
    for arn in arns:
        # Extract service and resource type from ARN
        parts = arn.split(':')
        if len(parts) >= 6:
            service = parts[2]
            resource_type = determine_resource_type(service, parts[5])
            
            resources.append({
                "type": resource_type,
                "arn": arn
            })
    
    return resources


def determine_resource_type(service: str, resource_path: str) -> str:
    """
    Determine the AWS resource type based on service and resource path.
    
    Args:
        service: AWS service name from ARN
        resource_path: Resource path from ARN
        
    Returns:
        AWS resource type in CloudFormation format (e.g., AWS::S3::Bucket)
    """
    # Common resource type mappings
    service_mappings = {
        "s3": "AWS::S3::Bucket",
        "ec2": "AWS::EC2::Instance",
        "iam": "AWS::IAM::Role",  # Default, will be refined below
        "lambda": "AWS::Lambda::Function",
        "dynamodb": "AWS::DynamoDB::Table",
        "sqs": "AWS::SQS::Queue",
        "sns": "AWS::SNS::Topic",
        "cloudformation": "AWS::CloudFormation::Stack",
        "guardduty": "AWS::GuardDuty::Detector",
        "events": "AWS::Events::Rule",
    }
    
    # Refine IAM resource types
    if service == "iam":
        if resource_path.startswith("role/"):
            return "AWS::IAM::Role"
        elif resource_path.startswith("policy/"):
            return "AWS::IAM::Policy"
        elif resource_path.startswith("user/"):
            return "AWS::IAM::User"
        elif resource_path.startswith("group/"):
            return "AWS::IAM::Group"
    
    # Return the mapped type or a generic type if not found
    return service_mappings.get(service, f"AWS::{service.capitalize()}::Resource")


def process_cloudtrail_event(event_json: str) -> List[Dict[str, str]]:
    """
    Process a CloudTrail event JSON string and extract resources.
    
    Args:
        event_json: CloudTrail event as a JSON string
        
    Returns:
        List of impacted resources
    """
    try:
        event = json.loads(event_json)
        return extract_resources(event)
    except json.JSONDecodeError:
        print(f"Error decoding JSON: {event_json[:100]}...")
        return []


def process_csv_file(file_path: str, output_path: str = "extracted_resources.csv", chunk_size: int = 1000, debug: bool = False):
    """
    Process a CSV file containing CloudTrail events and extract resources.
    
    Args:
        file_path: Path to the CSV file
        output_path: Path to write the output CSV
        chunk_size: Number of events to process at once
        debug: Whether to print debug information
    """
    with open(output_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['EventID', 'EventName', 'ResourceType', 'ResourceARN'])
        
        with open(file_path, 'r') as infile:
            reader = csv.reader(infile)
            
            # Process in chunks to handle large files
            chunk = []
            total_events = 0
            events_with_resources = 0
            
            for i, row in enumerate(reader):
                if i > 0 and i % chunk_size == 0:
                    events_processed, events_with_res = process_chunk(chunk, writer, debug)
                    total_events += events_processed
                    events_with_resources += events_with_res
                    chunk = []
                    print(f"Processed {i} events... ({events_with_resources}/{total_events} with resources)")
                
                chunk.append(row)
            
            # Process the last chunk
            if chunk:
                events_processed, events_with_res = process_chunk(chunk, writer, debug)
                total_events += events_processed
                events_with_resources += events_with_res
            
            print(f"Total events processed: {total_events}")
            print(f"Events with extracted resources: {events_with_resources} ({events_with_resources/total_events*100:.2f}%)")


def process_chunk(chunk: List[List[str]], writer: csv.writer, debug: bool = False) -> tuple:
    """
    Process a chunk of CloudTrail events
    
    Returns:
        Tuple of (total events processed, events with resources)
    """
    events_processed = 0
    events_with_resources = 0
    
    for row in chunk:
        if not row:
            continue
            
        try:
            events_processed += 1
            event_json = row[0]
            event = json.loads(event_json)
            event_id = event.get("eventID", "")
            event_name = event.get("eventName", "")
            event_source = event.get("eventSource", "")
            
            resources = extract_resources(event)
            
            if resources:
                events_with_resources += 1
                for resource in resources:
                    writer.writerow([
                        event_id,
                        event_name,
                        resource["type"],
                        resource["arn"]
                    ])
            elif debug:
                print(f"No resources extracted for event: {event_name} from {event_source}")
                
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Error processing event: {e}")
    
    return events_processed, events_with_resources


def main():
    """Main function to run the resource extractor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract resources from CloudTrail events')
    parser.add_argument('input_file', help='Path to the CSV file containing CloudTrail events')
    parser.add_argument('--output', '-o', default='extracted_resources.csv',
                        help='Path to write the output CSV (default: extracted_resources.csv)')
    parser.add_argument('--chunk-size', '-c', type=int, default=1000,
                        help='Number of events to process at once (default: 1000)')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Print debug information')
    
    args = parser.parse_args()
    
    print(f"Processing CloudTrail events from {args.input_file}...")
    process_csv_file(args.input_file, args.output, args.chunk_size, args.debug)
    print(f"Extracted resources written to {args.output}")


if __name__ == "__main__":
    #main()
    print(extract_resources(
{
  "eventVersion": "1.10",
  "userIdentity": {
    "type": "AssumedRole",
    "principalId": "AROAWXMV26S4CAWFEE3RL:access-analyzer",
    "arn": "arn:aws:sts::462560097464:assumed-role/AWSServiceRoleForAccessAnalyzer/access-analyzer",
    "accountId": "462560097464",
    "accessKeyId": "ASIAWXMV26S4DT77CKN2",
    "sessionContext": {
      "sessionIssuer": {
        "type": "Role",
        "principalId": "AROAWXMV26S4CAWFEE3RL",
        "arn": "arn:aws:iam::462560097464:role/aws-service-role/access-analyzer.amazonaws.com/AWSServiceRoleForAccessAnalyzer",
        "accountId": "462560097464",
        "userName": "AWSServiceRoleForAccessAnalyzer"
      },
      "attributes": {
        "creationDate": "2024-10-22T19:14:19Z",
        "mfaAuthenticated": "false"
      }
    },
    "invokedBy": "access-analyzer.amazonaws.com"
  },
  "eventTime": "2024-10-22T19:14:20Z",
  "eventSource": "iam.amazonaws.com",
  "eventName": "GetRole",
  "awsRegion": "us-east-1",
  "sourceIPAddress": "access-analyzer.amazonaws.com",
  "userAgent": "access-analyzer.amazonaws.com",
  "errorCode": "NoSuchEntityException",
  "errorMessage": "The role with name awssigninwebsite-prod-eu-central-2 cannot be found.",
  "requestParameters": {
    "roleName": "awssigninwebsite-prod-eu-central-2"
  },
  "responseElements": None,
  "requestID": "2269606d-c2b8-4192-9e03-61d73e48968c",
  "eventID": "6fca474f-c4af-4aef-9a06-00bac858fd37",
  "readOnly": True,
  "eventType": "AwsApiCall",
  "managementEvent": True,
  "recipientAccountId": "462560097464",
  "eventCategory": "Management"
}

    ))
