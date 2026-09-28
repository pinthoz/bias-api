"""API Gateway (HTTP API) Lambda authorizer: checks the x-api-key header.

The expected key is read once per cold start from SSM Parameter Store
(SecureString), so it never appears in the function's configuration.
Uses the "simple response" format: {"isAuthorized": bool}.
"""

import hmac
import os

import boto3

KEY = boto3.client("ssm").get_parameter(
    Name=os.environ["API_KEY_PARAM"], WithDecryption=True
)["Parameter"]["Value"]


def handler(event, context):
    # HTTP APIs lower-case header names
    sent = (event.get("headers") or {}).get("x-api-key", "")
    # Constant-time comparison, so response time does not leak the key
    return {"isAuthorized": hmac.compare_digest(sent.encode(), KEY.encode())}
