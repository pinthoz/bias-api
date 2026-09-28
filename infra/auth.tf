# API key auth for the HTTP API (which has no native API keys): a small
# Lambda authorizer compares the x-api-key header with a key kept in SSM.

resource "random_password" "api_key" {
  length  = 40
  special = false
}

# SecureString: encrypted at rest with the AWS-managed aws/ssm KMS key
resource "aws_ssm_parameter" "api_key" {
  name  = "/${var.project}/api-key"
  type  = "SecureString"
  value = random_password.api_key.result
}

data "archive_file" "authorizer" {
  type             = "zip"
  source_file      = "${path.module}/authorizer/authorizer.py"
  output_path      = "${path.module}/build/authorizer.zip"
  output_file_mode = "0644" # same zip hash on Windows/WSL and in CI
}

resource "aws_iam_role" "authorizer" {
  name               = "${var.project}-authorizer"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

resource "aws_iam_role_policy_attachment" "authorizer_logs" {
  role       = aws_iam_role.authorizer.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# The authorizer may read this one parameter and nothing else
resource "aws_iam_role_policy" "authorizer_ssm" {
  name = "read-api-key"
  role = aws_iam_role.authorizer.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "ssm:GetParameter"
      Resource = aws_ssm_parameter.api_key.arn
    }]
  })
}

resource "aws_cloudwatch_log_group" "authorizer" {
  name              = "/aws/lambda/${var.project}-authorizer"
  retention_in_days = 7
}

resource "aws_lambda_function" "authorizer" {
  function_name    = "${var.project}-authorizer"
  role             = aws_iam_role.authorizer.arn
  runtime          = "python3.12"
  handler          = "authorizer.handler"
  filename         = data.archive_file.authorizer.output_path
  source_code_hash = data.archive_file.authorizer.output_base64sha256
  architectures    = ["arm64"]
  memory_size      = 128
  timeout          = 5

  environment {
    variables = {
      API_KEY_PARAM = aws_ssm_parameter.api_key.name
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.authorizer_logs,
    aws_cloudwatch_log_group.authorizer,
  ]
}

resource "aws_apigatewayv2_authorizer" "api_key" {
  api_id                            = aws_apigatewayv2_api.http.id
  name                              = "${var.project}-api-key"
  authorizer_type                   = "REQUEST"
  authorizer_uri                    = aws_lambda_function.authorizer.invoke_arn
  authorizer_payload_format_version = "2.0"
  enable_simple_responses           = true
  # A request without the header is rejected (401) before the authorizer runs
  identity_sources = ["$request.header.x-api-key"]
  # Decisions are cached per key value, so most requests skip the authorizer
  authorizer_result_ttl_in_seconds = 300
}

resource "aws_lambda_permission" "authorizer" {
  statement_id  = "AllowAPIGatewayAuthorizer"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.authorizer.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/authorizers/${aws_apigatewayv2_authorizer.api_key.id}"
}
