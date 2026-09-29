terraform {
  required_version = ">= 1.10" # S3 native state locking (use_lockfile)
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.7"
    }
  }

  # Shared state, so CI and local runs see the same resources.
  # The bucket is created by bootstrap/ (backend blocks cannot use variables)
  backend "s3" {
    bucket       = "pinthoz-bias-api-infra"
    key          = "bias-api/terraform.tfstate"
    region       = "eu-west-1"
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      Project = var.project
    }
  }
}

resource "aws_ecr_repository" "api" {
  name         = var.project
  force_delete = true # This permits destroy even with images inside

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Keep only the 3 most recent images (storage = cost)
resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "manter 3 imagens"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 3
      }
      action = { type = "expire" }
    }]
  })
}

# IAM: what the Lambda can do
data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "${var.project}-lambda"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

resource "aws_iam_role_policy_attachment" "logs" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Created by Terraform to be deleted in destroy
resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${var.project}"
  retention_in_days = 7
}

# Lambda
resource "aws_lambda_function" "api" {
  function_name = var.project
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.api.repository_url}:${var.image_tag}"
  architectures = ["arm64"] # Graviton: ~20 % cheaper per GB-second than x86_64
  memory_size   = 3008
  timeout       = 90

  depends_on = [
    aws_iam_role_policy_attachment.logs,
    aws_cloudwatch_log_group.lambda,
  ]
}

# API Gateway (HTTP API)
resource "aws_apigatewayv2_api" "http" {
  name          = "${var.project}-http"
  protocol_type = "HTTP"

  # The deployed site calls the API through CloudFront on its own domain, so
  # it needs no CORS. This is only for `npm run dev`, which calls the
  # CloudFront URL from localhost (see frontend/.env.local.example)
  cors_configuration {
    allow_origins = ["http://localhost:3000"]
    allow_methods = ["POST", "OPTIONS"]
    allow_headers = ["content-type"]
  }
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.api.invoke_arn
  payload_format_version = "2.0"
}

# Both routes go to the same Lambda, which dispatches on the route key.
# Both require the x-api-key header (see auth.tf)
resource "aws_apigatewayv2_route" "predict" {
  for_each = toset(["POST /predict", "POST /predict/batch"])

  api_id             = aws_apigatewayv2_api.http.id
  route_key          = each.key
  target             = "integrations/${aws_apigatewayv2_integration.lambda.id}"
  authorization_type = "CUSTOM"
  authorizer_id      = aws_apigatewayv2_authorizer.api_key.id
}

# The route used to be a single resource: rename it in the state instead of
# destroying and re-creating it
moved {
  from = aws_apigatewayv2_route.predict
  to   = aws_apigatewayv2_route.predict["POST /predict"]
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.http.id
  name        = "$default"
  auto_deploy = true

  # Limiting requests protects your credits, even with an API key
  default_route_settings {
    throttling_burst_limit = 5
    throttling_rate_limit  = 2
  }
}

# Who can invoke the Lambda
resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*"
}
