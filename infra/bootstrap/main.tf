# One-time setup, applied by hand with admin credentials (local state).
# Creates what the main configuration and CI need before they can run:
#   - the S3 bucket holding the Terraform state and the int8 model file
#   - GitHub's OIDC identity provider and the role GitHub Actions assumes
# Kept separate so CI never manages the role it runs as.

terraform {
  required_version = ">= 1.10" # S3 native state locking (use_lockfile)
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
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

variable "region" {
  default = "eu-west-1"
}

variable "project" {
  default = "bias-api"
}

variable "bucket" {
  # Must match the backend "s3" block in ../main.tf
  default = "pinthoz-bias-api-infra"
}

# GitHub puts the immutable numeric IDs of the owner and the repository in the
# OIDC subject ("repo:owner@<id>/repo@<id>:..."), so a repository re-created
# under the same name, or a renamed account's old name, cannot match
variable "github_repo" {
  default = "pinthoz@69254873/bias-api@1393644182"
}

data "aws_caller_identity" "current" {}

locals {
  account = data.aws_caller_identity.current.account_id
}

# ---------- State + artifacts bucket ----------
resource "aws_s3_bucket" "infra" {
  bucket = var.bucket

  lifecycle {
    prevent_destroy = true # losing the state means losing track of everything
  }
}

resource "aws_s3_bucket_versioning" "infra" {
  bucket = aws_s3_bucket.infra.id
  versioning_configuration {
    status = "Enabled" # every state write can be rolled back
  }
}

resource "aws_s3_bucket_public_access_block" "infra" {
  bucket                  = aws_s3_bucket.infra.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "infra" {
  bucket = aws_s3_bucket.infra.id
  rule {
    id     = "expire-old-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# ---------- GitHub Actions OIDC ----------
resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
}

# Only workflows running on the main branch of this repo may assume the role
data "aws_iam_policy_document" "github_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:ref:refs/heads/main"]
    }
  }
}

resource "aws_iam_role" "github_deploy" {
  # Deliberately not "bias-api-*": the deploy policy below manages roles with
  # that prefix, and must not be able to edit its own role
  name               = "github-${var.project}-deploy"
  assume_role_policy = data.aws_iam_policy_document.github_trust.json
}

# What `docker push` + `terraform apply` of ../ need, scoped to this project
data "aws_iam_policy_document" "deploy" {
  statement {
    sid       = "TerraformState"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.infra.arn]
  }
  statement {
    sid       = "TerraformStateObjects"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.infra.arn}/*"]
  }
  statement {
    sid       = "EcrLogin"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid       = "EcrRepo"
    actions   = ["ecr:*"]
    resources = ["arn:aws:ecr:${var.region}:${local.account}:repository/${var.project}"]
  }
  statement {
    sid       = "Lambda"
    actions   = ["lambda:*"]
    resources = ["arn:aws:lambda:${var.region}:${local.account}:function:${var.project}*"]
  }
  statement {
    sid       = "ApiGateway"
    actions   = ["apigateway:*"]
    resources = ["arn:aws:apigateway:${var.region}::/*"]
  }
  statement {
    sid = "IamProjectRoles"
    actions = [
      "iam:GetRole", "iam:CreateRole", "iam:DeleteRole", "iam:UpdateRole",
      "iam:UpdateAssumeRolePolicy", "iam:TagRole", "iam:UntagRole", "iam:ListRoleTags",
      "iam:ListRolePolicies", "iam:GetRolePolicy", "iam:PutRolePolicy", "iam:DeleteRolePolicy",
      "iam:ListAttachedRolePolicies", "iam:AttachRolePolicy", "iam:DetachRolePolicy",
      "iam:ListInstanceProfilesForRole", "iam:PassRole",
    ]
    resources = ["arn:aws:iam::${local.account}:role/${var.project}-*"]
  }
  statement {
    sid = "Logs"
    actions = [
      "logs:CreateLogGroup", "logs:DeleteLogGroup", "logs:PutRetentionPolicy",
      "logs:DeleteRetentionPolicy", "logs:TagResource", "logs:UntagResource",
      "logs:ListTagsForResource", "logs:TagLogGroup", "logs:ListTagsLogGroup",
    ]
    resources = ["arn:aws:logs:${var.region}:${local.account}:log-group:/aws/lambda/${var.project}*"]
  }
  statement {
    sid = "Describe"
    actions = [
      "logs:DescribeLogGroups", "ssm:DescribeParameters", "cloudwatch:DescribeAlarms",
      # Looking up the AWS-managed cache / origin-request policies by name
      "cloudfront:ListCachePolicies", "cloudfront:GetCachePolicy",
      "cloudfront:ListOriginRequestPolicies", "cloudfront:GetOriginRequestPolicy",
    ]
    resources = ["*"]
  }
  statement {
    sid = "Ssm"
    actions = [
      "ssm:GetParameter", "ssm:GetParameters", "ssm:PutParameter", "ssm:DeleteParameter",
      "ssm:AddTagsToResource", "ssm:RemoveTagsFromResource", "ssm:ListTagsForResource",
    ]
    resources = ["arn:aws:ssm:${var.region}:${local.account}:parameter/${var.project}/*"]
  }
  statement {
    sid       = "Sns"
    actions   = ["sns:*"]
    resources = ["arn:aws:sns:${var.region}:${local.account}:${var.project}-*"]
  }
  statement {
    # Prediction counters for drift / fairness monitoring (../dynamodb.tf)
    sid       = "DynamoDB"
    actions   = ["dynamodb:*"]
    resources = ["arn:aws:dynamodb:${var.region}:${local.account}:table/${var.project}-*"]
  }
  statement {
    # Daily monitoring run (../monitor.tf)
    sid       = "Scheduler"
    actions   = ["scheduler:*"]
    resources = ["arn:aws:scheduler:${var.region}:${local.account}:schedule/default/${var.project}-*"]
  }
  statement {
    sid = "Dashboards"
    actions = [
      "cloudwatch:GetDashboard", "cloudwatch:PutDashboard", "cloudwatch:DeleteDashboards",
    ]
    resources = ["arn:aws:cloudwatch::${local.account}:dashboard/${var.project}-*"]
  }
  statement {
    sid       = "SiteBucket"
    actions   = ["s3:*"]
    resources = ["arn:aws:s3:::${var.project}-site-*", "arn:aws:s3:::${var.project}-site-*/*"]
  }
  statement {
    # CloudFront ARNs carry random ids, not names, so this cannot be narrowed
    # to this project's distribution
    sid       = "CloudFront"
    actions   = ["cloudfront:*"]
    resources = ["arn:aws:cloudfront::${local.account}:*"]
  }
  statement {
    sid = "Alarms"
    actions = [
      "cloudwatch:PutMetricAlarm", "cloudwatch:DeleteAlarms", "cloudwatch:TagResource",
      "cloudwatch:UntagResource", "cloudwatch:ListTagsForResource",
    ]
    resources = ["arn:aws:cloudwatch:${var.region}:${local.account}:alarm:${var.project}-*"]
  }
}

resource "aws_iam_role_policy" "deploy" {
  name   = "deploy"
  role   = aws_iam_role.github_deploy.id
  policy = data.aws_iam_policy_document.deploy.json
}

output "deploy_role_arn" {
  value = aws_iam_role.github_deploy.arn
}

output "bucket" {
  value = aws_s3_bucket.infra.bucket
}
