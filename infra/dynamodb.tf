resource "aws_dynamodb_table" "metrics" {
  name         = "${var.project}-metrics"
  billing_mode = "PAY_PER_REQUEST"   # pay per on-demand
  hash_key     = "pk"
  range_key    = "sk"

  # Just need key attributes to be declared
  attribute {
    name = "pk"
    type = "S"
  }

  attribute {
    name = "sk"
    type = "S"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}

output "metrics_table" {
  value = aws_dynamodb_table.metrics.name
}

# The Lambda of inference can only update items from THIS table
data "aws_iam_policy_document" "lambda_metrics" {
  statement {
    actions   = ["dynamodb:UpdateItem"]
    resources = [aws_dynamodb_table.metrics.arn]
  }
}

resource "aws_iam_role_policy" "lambda_metrics" {
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.lambda_metrics.json
}