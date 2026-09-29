# Code: zip generated at each plan
data "archive_file" "monitor" {
  type        = "zip"
  source_file = "${path.module}/../monitor/monitor.py"
  output_path      = "${path.module}/build/monitor.zip" # infra/build/ is not versioned
  output_file_mode = "0644"                              # same zip hash on WSL and in CI
}

# IAM
resource "aws_iam_role" "monitor" {
  name               = "${var.project}-monitor"
  assume_role_policy = data.aws_iam_policy_document.assume.json   # same trust policy as inference Lambda
}

resource "aws_iam_role_policy_attachment" "monitor_logs" {
  role       = aws_iam_role.monitor.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "monitor" {
  statement {
    sid       = "ReadMetrics"
    actions   = ["dynamodb:Query"]   # only Query: no Scan, no write
    resources = [aws_dynamodb_table.metrics.arn]
  }

  statement {
    sid       = "PublishMetrics"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]   # PutMetricData doesn't accept ARNs...
    condition {         # ...so we restrict by namespace
      test     = "StringEquals"
      variable = "cloudwatch:namespace"
      values   = ["BiasApi/Monitoring"]
    }
  }

  statement {
    sid       = "Canary"
    actions   = ["lambda:InvokeFunction"]
    resources = [aws_lambda_function.api.arn]
  }
}

resource "aws_iam_role_policy" "monitor" {
  role   = aws_iam_role.monitor.id
  policy = data.aws_iam_policy_document.monitor.json
}

# Lambda
resource "aws_cloudwatch_log_group" "monitor" {
  name              = "/aws/lambda/${var.project}-monitor"
  retention_in_days = 14
}

resource "aws_lambda_function" "monitor" {
  function_name    = "${var.project}-monitor"
  role             = aws_iam_role.monitor.arn
  runtime          = "python3.12"
  handler          = "monitor.handler"          # file.function
  filename         = data.archive_file.monitor.output_path
  source_code_hash = data.archive_file.monitor.output_base64sha256   # redeploy when code changes
  timeout          = 120
  memory_size      = 256

  environment {
    variables = {
      METRICS_TABLE    = aws_dynamodb_table.metrics.name
      API_FUNCTION     = aws_lambda_function.api.function_name
      BUCKET_MINUTES   = var.bucket_minutes
      WINDOW_MINUTES   = var.window_minutes
      BASELINE_MINUTES = var.baseline_minutes
      MIN_SAMPLES      = 20
    }
  }

  depends_on = [aws_iam_role_policy_attachment.monitor_logs, aws_cloudwatch_log_group.monitor]
}

# EventBridge Scheduler
data "aws_iam_policy_document" "scheduler_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "scheduler" {
  name               = "${var.project}-scheduler"
  assume_role_policy = data.aws_iam_policy_document.scheduler_assume.json
}

resource "aws_iam_role_policy" "scheduler" {
  role = aws_iam_role.scheduler.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "lambda:InvokeFunction"
      Resource = aws_lambda_function.monitor.arn
    }]
  })
}

resource "aws_scheduler_schedule" "monitor" {
  name                         = "${var.project}-monitor"
  schedule_expression          = var.monitor_schedule
  schedule_expression_timezone = "Europe/Lisbon"

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_lambda_function.monitor.arn
    role_arn = aws_iam_role.scheduler.arn
  }
}