# Email: the SNS topic and its subscription already exist in alarms.tf
# (aws_sns_topic.alerts); these alarms reuse them

# Model metrics alarms
locals {
  model_alarms = {
    ScorePSI          = { threshold = 0.2,  description = "Drift in p_biased distribution" }
    LengthPSI         = { threshold = 0.2,  description = "Drift in text length" }
    MaxGroupGap       = { threshold = 0.15, description = "Large difference in positive rates between groups" }
    CounterfactualGap = { threshold = 0.1,  description = "Model reacts to identity in neutral sentences" }
  }
}

resource "aws_cloudwatch_metric_alarm" "model" {
  for_each = local.model_alarms

  alarm_name          = "${var.project}-${each.key}"
  alarm_description   = each.value.description
  namespace           = "BiasApi/Monitoring"
  metric_name         = each.key
  statistic           = "Maximum"
  period              = 3600
  evaluation_periods  = 1
  comparison_operator = "GreaterThanThreshold"
  threshold           = each.value.threshold
  treat_missing_data  = "notBreaching"   # metric only arrives once a day: hour without data = OK
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
}

# Monitor monitoring
resource "aws_cloudwatch_metric_alarm" "monitor_errors" {
  alarm_name          = "${var.project}-monitor-errors"
  alarm_description   = "The monitoring Lambda failed"
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  dimensions          = { FunctionName = aws_lambda_function.monitor.function_name }
  statistic           = "Sum"
  period              = 3600
  evaluation_periods  = 1
  comparison_operator = "GreaterThanThreshold"
  threshold           = 0
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# Dashboard
resource "aws_cloudwatch_dashboard" "monitoring" {
  dashboard_name = "${var.project}-monitoring"

  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric", x = 0, y = 0, width = 12, height = 6
        properties = {
          title  = "Drift (PSI)"
          region = var.region
          stat   = "Maximum"
          period = 3600
          metrics = [
            ["BiasApi/Monitoring", "ScorePSI"],
            ["BiasApi/Monitoring", "LengthPSI"],
          ]
          annotations = { horizontal = [{ label = "threshold", value = 0.2 }] }
        }
      },
      {
        type = "metric", x = 12, y = 0, width = 12, height = 6
        properties = {
          title  = "Fairness"
          region = var.region
          stat   = "Maximum"
          period = 3600
          metrics = [
            ["BiasApi/Monitoring", "MaxGroupGap"],
            ["BiasApi/Monitoring", "CounterfactualGap"],
          ]
        }
      },
      {
        type = "metric", x = 0, y = 6, width = 12, height = 6
        properties = {
          title  = "Group positive rates"
          region = var.region
          period = 3600
          metrics = [[{
            expression = "SEARCH('{BiasApi/Monitoring,Group} MetricName=\"GroupPositiveRate\"', 'Maximum', 3600)"
            id         = "groups"
          }]]
        }
      },
      {
        type = "metric", x = 12, y = 6, width = 12, height = 6
        properties = {
          title  = "Traffic"
          region = var.region
          stat   = "Maximum"
          period = 3600
          metrics = [
            ["BiasApi/Monitoring", "RequestCount"],
            ["BiasApi/Monitoring", "PositiveRate", { yAxis = "right" }],
          ]
        }
      },
    {
        type = "metric", x = 0, y = 12, width = 24, height = 6
        properties = {
          title  = "Bias categories (share of texts with each category)"
          region = var.region
          period = 3600
          metrics = [[{
            expression = "SEARCH('{BiasApi/Monitoring,Tag} MetricName=\"TagRate\"', 'Maximum', 3600)"
            id         = "tags"
          }]]
        }
      },

    ]
  })
}

output "dashboard_url" {
  value = "https://${var.region}.console.aws.amazon.com/cloudwatch/home?region=${var.region}#dashboards/dashboard/${aws_cloudwatch_dashboard.monitoring.dashboard_name}"
}