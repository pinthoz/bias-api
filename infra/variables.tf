variable "region" {
  default = "eu-west-1"
}

variable "project" {
  default = "bias-api"
}

# No default on purpose: CI passes the commit SHA it just pushed. A default
# here would roll the Lambda back to an old image on a local `terraform apply`
variable "image_tag" {
  type = string
}

variable "alert_email" {
  description = "Where the CloudWatch error-rate alarm sends email"
  type        = string
}

variable "monitor_schedule" {
  type    = string
  default = "cron(0 10 * * ? *)"  
}

variable "bucket_minutes" {
  type    = number
  default = 60 # 1 hour buckets 
}

variable "window_minutes" {
  type    = number
  default = 1440 # 24 hours actual window 
}

variable "baseline_minutes" {
  type    = number
  default = 10080  # 7 days before current window
}