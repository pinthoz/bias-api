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
