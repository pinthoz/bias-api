output "ecr_url" {
  value = aws_ecr_repository.api.repository_url
}

output "api_url" {
  value = "${aws_apigatewayv2_api.http.api_endpoint}/predict"
}

output "api_batch_url" {
  value = "${aws_apigatewayv2_api.http.api_endpoint}/predict/batch"
}

# Read it with: terraform output -raw api_key
output "api_key" {
  value     = random_password.api_key.result
  sensitive = true
}
