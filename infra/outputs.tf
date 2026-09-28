output "ecr_url" {
  value = aws_ecr_repository.api.repository_url
}

output "api_url" {
  value = "${aws_apigatewayv2_api.http.api_endpoint}/predict"
}