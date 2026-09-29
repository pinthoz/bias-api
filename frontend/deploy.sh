#!/usr/bin/env bash
# Build the Next.js site and publish it to S3 + CloudFront.
# Run from anywhere: ./frontend/deploy.sh  (needs `terraform apply` done in ../infra)
# CI runs the same steps on every push to main (.github/workflows/deploy.yml).
set -euo pipefail
cd "$(dirname "$0")"

tf() { terraform -chdir=../infra output -raw "$1"; }
BUCKET=$(tf site_bucket)
DIST=$(tf cloudfront_id)

# No API URL at build time: the site calls /predict on its own domain and
# CloudFront forwards it to API Gateway with the key
echo "→ Building"
npm ci
npm run build

# 1) Hashed assets first, cached for a year: their names change on every build
aws s3 sync out/_next "s3://$BUCKET/_next" --delete \
  --cache-control "public,max-age=31536000,immutable"

# 2) HTML and the rest, short cache so new versions show up quickly
aws s3 sync out/ "s3://$BUCKET" --delete --exclude "_next/*" \
  --cache-control "public,max-age=60"

# 3) Clear the CloudFront cache so the new HTML is served right away
aws cloudfront create-invalidation --distribution-id "$DIST" --paths "/*" >/dev/null

echo "✓ Deployed: $(tf site_url)"
