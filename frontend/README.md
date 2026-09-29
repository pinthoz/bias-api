# Bias Detector — frontend

Next.js (static export) + Tailwind CSS. Sends a sentence to the bias-api and
shows the flagged words, underlined by category (generalisation, unfair
language, stereotype), with the flagged spans and their scores.

Using it needs an API key: the visitor pastes it into the page, which keeps it
only in that browser (localStorage) and sends it as the `x-api-key` header.
Get it with `terraform -chdir=../infra output -raw api_key`.

The site is served from a private S3 bucket through CloudFront
([../infra/frontend.tf](../infra/frontend.tf)). It calls `/predict` on its own
domain: CloudFront forwards that path, with the key header, to API Gateway, so
no CORS is involved. A wrong key comes back as 404 rather than 403, because
CloudFront turns every 403 into its 404 page (meant for missing files).

## Local development

```bash
npm install
cp .env.local.example .env.local   # set the CloudFront URL + /predict
npm run dev                         # http://localhost:3000
```

The API allows CORS (with the `x-api-key` header) from `http://localhost:3000`,
so the dev server can call the deployed site's `/predict`.

## Deploy

Every push to `main` builds and publishes the site in CI. To do it by hand:

```bash
./deploy.sh
```

It builds the static site, uploads `out/` to the S3 bucket and invalidates CloudFront.
