# Bias Detector — frontend

Next.js (static export) + Tailwind CSS. Sends a sentence to the bias-api and
shows the flagged words, underlined by category (generalisation, unfair
language, stereotype), with the flagged spans and their scores.

The site is served from a private S3 bucket through CloudFront
([../infra/frontend.tf](../infra/frontend.tf)). It calls `/predict` on its own
domain: CloudFront forwards that path to API Gateway and adds the `x-api-key`
header itself, so the key never reaches the browser and no CORS is involved.

## Local development

```bash
npm install
cp .env.local.example .env.local   # set the CloudFront URL + /predict
npm run dev                         # http://localhost:3000
```

The API allows CORS from `http://localhost:3000`, so the dev server can call
the deployed site's `/predict`.

## Deploy

Every push to `main` builds and publishes the site in CI. To do it by hand:

```bash
./deploy.sh
```

It builds the static site, uploads `out/` to the S3 bucket and invalidates CloudFront.
