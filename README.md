# bias-api

A serverless API for **token-level social-bias detection**. It serves
[GUS-Net (BERT)](https://huggingface.co/pinthoz/gus-net-bert), a fine-tuned
`bert-base-uncased` token classifier developed in the
[Attention Atlas](https://github.com/pinthoz/attention-atlas) project. Given a
sentence, the API returns the words that carry bias and the type of bias:
**generalisation (GEN)**, **unfair language (UNFAIR)** or **stereotype (STEREO)**.

The model runs as an int8-quantized ONNX graph inside an arm64 (Graviton) AWS
Lambda container image, behind an API Gateway HTTP API protected by an API key.
All the infrastructure is defined in Terraform and deployed by GitHub Actions.

```
client ──POST /predict──────► API Gateway (HTTP API, throttled) ──► Lambda (container image, arm64)
         POST /predict/batch        │                                ├─ tokenizers (tokenizer.json)
         x-api-key: …               ▼                                └─ onnxruntime (model.int8.onnx)
                              Lambda authorizer ── SSM Parameter Store (API key)
```

## API

Every request needs the `x-api-key` header. Requests without it get `401`, and
requests with a wrong key get `403`.

`POST /predict` takes a JSON body with a `text` field:

```bash
curl -s -X POST "$API_URL" -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"text": "Women are bad at math."}'
```

Response (abridged):

```json
{
  "biased": true,
  "categories": ["GEN", "STEREO", "UNFAIR"],
  "biased_tokens": ["Women", "are", "bad", "at", "math"],
  "spans": [
    {"category": "GEN",    "start": 0,  "end": 5,  "score": 0.6151, "text": "Women"},
    {"category": "UNFAIR", "start": 0,  "end": 5,  "score": 0.3961, "text": "Women"},
    {"category": "STEREO", "start": 0,  "end": 21, "score": 0.6276, "text": "Women are bad at math"},
    {"category": "UNFAIR", "start": 10, "end": 21, "score": 0.474,  "text": "bad at math"}
  ],
  "tokens": [
    {"token": "Women", "start": 0, "end": 5,
     "labels": ["B-GEN", "B-UNFAIR", "B-STEREO"], "categories": ["GEN", "STEREO", "UNFAIR"],
     "scores": {"B-GEN": 0.6151, "B-UNFAIR": 0.3961, "B-STEREO": 0.6276}},
    {"token": "are", "start": 6, "end": 9,
     "labels": ["I-STEREO"], "categories": ["STEREO"], "scores": {"I-STEREO": 0.457}},
    ...
    {"token": ".", "start": 21, "end": 22, "labels": [], "categories": [], "scores": {}}
  ]
}
```

| Field | Meaning |
|---|---|
| `biased` | `true` if any word was flagged. |
| `categories` | Bias categories found in the sentence. |
| `biased_tokens` | The flagged words, as they appear in the input. |
| `spans` | Consecutive words flagged with the same category, merged into one span. `start`/`end` are character offsets into `text`; `score` is the highest probability in the span. |
| `tokens` | Every word of the input in order, with the BIO labels that fired, their categories and probabilities. Neutral words have empty lists, so a client can render the whole sentence with highlights. |

A missing, empty or non-string `text` returns `400`.

### Batch

`POST /predict/batch` takes up to 32 texts and returns one result per text, in
the same order and with the same shape as `/predict`:

```bash
curl -s -X POST "$API_URL/batch" -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"texts": ["Women are bad at math.", "The meeting is at noon."]}'
# {"results": [{"biased": true, ...}, {"biased": false, ...}]}
```

The texts are run through the model one at a time, not as one padded batch.
With dynamic int8 quantization, the activation scale is computed over the whole
input tensor, so padding sentences together changes each one's probabilities
(by up to 0.06 in a quick test, against exactly 0 for fp32). Running them
separately keeps every batch result identical to the single-text endpoint.

### How a label fires

GUS-Net is **multi-label**: every token gets an independent sigmoid probability
for each of the 7 BIO labels (`O`, `B-/I-STEREO`, `B-/I-GEN`, `B-/I-UNFAIR`).
A label fires when its probability reaches that label's own threshold, which was
optimised for F1 on the validation split (see
[Quantization](#quantization-and-threshold-re-tuning)). A word can therefore carry several
categories at once. BERT subword pieces are merged back into whole words, and a
word takes the maximum probability of its pieces.

## Repository layout

```
app/                     Lambda container
  app.py                 handler: tokenize → ONNX inference → words/spans
  Dockerfile             public.ecr.aws/lambda/python:3.12 base image
  .dockerignore          keeps the fp32 model out of the image
  requirements.txt       runtime dependencies (pinned)
  model/                 labels, thresholds, tokenizer (*.onnx is not versioned)
export/                  offline tooling (run once, from the repo root)
  export_onnx.py         Hugging Face checkpoint → model.onnx (+ parity check)
  quantize_compare.py    model.onnx → model.int8.onnx, fp32 vs int8 comparison
  retune_thresholds.py   re-tunes the per-label thresholds for int8
  parity_check.py        deployed API (Graviton) vs local run of the same model
  *_results.json         results of the scripts above
infra/                   Terraform (state in S3)
  main.tf                ECR, Lambda, API Gateway routes and stage
  auth.tf                API key in SSM + Lambda authorizer
  alarms.tf              5xx-rate alarm → SNS → email
  frontend.tf            private S3 bucket + CloudFront for the website, proxying /predict to the API
  authorizer/            authorizer source code
  bootstrap/             one-time setup: state bucket, GitHub OIDC deploy role
frontend/                Next.js static site: the sentence with flagged words underlined by category
.github/workflows/
  deploy.yml             push to main → build arm64 image → terraform apply → smoke test → publish site
```

The website calls `/predict` on its own CloudFront domain. CloudFront forwards
that path to API Gateway and adds the `x-api-key` header itself, so the key
never reaches the browser. Calling API Gateway directly still requires the key.

## Rebuilding the model artifacts

The ONNX files are not in git (the fp32 model is 416 MB). To regenerate them:

```bash
pip install -r export/requirements.txt

# 1. Export pinthoz/gus-net-bert to ONNX. Checks that ONNX and PyTorch logits
#    match on a padded batch (max abs diff ~2e-6).
python export/export_onnx.py

# 2. Dynamic int8 quantization + fp32 vs int8 comparison
python export/quantize_compare.py

# 3. Re-tune the thresholds for int8 on the original validation split.
#    Needs the cleaned GUS-Net corpus from the Attention Atlas repo, checked
#    out next to this one (../attention-atlas/dataset/old-datasets/).
python export/retune_thresholds.py                 # on this machine (x86)
TARGET=lambda python export/retune_thresholds.py   # on the deployed Lambda (what is served)
```

## Quantization and threshold re-tuning

The fp32 model was too slow to cold-start on Lambda (see below), so the API
serves a dynamically quantized **int8** model. Quantization is not neutral for a
bias detector: it shifts the output probabilities, and the thresholds tuned for
fp32 then move decisions near the boundary. The thresholds were therefore
re-tuned for the int8 model, on the hardware that serves it.

`retune_thresholds.py` reproduces the split and procedure used to train the
published checkpoint (Attention Atlas, `colab_sparse_training_clean.ipynb`): the
cleaned corpus, `StratifiedKFold(5, shuffle=True, random_state=42)` with fold 4
held out as test (747 sentences), a 10 % stratified validation split (300
sentences), and a per-label grid search plus bounded refinement. As a check, the
same procedure applied to the fp32 model gives back the published thresholds
exactly, and its test metrics match the model card.

Two details decide what the thresholds must be tuned on:

- **Each sentence runs alone, unpadded**, as in an API request. Dynamic int8
  quantization scales activations over the whole input tensor, so padding and
  batch neighbours change the probabilities.
- **The CPU matters.** onnxruntime's int8 kernels differ per architecture (x86
  without VNNI saturates intermediate sums). Sending the same 256 sentences to
  the arm64 Lambda and to an x86 laptop gave different labels on 7.3 % of tokens
  and a different biased/not-biased verdict on 7 sentences
  ([export/parity_check.py](export/parity_check.py)). Emulating arm64 with QEMU
  did not reproduce Graviton's numbers either. So the deployed thresholds were
  tuned on probabilities computed by the Lambda itself (`TARGET=lambda`, through
  a direct-invocation hook that API Gateway cannot reach).

Test set on the Lambda (arm64), F1 / recall per category
([export/retune_results.arm64.json](export/retune_results.arm64.json)):

| | fp32 (reference) | int8, fp32 thresholds | int8, thresholds tuned on x86 | **int8, tuned on arm64 (deployed)** |
|---|---|---|---|---|
| GEN | 0.741 / 0.682 | 0.731 / 0.666 | 0.727 / 0.645 | 0.733 / 0.671 |
| UNFAIR | 0.451 / 0.497 | 0.429 / 0.425 | 0.406 / 0.463 | 0.445 / 0.476 |
| STEREO | 0.733 / 0.708 | 0.701 / 0.629 | 0.732 / 0.715 | 0.732 / 0.709 |
| micro | 0.637 / 0.645 | 0.616 / 0.589 | 0.622 / 0.631 | 0.630 / 0.640 |

With the fp32 thresholds, int8 loses about 8 points of stereotype recall.
Thresholds tuned on x86 recover STEREO but cost UNFAIR on Graviton. Tuning on
the serving hardware leaves every category within 0.008 F1 of fp32. The x86
results are in [export/retune_results.json](export/retune_results.json).

Per-channel weight quantization was also tried. On x86 CPUs it breaks the
model: UNFAIR F1 falls to 0, and 558 of 748 sentences flip their biased/not-biased
verdict ([export/quant_results_per_channel.json](export/quant_results_per_channel.json)).
The deployed model uses per-tensor quantization.

## Cold start

Lambda container images are loaded lazily, and the init phase gets 10 s before
it is retried inside the invocation. API Gateway HTTP APIs time out at 30 s.

| Arch | Memory | Model | Cold start | First request | Max memory used |
|---|---|---|---|---|---|
| x86_64 | 2048 MB | fp32 (416 MB) | > 40 s | 503 | – (never finished loading) |
| x86_64 | 3008 MB | fp32 (416 MB) | ~33.4 s (10 s timeout + 23.4 s) | 503 | 1132 MB |
| x86_64 | 3008 MB | int8 (105 MB) | ~13.5 s (10 s timeout + 3.5 s) | **200** | 347 MB |

A warm request takes about 40 ms. On x86_64, the model's output on AWS matched
the local ONNX run to the last reported decimal. The function now runs on arm64
(Graviton), which is about 20 % cheaper per GB-second.

## Deploying

Deploys run in GitHub Actions ([.github/workflows/deploy.yml](.github/workflows/deploy.yml)):
every push to `main` builds the arm64 image on a native arm64 runner, pushes it
to ECR tagged with the commit SHA, runs `terraform apply`, and smoke-tests the
live API (`200` with the key, `401` without). AWS access is keyless: the
workflow exchanges a GitHub OIDC token for short-lived credentials of a role
that only trusts the `main` branch of this repository.

### One-time setup

Requirements: an AWS account, the AWS CLI with admin credentials, and
Terraform ≥ 1.10. The region is `eu-west-1`.

```bash
# 1. State bucket + GitHub OIDC provider + deploy role (local state)
cd infra/bootstrap
terraform init
terraform apply

# 2. The model is not in git: upload it where CI fetches it
aws s3 cp ../../app/model/model.int8.onnx "s3://$(terraform output -raw bucket)/model/model.int8.onnx"
```

3. In the GitHub repository (Settings → Secrets and variables → Actions), add
   the variables `AWS_ROLE_ARN` (`terraform output -raw deploy_role_arn`) and
   `STATE_BUCKET` (`terraform output -raw bucket`), and the secret `ALERT_EMAIL`.
4. Push to `main`. After the first deploy, confirm the SNS subscription from the
   email AWS sends, or the alarm cannot reach you.

The API key is generated by Terraform and stored in SSM Parameter Store:

```bash
cd infra
terraform init
terraform output -raw api_key
```

### Running Terraform locally

`image_tag` and `alert_email` have no defaults. Pass the tag that is currently
deployed (the last commit SHA that CI pushed), or the plan will try to change the
image:

```bash
terraform plan -var image_tag=<commit sha> -var alert_email=<email>
```

`terraform destroy` removes everything except the bootstrap resources, including
the images (`force_delete = true` on the repository). The state bucket has
`prevent_destroy` set.

### Security and cost notes

- Every route requires the `x-api-key` header. HTTP APIs have no native API
  keys, so a small Lambda authorizer compares the header with a key stored as an
  SSM SecureString, using a constant-time comparison. Decisions are cached for
  5 minutes per key. To rotate the key, run `terraform apply -replace=random_password.api_key`.
- The stage is throttled to 2 requests/s with bursts of 5, which caps the cost
  of abuse even with a leaked key. The website's `/predict` proxy is open to
  anyone, so it shares that same budget.
- Each role gets the minimum: the model Lambda can only write logs, the
  authorizer can also read that one parameter, and API Gateway may invoke each
  function only from this API (`aws_lambda_permission` with `source_arn`). The
  CI role's permissions are scoped to this project's resources, but it can edit
  the `bias-api-*` roles, so anyone able to push to `main` effectively controls
  this project's AWS resources.
- A CloudWatch alarm emails the `ALERT_EMAIL` address when more than 5 % of
  requests in a 5-minute window return 5xx. With little traffic, a single cold-start
  503 can trip it.
- Logs are kept for 7 days. ECR keeps the 3 most recent images. Images are scanned
  on push (basic scanning covers OS packages only, not Python dependencies).
- New AWS accounts cap Lambda memory at 3008 MB until a quota increase is granted.

## Limitations

- English only. Input is truncated at 512 tokens.
- The model captures one specific operationalisation of social bias (explicit
  generalisations, unfair language and stereotypes about a group). Implicit or
  context-dependent bias may be missed. UNFAIR is the weakest category (F1 ≈ 0.44).
- Treat the output as evidence to review, not as ground truth. Do not use it for
  automated decisions about individuals.

## Credits

- Model: [`pinthoz/gus-net-bert`](https://huggingface.co/pinthoz/gus-net-bert) (Apache-2.0),
  trained on [`ethical-spectacle/gus-dataset-v1`](https://huggingface.co/datasets/ethical-spectacle/gus-dataset-v1).
- GUS-Net: Powers et al., *GUS-Net: Social Bias Classification in Text with
  Generalizations, Unfairness, and Stereotypes*, arXiv:2410.08388 (2024).
