# bias-api

A serverless API for **token-level social-bias detection**. It serves
[GUS-Net (BERT)](https://huggingface.co/pinthoz/gus-net-bert), a fine-tuned
`bert-base-uncased` token classifier developed in the
[Attention Atlas](https://github.com/pinthoz/attention-atlas) project. Given a
sentence, the API returns the words that carry bias and the type of bias:
**generalisation (GEN)**, **unfair language (UNFAIR)** or **stereotype (STEREO)**.

The model runs as an int8-quantized ONNX graph inside an AWS Lambda container
image, behind an API Gateway HTTP API. All the infrastructure is defined in
Terraform.

```
client ──POST /predict──► API Gateway (HTTP API, throttled) ──► Lambda (container image)
                                                                  ├─ tokenizers (tokenizer.json)
                                                                  └─ onnxruntime (model.int8.onnx)
```

## API

`POST /predict` takes a JSON body with a `text` field:

```bash
curl -s -X POST "$API_URL" -H "Content-Type: application/json" \
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
  *_results.json         results of the two scripts above
infra/                   Terraform: ECR, Lambda, IAM, API Gateway, CloudWatch
```

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
python export/retune_thresholds.py
```

## Quantization and threshold re-tuning

The fp32 model was too slow to cold-start on Lambda (see below), so the API
serves a dynamically quantized **int8** model. Quantization is not neutral for a
bias detector: it shifts the output probabilities, and the thresholds tuned for
fp32 then move decisions near the boundary. The thresholds were therefore
re-tuned for the int8 model.

`retune_thresholds.py` reproduces the split and procedure used to train the
published checkpoint (Attention Atlas, `colab_sparse_training_clean.ipynb`): the
cleaned corpus, `StratifiedKFold(5, shuffle=True, random_state=42)` with fold 4
held out as test (747 sentences), a 10 % stratified validation split (300
sentences), and a per-label grid search plus bounded refinement. As a check, the
same procedure applied to the fp32 model gives back the published thresholds
exactly, and its test metrics match the model card.

Test set, F1 / recall per category ([export/retune_results.json](export/retune_results.json)):

| | fp32 (reference) | int8, fp32 thresholds | **int8, re-tuned (deployed)** |
|---|---|---|---|
| GEN | 0.741 / 0.682 | 0.740 / 0.694 | 0.729 / 0.664 |
| UNFAIR | 0.451 / 0.497 | 0.424 / 0.423 | 0.435 / 0.478 |
| STEREO | 0.733 / 0.708 | 0.705 / 0.635 | 0.732 / 0.707 |

With the fp32 thresholds, int8 loses about 7 points of stereotype recall.
Re-tuning recovers it and leaves every category within about 0.016 F1 of fp32.
GEN pays a small part of the cost.

Per-channel weight quantization was also tried. On x86 CPUs it breaks the
model: UNFAIR F1 falls to 0, and 558 of 748 sentences flip their biased/not-biased
verdict ([export/quant_results_per_channel.json](export/quant_results_per_channel.json)).
The deployed model uses per-tensor quantization.

## Cold start

Lambda container images are loaded lazily, and the init phase gets 10 s before
it is retried inside the invocation. API Gateway HTTP APIs time out at 30 s.

| Memory | Model | Cold start | First request | Max memory used |
|---|---|---|---|---|
| 2048 MB | fp32 (416 MB) | > 40 s | 503 | – (never finished loading) |
| 3008 MB | fp32 (416 MB) | ~33.4 s (10 s timeout + 23.4 s) | 503 | 1132 MB |
| 3008 MB | int8 (105 MB) | ~13.5 s (10 s timeout + 3.5 s) | **200** | 347 MB |

A warm request takes about 40 ms. The model's output on AWS matched the local
ONNX run to the last reported decimal.

## Deploying

Requirements: an AWS account, the AWS CLI with credentials, Docker and
Terraform ≥ 1.6. The region defaults to `eu-west-1`
([infra/variables.tf](infra/variables.tf)).

```bash
# 1. Create the ECR repository first (the Lambda needs an image to exist)
cd infra
terraform init
terraform apply -target=aws_ecr_repository.api -target=aws_ecr_lifecycle_policy.api
REPO=$(terraform output -raw ecr_url)

# 2. Build, test locally and push the image
cd ../app
docker build --platform linux/amd64 --provenance=false -t "$REPO:v3" .
docker run --rm -p 9000:8080 "$REPO:v3"   # in another terminal:
curl -s -X POST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -d '{"body": "{\"text\": \"Women are bad at math.\"}"}'
aws ecr get-login-password --region eu-west-1 \
  | docker login --username AWS --password-stdin "${REPO%/*}"
docker push "$REPO:v3"

# 3. Create the rest (image_tag in variables.tf must match the pushed tag)
cd ../infra
terraform apply
curl -s -X POST "$(terraform output -raw api_url)" \
  -H "Content-Type: application/json" -d '{"text": "Women are bad at math."}'
```

`--platform linux/amd64` builds for Lambda's x86 architecture, and
`--provenance=false` avoids a multi-platform manifest that Lambda rejects.
To release a new version, push a new tag, update `image_tag` and run
`terraform apply`. The plan should show only `image_uri` changing.

`terraform destroy` removes everything, including the images
(`force_delete = true` on the repository).

### Security and cost notes

- The endpoint is **public** (`authorization_type = "NONE"`). It is throttled
  to 2 requests/s with bursts of 5, which caps the cost of abuse.
- The Lambda's execution role only allows writing logs
  (`AWSLambdaBasicExecutionRole`). API Gateway may invoke the function only from
  this API (`aws_lambda_permission` with `source_arn`).
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
