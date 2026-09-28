"""Compare the deployed API (arm64 / Graviton) with the same model run locally.

onnxruntime picks different int8 kernels per CPU, so the probabilities on
Graviton may differ slightly from the x86 run the thresholds were tuned on.
This sends sentences from gus-dataset-v1 to /predict/batch and diffs every
token's fired labels against a local run of app.analyze().

Run from the repo root:
    API_URL=$(terraform -chdir=infra output -raw api_batch_url) \
    API_KEY=$(terraform -chdir=infra output -raw api_key) \
    python3 export/parity_check.py
"""

import json
import os
import sys
import time
import urllib.request

from datasets import load_dataset

N_SENTENCES = 256
BATCH = 32

os.environ.setdefault("MODEL_DIR", "app/model")
os.environ.setdefault("MODEL_FILE", "model.int8.onnx")
os.environ.setdefault("THRESHOLDS_FILE", "thresholds.int8.json")
sys.path.insert(0, "app")
import app  # noqa: E402  (reads the env vars above at import)

texts = load_dataset("ethical-spectacle/gus-dataset-v1", split="train")["text_str"][:N_SENTENCES]


def remote(batch):
    req = urllib.request.Request(
        os.environ["API_URL"],
        data=json.dumps({"texts": batch}).encode(),
        headers={"Content-Type": "application/json", "x-api-key": os.environ["API_KEY"]},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["results"]


api = []
for b in range(0, len(texts), BATCH):
    api += remote(texts[b:b + BATCH])
    time.sleep(0.6)  # stay under the stage throttle (2 req/s)
local = [app.analyze(t) for t in texts]

same_sentences = sum(a == b for a, b in zip(api, local))
label_diffs = score_diffs = n_tokens = verdict_flips = 0
for a, b in zip(api, local):
    verdict_flips += a["biased"] != b["biased"]
    for ta, tb in zip(a["tokens"], b["tokens"]):
        n_tokens += 1
        label_diffs += set(ta["labels"]) != set(tb["labels"])
        common = ta["scores"].keys() & tb["scores"].keys()
        score_diffs = max([score_diffs] + [abs(ta["scores"][k] - tb["scores"][k]) for k in common])

print(json.dumps({
    "sentences": len(texts),
    "identical_responses": same_sentences,
    "tokens": n_tokens,
    "tokens_with_different_labels": label_diffs,
    "biased_verdict_flips": verdict_flips,
    "max_score_diff_on_shared_labels": round(score_diffs, 4),
}, indent=2))
