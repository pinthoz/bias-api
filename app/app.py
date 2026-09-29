import base64
import json
import os
import re
import time
from datetime import datetime, timezone

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/model")
MODEL_FILE = os.environ.get("MODEL_FILE", "model.onnx")
# Thresholds are tuned per model file (int8 shifts the probabilities)
THRESHOLDS_FILE = os.environ.get("THRESHOLDS_FILE", "thresholds.json")
MAX_BATCH = int(os.environ.get("MAX_BATCH", "32"))

# Monitoring: per-interval counters in DynamoDB (infra/dynamodb.tf).
# No table configured = no recording (local runs, tests)
TABLE = os.environ.get("METRICS_TABLE")
BUCKET_MIN = int(os.environ.get("BUCKET_MINUTES", "60"))
RETENTION_DAYS = 90

# Histogram edges: MUST match monitor.py
P_BINS = 10  # p_biased in [0, 1]
TOK_BINS, TOK_MAX = 8, 256  # number of tokens
LAT_EDGES = [50, 100, 200, 300, 500, 750, 1000, 2000, 5000]  # ms; last bin = above

# Cold start
tokenizer = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
tokenizer.enable_truncation(max_length=512)
session = ort.InferenceSession(
    f"{MODEL_DIR}/{MODEL_FILE}", providers=["CPUExecutionProvider"]
)
INPUTS = {i.name for i in session.get_inputs()}
with open(f"{MODEL_DIR}/labels.json") as f:
    LABELS = {int(k): v for k, v in json.load(f).items()}
with open(f"{MODEL_DIR}/{THRESHOLDS_FILE}") as f:
    THRESHOLDS = np.array(json.load(f))

# GUS-Net is multi-label per token: a category fires when its B- or I- prob
# reaches that label's F1-optimised threshold (same rule as attention-atlas)
CATEGORIES = {
    cat: [i for i, lab in LABELS.items() if lab.endswith(f"-{cat}")]
    for cat in ("GEN", "UNFAIR", "STEREO")
}
BIAS_IDX = [i for idx in CATEGORIES.values() for i in idx]

if TABLE:
    import boto3  # in the Lambda runtime; not needed without a table

    table = boto3.resource("dynamodb").Table(TABLE)
else:
    table = None

# Minimal group vocabulary. Adjust it to the groups in your benchmark.
GROUP_TERMS = {
    "gender": ["women", "woman", "men", "man", "girls", "boys", "female", "male"],
    "race": ["black", "white", "asian", "latino", "latina", "hispanic"],
    "religion": ["muslim", "muslims", "christian", "christians", "jew", "jews", "jewish"],
    "age": ["old", "elderly", "young", "teenagers", "boomers"],
    "lgbtq": ["gay", "lesbian", "trans", "transgender", "queer"],
}
GROUP_RE = {
    g: re.compile(r"\b(" + "|".join(ts) + r")\b", re.I) for g, ts in GROUP_TERMS.items()
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def find_words(text, enc, probs):
    """Group subword tokens into words; a word takes the max prob of its pieces."""
    words = []
    for t, wid in enumerate(enc.word_ids):
        if wid is None:  # [CLS], [SEP]
            continue
        start, end = enc.offsets[t]
        if words and words[-1]["wid"] == wid:
            words[-1]["end"] = end
            words[-1]["probs"] = np.maximum(words[-1]["probs"], probs[t])
        else:
            words.append({"wid": wid, "start": start, "end": end, "probs": probs[t]})

    tokens = []
    for w in words:
        p = w["probs"]
        fired = {
            LABELS[i]: round(float(p[i]), 4)
            for idx in CATEGORIES.values()
            for i in idx
            if p[i] >= THRESHOLDS[i]
        }
        tokens.append(
            {
                "token": text[w["start"] : w["end"]],
                "start": w["start"],
                "end": w["end"],
                "labels": list(fired),
                "categories": sorted({lab.split("-", 1)[1] for lab in fired}),
                "scores": fired,
            }
        )
    return tokens


def find_spans(text, tokens):
    """Merge consecutive words flagged with the same category into spans."""
    spans = []
    open_spans = {}
    for tok in tokens:
        for cat in CATEGORIES:
            span = open_spans.get(cat)
            if cat not in tok["categories"]:
                open_spans.pop(cat, None)
                continue
            score = max(s for lab, s in tok["scores"].items() if lab.endswith(cat))
            if span is None:
                span = {"category": cat, "start": tok["start"], "end": tok["end"], "score": score}
                open_spans[cat] = span
                spans.append(span)
            else:
                span["end"] = tok["end"]
                span["score"] = max(span["score"], score)
    for span in spans:
        span["text"] = text[span["start"] : span["end"]]
    return spans


def analyze(text):
    """Returns the API result for one text and its number of tokens."""
    enc = tokenizer.encode(text)
    feeds = {
        "input_ids": np.array([enc.ids], dtype=np.int64),
        "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
    }
    feeds = {k: v for k, v in feeds.items() if k in INPUTS}

    probs = sigmoid(session.run(None, feeds)[0])[0]  # [seq, num_labels]
    tokens = find_words(text, enc, probs)
    spans = find_spans(text, tokens)
    # Sentence-level score for monitoring: the most suspect label on any real
    # token (special tokens excluded), before thresholds
    real = [t for t, wid in enumerate(enc.word_ids) if wid is not None]
    p_biased = float(probs[real][:, BIAS_IDX].max()) if real else 0.0
    return {
        "biased": bool(spans),
        "categories": sorted({s["category"] for s in spans}),
        "biased_tokens": [t["token"] for t in tokens if t["labels"]],
        "p_biased": round(p_biased, 4),
        "spans": spans,
        "tokens": tokens,
    }, len(enc.ids)


def raw_probs(event):
    """Sigmoid probabilities for pre-tokenized input, for threshold tuning.

    int8 outputs depend on the CPU's int8 kernels, so thresholds must be tuned
    on the hardware that serves the model (export/retune_thresholds.py with
    TARGET=lambda). Each sequence runs alone and unpadded, exactly like a
    /predict request. Returns one [seq, labels] list per sequence.
    """
    probs = []
    for ids in event["eval_inputs"]:
        ids = np.array([ids], dtype=np.int64)
        feeds = {
            "input_ids": ids,
            "attention_mask": np.ones_like(ids),
            "token_type_ids": np.zeros_like(ids),
        }
        out = session.run(None, {k: v for k, v in feeds.items() if k in INPUTS})[0][0]
        probs.append(np.round(sigmoid(out), 5).tolist())
    return {"probs": probs}


def detect_groups(text):
    return [g for g, rx in GROUP_RE.items() if rx.search(text)]


def bucket_key(ts):
    """Rounds down to the start of the interval: '2026-09-29T13:00'."""
    minute = (ts.hour * 60 + ts.minute) // BUCKET_MIN * BUCKET_MIN
    return ts.strftime("%Y-%m-%dT") + f"{minute // 60:02d}:{minute % 60:02d}"


def record(source, p_biased, n_tokens, latency_ms, groups, categories):
    """Increments the counters of the current interval: one UpdateItem per text."""
    if not table:
        return
    is_biased = bool(categories)
    adds = {
        "n": 1,
        "pos": int(is_biased),
        f"p_bin_{min(int(p_biased * P_BINS), P_BINS - 1)}": 1,
        f"tok_bin_{min(n_tokens * TOK_BINS // TOK_MAX, TOK_BINS - 1)}": 1,
        f"lat_bin_{sum(latency_ms > e for e in LAT_EDGES)}": 1,
    }
    for c in categories:  # texts with at least one span of each category
        adds[f"tag_{c}"] = 1
    for g in groups:
        adds[f"grp_{g}_n"] = 1
        adds[f"grp_{g}_pos"] = int(is_biased)
    try:
        table.update_item(
            Key={"pk": f"AGG#{source}", "sk": bucket_key(datetime.now(timezone.utc))},
            UpdateExpression="ADD "
            + ", ".join(f"{k} :{k}" for k in adds)
            + " SET expires_at = if_not_exists(expires_at, :ttl)",
            ExpressionAttributeValues={
                **{f":{k}": v for k, v in adds.items()},
                ":ttl": int(time.time()) + RETENTION_DAYS * 86400,
            },
        )
    except Exception as e:  # monitoring must never break a prediction
        print(f"monitoring record failed: {e}")


def analyze_and_record(text, source):
    t0 = time.perf_counter()
    result, n_tokens = analyze(text)
    record(
        source,
        result["p_biased"],
        n_tokens,
        (time.perf_counter() - t0) * 1000,
        detect_groups(text),
        result["categories"],
    )
    return result


def response(status, payload):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload, ensure_ascii=False),
    }


def is_text(value):
    return isinstance(value, str) and bool(value.strip())


# Every request
def handler(event, context):
    # Direct `aws lambda invoke` only (needs IAM rights on the function):
    # API Gateway events always carry a requestContext
    if "eval_inputs" in event and "requestContext" not in event:
        return raw_probs(event)

    # Only two possible values: a header cannot create keys of the caller's choice
    headers = event.get("headers") or {}
    source = "canary" if headers.get("x-source") == "canary" else "api"

    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode()
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        body = None
    if not isinstance(body, dict):
        return response(400, {"error": "send a JSON object"})

    if event.get("routeKey", "").endswith("/predict/batch"):
        texts = body.get("texts")
        if (
            not isinstance(texts, list)
            or not 1 <= len(texts) <= MAX_BATCH
            or not all(is_text(t) for t in texts)
        ):
            return response(
                400,
                {"error": f"send a JSON with 'texts': a list of 1 to {MAX_BATCH} non-empty strings"},
            )
        # One ONNX run per text, not one padded batch: with dynamic int8
        # quantization the activation scale is computed over the whole input
        # tensor, so batching would make each result depend on its neighbours
        return response(200, {"results": [analyze_and_record(t, source) for t in texts]})

    text = body.get("text")
    if not is_text(text):
        return response(400, {"error": "send a JSON with a non-empty string field 'text'"})
    return response(200, analyze_and_record(text, source))
