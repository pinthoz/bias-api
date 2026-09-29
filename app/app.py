import base64
import json
import os
import re
import time
from datetime import datetime, timezone

import boto3
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/model")
MODEL_FILE = os.environ.get("MODEL_FILE", "model.onnx")
TABLE = os.environ.get("METRICS_TABLE")  # without table = no logging
BUCKET_MIN = int(os.environ.get("BUCKET_MINUTES", "60"))
RETENTION_DAYS = 90

# Histogram limits: MUST be equal to monitor.py
P_BINS = 10  # p_biased in [0, 1]
TOK_BINS, TOK_MAX = 8, 256  # number of tokens
LAT_EDGES = [
    50,
    100,
    200,
    300,
    500,
    750,
    1000,
    2000,
    5000,
]  # ms; last interval = above

# Cold start
tokenizer = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
tokenizer.enable_truncation(max_length=256)
session = ort.InferenceSession(
    f"{MODEL_DIR}/{MODEL_FILE}", providers=["CPUExecutionProvider"]
)
INPUTS = {i.name for i in session.get_inputs()}
with open(f"{MODEL_DIR}/labels.json") as f:
    LABELS = {int(k): v for k, v in json.load(f).items()}
O_IDX = next(
    i for i, l in LABELS.items() if l == "O"
)  # labels BIO: O, B-STEREO, I-STEREO, ...
table = boto3.resource("dynamodb").Table(TABLE) if TABLE else None

# Minimal group vocabulary. Adjust it to the groups in your benchmark.
GROUP_TERMS = {
    "gender": ["women", "woman", "men", "man", "girls", "boys", "female", "male"],
    "race": ["black", "white", "asian", "latino", "latina", "hispanic"],
    "religion": [
        "muslim",
        "muslims",
        "christian",
        "christians",
        "jew",
        "jews",
        "jewish",
    ],
    "age": ["old", "elderly", "young", "teenagers", "boomers"],
    "lgbtq": ["gay", "lesbian", "trans", "transgender", "queer"],
}
GROUP_RE = {
    g: re.compile(r"\b(" + "|".join(ts) + r")\b", re.I) for g, ts in GROUP_TERMS.items()
}


def detect_groups(text):
    return [g for g, rx in GROUP_RE.items() if rx.search(text)]


def bucket_key(ts):
    """Arredonda para o início do intervalo: '2026-09-29T13:45'."""
    minute = (ts.hour * 60 + ts.minute) // BUCKET_MIN * BUCKET_MIN
    return ts.strftime("%Y-%m-%dT") + f"{minute // 60:02d}:{minute % 60:02d}"


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def analyse(logits, text, enc):
    """Converte logits por token em sinais por frase e em spans de texto."""
    probs = softmax(logits)[0]  # [tokens, n_labels]
    real = [
        i for i, m in enumerate(enc.special_tokens_mask) if m == 0
    ]  # remove [CLS]/[SEP]
    probs = probs[real]
    pred = [int(i) for i in probs.argmax(-1)]
    offsets = [enc.offsets[i] for i in real]

    # Probability of the most "suspect" token not being O: a score per sentence
    p_biased = float((1 - probs[:, O_IDX]).max()) if real else 0.0

    spans, cur = [], None
    for (s, e), idx in zip(offsets, pred):
        label = LABELS[idx]
        tag = label.split("-", 1)[1] if label != "O" else None
        if tag and (label.startswith("B-") or cur is None or cur["type"] != tag):
            cur = {"type": tag, "start": s, "end": e}  # starting a new span
            spans.append(cur)
        elif tag:
            cur["end"] = e  # continuing the span (I-)
        else:
            cur = None
    for sp in spans:
        sp["text"] = text[sp["start"] : sp["end"]]
    return p_biased, spans


def response(status, payload):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload, ensure_ascii=False),
    }


def record(source, p_biased, n_tokens, latency_ms, groups, tags):
    """Incrementing the counters of the current interval. A single UpdateItem per request."""
    if not table:
        return
    is_biased = bool(tags)
    adds = {
        "n": 1,
        "pos": int(is_biased),
        f"p_bin_{min(int(p_biased * P_BINS), P_BINS - 1)}": 1,
        f"tok_bin_{min(n_tokens * TOK_BINS // TOK_MAX, TOK_BINS - 1)}": 1,
        f"lat_bin_{sum(latency_ms > e for e in LAT_EDGES)}": 1,
    }
    for t in tags:  # sentences with at least one span of each type
        adds[f"tag_{t}"] = 1
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
    except Exception as e:  # never let the record break the prediction
        print(f"monitoring record failed: {e}")


# Each request
def handler(event, context):
    t0 = time.perf_counter()
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode()
    try:
        text = json.loads(raw)["text"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return response(400, {"erro": "envia um JSON com o campo 'text'"})

    enc = tokenizer.encode(text)
    feeds = {
        "input_ids": np.array([enc.ids], dtype=np.int64),
        "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
    }
    feeds = {k: v for k, v in feeds.items() if k in INPUTS}

    p_biased, spans = analyse(session.run(None, feeds)[0], text, enc)
    result = {
        "label": "biased" if spans else "neutral",  # compatible with the frontend
        "score": round(p_biased if spans else 1 - p_biased, 4),
        "p_biased": round(p_biased, 4),
        "spans": spans,  # ex.: [{"type": "GEN", "text": "Women are bad at math", ...}]
    }

    # Only two possible values: a header cannot create keys at the caller's choice
    headers = event.get("headers") or {}
    source = "canary" if headers.get("x-source") == "canary" else "api"
    record(
        source,
        p_biased,
        len(enc.ids),
        (time.perf_counter() - t0) * 1000,
        detect_groups(text),
        sorted({sp["type"] for sp in spans}),
    )
    return response(200, result)
