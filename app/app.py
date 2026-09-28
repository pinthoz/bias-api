import base64
import json
import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/model")
MODEL_FILE = os.environ.get("MODEL_FILE", "model.onnx")
# Thresholds are tuned per model file (int8 shifts the probabilities)
THRESHOLDS_FILE = os.environ.get("THRESHOLDS_FILE", "thresholds.json")

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


def response(status, payload):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload, ensure_ascii=False),
    }


# Every request
def handler(event, context):
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode()
    try:
        text = json.loads(raw)["text"]
    except (json.JSONDecodeError, KeyError, TypeError):
        text = None
    if not isinstance(text, str) or not text.strip():
        return response(400, {"error": "send a JSON with a non-empty string field 'text'"})

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
    return response(
        200,
        {
            "biased": bool(spans),
            "categories": sorted({s["category"] for s in spans}),
            "biased_tokens": [t["token"] for t in tokens if t["labels"]],
            "spans": spans,
            "tokens": tokens,
        },
    )
