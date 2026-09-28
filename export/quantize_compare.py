"""Quantize the exported GUS-Net model to int8 and compare it against fp32.

Both models are evaluated on the same held-out split of
ethical-spectacle/gus-dataset-v1 (80/20, random_state=42, as in attention-atlas'
span_faithfulness_eval.py), with the same per-label thresholds the API uses.

Run from the repo root, after export_onnx.py:
    python3 export/quantize_compare.py
"""

import ast
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from datasets import load_dataset
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

MODEL_DIR = Path("app/model")
FP32 = MODEL_DIR / "model.onnx"
INT8 = MODEL_DIR / "model.int8.onnx"
RESULTS = Path("export/quant_results.json")
MAX_LENGTH = 128
BATCH = 32

LABELS = {int(k): v for k, v in json.loads((MODEL_DIR / "labels.json").read_text()).items()}
LABEL2ID = {v: k for k, v in LABELS.items()}
THRESHOLDS = np.array(json.loads((MODEL_DIR / "thresholds.json").read_text()))
CATEGORIES = {
    cat: [i for i, lab in LABELS.items() if lab.endswith(f"-{cat}")]
    for cat in ("GEN", "UNFAIR", "STEREO")
}

# 1. Quantize (weights only; activations are quantized on the fly at runtime)
quantize_dynamic(str(FP32), str(INT8), weight_type=QuantType.QInt8)
sizes = {p.name: round(p.stat().st_size / 2**20, 1) for p in (FP32, INT8)}
print("sizes (MB):", sizes)

# 2. Held-out split, word-level tags aligned to every subword (as in training)
data = []
for row in load_dataset("ethical-spectacle/gus-dataset-v1", split="train"):
    try:
        data.append((row["text_str"], ast.literal_eval(row["ner_tags"])))
    except (ValueError, SyntaxError):
        continue
_, test = train_test_split(data, test_size=0.20, random_state=42, shuffle=True)
print("test sentences:", len(test))

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
enc_ids, enc_mask, gold = [], [], []
for text, tags in test:
    words = text.split()
    n = min(len(words), len(tags))
    enc = tok(words[:n], is_split_into_words=True, truncation=True,
              max_length=MAX_LENGTH, padding="max_length")
    y = np.full((MAX_LENGTH, len(LABELS)), -1, dtype=np.int8)  # -1 = ignore
    for t, wid in enumerate(enc.word_ids()):
        if wid is not None:
            y[t] = 0
            for tag in tags[wid]:
                if tag in LABEL2ID:
                    y[t, LABEL2ID[tag]] = 1
    enc_ids.append(enc["input_ids"])
    enc_mask.append(enc["attention_mask"])
    gold.append(y)
enc_ids, enc_mask, gold = (np.array(a) for a in (enc_ids, enc_mask, gold))
valid = gold[..., 0] >= 0  # real word tokens only


def run(path):
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    load_s = time.perf_counter() - t0
    feeds_all = {"input_ids": enc_ids, "attention_mask": enc_mask,
                 "token_type_ids": np.zeros_like(enc_ids)}
    names = {i.name for i in sess.get_inputs()}
    out = []
    for b in range(0, len(enc_ids), BATCH):
        feeds = {k: v[b:b + BATCH].astype(np.int64) for k, v in feeds_all.items() if k in names}
        out.append(sess.run(None, feeds)[0])
    probs = 1 / (1 + np.exp(-np.concatenate(out)))
    # single-sentence latency, like one API request
    one = {k: v[:1].astype(np.int64) for k, v in feeds_all.items() if k in names}
    sess.run(None, one)
    t0 = time.perf_counter()
    for _ in range(20):
        sess.run(None, one)
    latency_ms = (time.perf_counter() - t0) / 20 * 1000
    return probs, load_s, latency_ms


def prf(pred, true):
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "support": tp + fn}


def evaluate(probs):
    fired = probs >= THRESHOLDS  # [n, seq, labels]
    true = gold == 1
    per_label = {LABELS[i]: prf(fired[..., i][valid], true[..., i][valid]) for i in LABELS}
    per_category = {
        cat: prf(fired[..., idx].any(-1)[valid], true[..., idx].any(-1)[valid])
        for cat, idx in CATEGORIES.items()
    }
    return fired, {"per_label": per_label, "per_category": per_category}


results = {"sizes_mb": sizes, "test_sentences": len(test)}
fired = {}
for name, path in (("fp32", FP32), ("int8", INT8)):
    probs, load_s, latency_ms = run(path)
    fired[name], metrics = evaluate(probs)
    results[name] = {"session_load_s": round(load_s, 2),
                     "latency_ms_one_sentence": round(latency_ms, 1), **metrics}
    results[name + "_probs"] = probs  # kept for the agreement check below

# 3. How often does int8 change a decision fp32 made?
p32, p8 = results.pop("fp32_probs"), results.pop("int8_probs")
diff = np.abs(p32 - p8)[valid]
cat_fired = {n: np.stack([f[..., idx].any(-1) for idx in CATEGORIES.values()], -1)
             for n, f in fired.items()}
sent_biased = {n: (c & valid[..., None]).any((1, 2)) for n, c in cat_fired.items()}
results["agreement"] = {
    "prob_abs_diff_max": round(float(diff.max()), 4),
    "prob_abs_diff_mean": round(float(diff.mean()), 5),
    "token_label_decisions_changed": int((fired["fp32"] != fired["int8"])[valid].sum()),
    "token_label_decisions_total": int(valid.sum() * len(LABELS)),
    "token_category_agreement": {
        cat: round(float((cat_fired["fp32"][..., j] == cat_fired["int8"][..., j])[valid].mean()), 5)
        for j, cat in enumerate(CATEGORIES)
    },
    "sentence_biased_flips": int((sent_biased["fp32"] != sent_biased["int8"]).sum()),
}

RESULTS.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
