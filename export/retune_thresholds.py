"""Re-optimize the per-label thresholds for the int8 model.

Reproduces the split and procedure of attention-atlas'
colab_sparse_training_clean.ipynb (the run published as pinthoz/gus-net-bert):
  - cleaned corpus, labels on the first subtoken of each word only
  - StratifiedKFold(5, shuffle, seed 42) by bias presence, fold 4 = test
  - validation = 10 % of fold 4's train part, stratified, seed 42
  - per label: grid search 0.05..0.95 (step 0.025) on val F1, then a
    bounded refinement of +-0.05

Sanity check: re-running the procedure on the fp32 model must give back the
published thresholds, which shows the validation split was reproduced.

Run from the repo root, after quantize_compare.py:
    python3 export/retune_thresholds.py
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoTokenizer

CLEAN = Path("../attention-atlas/dataset/old-datasets/gus_dataset_clean.json")
MODEL_DIR = Path("app/model")
RESULTS = Path("export/retune_results.json")
SEED, FOLD, MAX_LENGTH, BATCH = 42, 4, 128, 32

LABELS = {int(k): v for k, v in json.loads((MODEL_DIR / "labels.json").read_text()).items()}
LABEL2ID = {v: k for k, v in LABELS.items()}
N = len(LABELS)
PUBLISHED = np.array(json.loads((MODEL_DIR / "thresholds.json").read_text()))
CATEGORIES = {
    cat: [i for i, lab in LABELS.items() if lab.endswith(f"-{cat}")]
    for cat in ("GEN", "UNFAIR", "STEREO")
}

# 1. Tokenize exactly as in training (secondary subtokens masked with -100)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
ids, masks, labels = [], [], []
for row in json.loads(CLEAN.read_text(encoding="utf-8")):
    words, tags = row["text_str"].split(), row["ner_tags"]
    n = min(len(words), len(tags))
    enc = tok(words[:n], is_split_into_words=True, truncation=True,
              max_length=MAX_LENGTH, padding="max_length")
    y = np.full((MAX_LENGTH, N), -100, dtype=np.int64)
    prev = None
    for t, wid in enumerate(enc.word_ids()):
        if wid is not None and wid != prev:
            y[t] = 0
            for tag in tags[wid]:
                if tag in LABEL2ID:
                    y[t, LABEL2ID[tag]] = 1
        prev = wid
    ids.append(enc["input_ids"])
    masks.append(enc["attention_mask"])
    labels.append(y)
ids, masks, labels = np.array(ids), np.array(masks), np.array(labels)

# 2. Same split as the notebook
valid_tok = labels[..., 0] != -100
strat = ((labels[..., 1:] == 1).any(-1) & valid_tok).any(-1).astype(int)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
train_val_idx, test_idx = list(skf.split(np.arange(len(strat)), strat))[FOLD]
_, val_idx = train_test_split(train_val_idx, test_size=0.10,
                              stratify=strat[train_val_idx], random_state=SEED)
print(f"sentences: {len(strat)}  val: {len(val_idx)}  test: {len(test_idx)}")


def predict(model_file, idx):
    sess = ort.InferenceSession(str(MODEL_DIR / model_file), providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    out = []
    for b in range(0, len(idx), BATCH):
        sel = idx[b:b + BATCH]
        feeds = {"input_ids": ids[sel], "attention_mask": masks[sel],
                 "token_type_ids": np.zeros_like(ids[sel])}
        out.append(sess.run(None, {k: v for k, v in feeds.items() if k in names})[0])
    probs = 1 / (1 + np.exp(-np.concatenate(out)))
    keep = valid_tok[idx]
    return probs[keep], labels[idx][keep]  # [tokens, N] each


def f1(y, p, t):
    return f1_score(y, (p >= t).astype(int), average="binary", zero_division=0)


def optimize(probs, gold):
    """Grid search + bounded refinement, as in the training notebook."""
    thr = np.zeros(N)
    for c in range(N):
        best_f, best_t = 0, 0.5
        for t in np.arange(0.05, 0.96, 0.025):
            f = f1(gold[:, c], probs[:, c], t)
            if f > best_f:
                best_f, best_t = f, t
        thr[c] = best_t
        res = minimize_scalar(lambda t, c=c: -f1(gold[:, c], probs[:, c], t),
                              bounds=(max(0.01, best_t - 0.05), min(0.99, best_t + 0.05)),
                              method="bounded")
        if -res.fun >= best_f:
            thr[c] = res.x
    return thr


def prf(pred, true):
    tp, fp, fn = (pred & true).sum(), (pred & ~true).sum(), (~pred & true).sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(float(p), 4), "recall": round(float(r), 4),
            "f1": round(float(2 * p * r / (p + r)) if p + r else 0.0, 4)}


def evaluate(probs, gold, thr):
    fired, true = probs >= thr, gold == 1
    per_cat = {cat: prf(fired[:, idx].any(1), true[:, idx].any(1)) for cat, idx in CATEGORIES.items()}
    bias = [i for idx in CATEGORIES.values() for i in idx]
    per_cat["micro"] = prf(fired[:, bias].ravel(), true[:, bias].ravel())
    return per_cat


val = {m: predict(m, val_idx) for m in ("model.onnx", "model.int8.onnx")}
test = {m: predict(m, test_idx) for m in ("model.onnx", "model.int8.onnx")}

thr_fp32 = optimize(*val["model.onnx"])
thr_int8 = optimize(*val["model.int8.onnx"])
print("published      :", np.round(PUBLISHED, 4))
print("re-derived fp32:", np.round(thr_fp32, 4), " max diff", round(float(np.abs(thr_fp32 - PUBLISHED).max()), 4))
print("int8           :", np.round(thr_int8, 4))

results = {
    "split": {"sentences": int(len(strat)), "val": int(len(val_idx)), "test": int(len(test_idx))},
    "thresholds": {
        "published_fp32": PUBLISHED.round(4).tolist(),
        "rederived_fp32": thr_fp32.round(4).tolist(),
        "int8": thr_int8.round(4).tolist(),
    },
    "test": {
        "fp32 + published thresholds": evaluate(*test["model.onnx"], PUBLISHED),
        "int8 + published thresholds": evaluate(*test["model.int8.onnx"], PUBLISHED),
        "int8 + int8 thresholds": evaluate(*test["model.int8.onnx"], thr_int8),
    },
}
RESULTS.write_text(json.dumps(results, indent=2))
(MODEL_DIR / "thresholds.int8.json").write_text(json.dumps(thr_int8.tolist()))
for name, metrics in results["test"].items():
    print(f"{name:30s}", {c: (m["f1"], m["recall"]) for c, m in metrics.items()})
