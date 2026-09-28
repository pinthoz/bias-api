import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForTokenClassification, AutoTokenizer

# GUS-Net BERT (token classification, 7 BIO labels, per-label sigmoid).
SRC = "pinthoz/gus-net-bert"
OUT = Path("app/model")
OUT.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(SRC)
model = AutoModelForTokenClassification.from_pretrained(SRC).eval()

sample = tok("An example sentence.", return_tensors="pt")
# input_names are bound by position to forward()'s signature, not by key, so
# follow that order (the tokenizer returns token_type_ids before attention_mask)
names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in sample]
dynamic = {n: {0: "batch", 1: "seq"} for n in names}
dynamic["logits"] = {0: "batch", 1: "seq"}  # [batch, seq, num_labels]

torch.onnx.export(
    model,
    args=(),
    kwargs=dict(sample),
    f=str(OUT / "model.onnx"),
    input_names=names,
    output_names=["logits"],
    dynamic_axes=dynamic,
    opset_version=17,
    dynamo=False,
)

# tokenizer.json is all the Lambda needs to tokenize
tok.save_pretrained(OUT)
(OUT / "labels.json").write_text(json.dumps(model.config.id2label))

# Per-label thresholds (applied to sigmoid(logits)), F1-optimised
thr = np.load(hf_hub_download(SRC, "optimized_thresholds.npy"))
(OUT / "thresholds.json").write_text(json.dumps(thr.tolist()))

# Validation: PyTorch and ONNX must produce the same logits
# on a padded batch with a different length than the trace sample
check = tok(
    ["Women are too emotional to be good leaders.", "Hi."],
    padding=True,
    return_tensors="pt",
)
with torch.no_grad():
    ref = model(**check).logits.numpy()
sess = ort.InferenceSession(str(OUT / "model.onnx"))
out = sess.run(None, {k: v.numpy() for k, v in check.items()})[0]
print("max difference:", np.abs(ref - out).max())
assert np.allclose(ref, out, atol=1e-4)
