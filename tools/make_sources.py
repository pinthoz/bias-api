"""Builds the two traffic sources used to simulate drift (monitoring worksheet, step 5).

  data/source_a.csv  sentences from the corpus GUS-Net was trained on (the reference)
  data/source_b.csv  news from AG News: another domain, longer texts (the drift)

Both are random samples with a fixed seed, so the files are reproducible.
Usage (from the repo root): python3 tools/make_sources.py [--n-a 500] [--n-b 300]
Needs: pip install --user datasets; the Attention Atlas repo next to this one.
"""

import argparse
import csv
import html
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

CLEAN = Path("../attention-atlas/dataset/old-datasets/gus_dataset_clean.json")
OUT = Path("data")
SEED = 42


def detokenize(text):
    """The cleaned corpus splits punctuation into its own tokens ('week ?'):
    glue it back so the requests look like text people type."""
    text = re.sub(r"\s+([?.!,;:%)\]])", r"\1", text)
    text = re.sub(r"([(\[$])\s+", r"\1", text)
    return re.sub(r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", text)


def clean_news(text):
    """AG News uses '\\' for line breaks and has half-escaped HTML ('#39;s')."""
    text = text.replace("\\", " ")
    text = re.sub(r"&?#(\d+);", lambda m: chr(int(m.group(1))), text)
    text = html.unescape(re.sub(r"(?<!&)\b(quot|amp|lt|gt);", r"&\1;", text))
    return re.sub(r"\s+", " ", text).strip()


def write(path, texts):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text"])
        w.writerows([t] for t in texts)
    words = [len(t.split()) for t in texts]
    print(f"{path}: {len(texts)} texts, {sum(words) / len(words):.1f} words on average "
          f"(min {min(words)}, max {max(words)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-a", type=int, default=500)
    ap.add_argument("--n-b", type=int, default=300)
    args = ap.parse_args()
    rng = random.Random(SEED)
    OUT.mkdir(exist_ok=True)

    # Source A: the training distribution
    corpus = [detokenize(r["text_str"]) for r in json.loads(CLEAN.read_text(encoding="utf-8"))]
    write(OUT / "source_a.csv", rng.sample(corpus, args.n_a))

    # Source B: news articles, at least 35 words, well under the 512-token limit
    news = load_dataset("fancyzhx/ag_news", split="test")["text"]
    news = [clean_news(t) for t in news]
    news = [t for t in news if 35 <= len(t.split()) <= 120]
    write(OUT / "source_b.csv", rng.sample(news, args.n_b))


if __name__ == "__main__":
    main()
