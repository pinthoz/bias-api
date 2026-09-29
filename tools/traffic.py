"""Sends texts from a CSV to the API at a controlled rate.

Usage: python3 tools/traffic.py data/source_a.csv --n 300 --rps 1.5
"""

import argparse
import csv
import json
import random
import subprocess
import time
import urllib.error
import urllib.request


def tf_output(name):
    out = subprocess.run(
        ["terraform", "-chdir=infra", "output", "-raw", name],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--n", type=int, default=200, help="number of requests")
    ap.add_argument(
        "--rps", type=float, default=1.5, help="requests per second (API limits to 2)"
    )
    ap.add_argument("--url", default=None)
    ap.add_argument("--key", default=None, help="API key (default: terraform output)")
    args = ap.parse_args()

    url = args.url or tf_output("api_url")
    key = args.key or tf_output("api_key")  # every route requires x-api-key
    with open(args.csv, newline="", encoding="utf-8") as f:
        texts = [r["text"] for r in csv.DictReader(f) if r.get("text")]

    ok = errors = 0
    for i in range(args.n):
        body = json.dumps({"text": random.choice(texts)}).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json", "x-api-key": key}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                json.load(r)
            ok += 1
        except urllib.error.HTTPError as e:
            errors += 1
            if e.code in (401, 403):
                raise SystemExit(f"\nHTTP {e.code}: the API key is missing or wrong")
            if e.code == 429:
                time.sleep(2)  # slow down
        print(f"\r{i + 1}/{args.n}  ok={ok}  errors={errors}", end="", flush=True)
        time.sleep(1 / args.rps)
    print()


if __name__ == "__main__":
    main()
