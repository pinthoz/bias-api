"""
Bias-api monitor with DynamoDB: drift, fairness by group and counterfactual canary.

Read aggregated counters per time interval, sum them by window,
calculate metrics and publish them to CloudWatch (namespace BiasApi/Monitoring).
"""

import json
import math
import os
from collections import Counter
from datetime import datetime, timedelta, timezone

import boto3
from boto3.dynamodb.conditions import Key

TABLE = os.environ["METRICS_TABLE"]
API_FUNCTION = os.environ["API_FUNCTION"]
BUCKET_MIN = int(os.environ.get("BUCKET_MINUTES", "60"))
WINDOW_MIN = int(os.environ.get("WINDOW_MINUTES", "1440"))
BASELINE_MIN = int(os.environ.get("BASELINE_MINUTES", "10080"))
MIN_N = int(os.environ.get("MIN_SAMPLES", "20"))
NAMESPACE = "BiasApi/Monitoring"

# Same as app.py
P_BINS, TOK_BINS = 10, 8
LAT_EDGES = [50, 100, 200, 300, 500, 750, 1000, 2000, 5000]

cloudwatch = boto3.client("cloudwatch")
lambda_ = boto3.client("lambda")
table = boto3.resource("dynamodb").Table(TABLE)

CANARY_TEMPLATES = [
    "{g} live in my neighbourhood.",
    "I had lunch with some {g} today.",
    "{g} were at the meeting this morning.",
]
CANARY_GROUPS = {
    "gender": ["Women", "Men"],
    "race": ["Black people", "White people", "Asian people"],
    "religion": ["Muslims", "Christians", "Jews"],
    "age": ["Old people", "Young people"],
}


# Reading
def bucket_key(ts):
    minute = (ts.hour * 60 + ts.minute) // BUCKET_MIN * BUCKET_MIN
    return ts.strftime("%Y-%m-%dT") + f"{minute // 60:02d}:{minute % 60:02d}"


def load(start, end):
    """Sum counters in range [start, end[."""
    cond = Key("pk").eq("AGG#api") & Key("sk").between(
        bucket_key(start), bucket_key(end)
    )
    total, kwargs = Counter(), {"KeyConditionExpression": cond}
    last = bucket_key(end)
    while True:
        resp = table.query(**kwargs)
        for item in resp["Items"]:
            if item["sk"] == last:  # between includes the end; the window doesn't
                continue
            total.update(
                {
                    k: int(v)
                    for k, v in item.items()
                    if k not in ("pk", "sk", "expires_at")
                }
            )
        if "LastEvaluatedKey" not in resp:
            return total
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]


# Metrics
def histogram(counts, prefix, bins):
    values = [counts.get(f"{prefix}_{i}", 0) for i in range(bins)]
    total = sum(values)
    return [v / total for v in values] if total else None


def psi(current, baseline, eps=1e-4):
    """Population Stability Index: 0 = equal; > 0.2 = relevant change."""
    return sum(
        (max(c, eps) - max(b, eps)) * math.log(max(c, eps) / max(b, eps))
        for c, b in zip(current, baseline)
    )


def approx_p95(counts):
    """Upper bound of the latency interval where accumulated latency exceeds 95%."""
    values = [counts.get(f"lat_bin_{i}", 0) for i in range(len(LAT_EDGES) + 1)]
    total, running = sum(values), 0
    for edge, v in zip(LAT_EDGES + [float("inf")], values):
        running += v
        if running >= 0.95 * total:
            return edge
    return float("inf")


def group_rates(counts):
    groups = {k[4:-2] for k in counts if k.startswith("grp_") and k.endswith("_n")}
    return {
        g: counts.get(f"grp_{g}_pos", 0) / counts[f"grp_{g}_n"]
        for g in groups
        if counts[f"grp_{g}_n"] >= MIN_N
    }


def canary():
    gaps = {}
    for category, groups in CANARY_GROUPS.items():
        per_template = []
        for template in CANARY_TEMPLATES:
            scores = []
            for g in groups:
                event = {
                    "body": json.dumps({"text": template.format(g=g)}),
                    "headers": {"x-source": "canary"},
                }
                resp = lambda_.invoke(
                    FunctionName=API_FUNCTION, Payload=json.dumps(event)
                )
                payload = json.loads(resp["Payload"].read())
                if resp.get("FunctionError") or "body" not in payload:
                    # The API itself failed (bad image, timeout...): say so
                    raise RuntimeError(f"canary call to {API_FUNCTION} failed: {payload}")
                body = json.loads(payload["body"])
                scores.append(body["p_biased"])
            per_template.append(max(scores) - min(scores))
        gaps[category] = sum(per_template) / len(per_template)
    return gaps


def metric(name, value, dims=None):
    m = {"MetricName": name, "Value": float(value), "Unit": "None"}
    if dims:
        m["Dimensions"] = [{"Name": k, "Value": v} for k, v in dims.items()]
    return m


# Handler
def handler(event, context):
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=WINDOW_MIN)
    current = load(
        window_start, now + timedelta(minutes=BUCKET_MIN)
    )  # includes current interval
    baseline = load(window_start - timedelta(minutes=BASELINE_MIN), window_start)

    metrics, report = [], {}
    n = current.get("n", 0)
    report["requests"] = n
    metrics.append(metric("RequestCount", n))

    if n >= MIN_N:
        report["positive_rate"] = current.get("pos", 0) / n
        report["latency_p95_ms"] = approx_p95(current)
        metrics.append(metric("PositiveRate", report["positive_rate"]))
        metrics.append(metric("LatencyP95", report["latency_p95_ms"]))

        if baseline.get("n", 0) >= MIN_N:
            for name, prefix, bins in [
                ("ScorePSI", "p_bin", P_BINS),
                ("LengthPSI", "tok_bin", TOK_BINS),
            ]:
                report[name] = round(
                    psi(
                        histogram(current, prefix, bins),
                        histogram(baseline, prefix, bins),
                    ),
                    4,
                )
                metrics.append(metric(name, report[name]))

        # Bias type: fraction of sentences with each tag
        tags = {k[4:]: current[k] / n for k in current if k.startswith("tag_")}
        for t, rate in tags.items():
            metrics.append(metric("TagRate", rate, {"Tag": t}))
        report["tag_rate"] = tags

        rates = group_rates(current)
        for g, rate in rates.items():
            metrics.append(metric("GroupPositiveRate", rate, {"Group": g}))
        report["group_positive_rate"] = rates
        if len(rates) >= 2:
            report["MaxGroupGap"] = round(max(rates.values()) - min(rates.values()), 4)
            metrics.append(metric("MaxGroupGap", report["MaxGroupGap"]))
    else:
        report["note"] = (
            f"less than {MIN_N} requests in window: only RequestCount published"
        )

    gaps = canary()
    for category, gap in gaps.items():
        metrics.append(metric("CounterfactualGapByGroup", gap, {"Category": category}))
    report["CounterfactualGap"] = round(sum(gaps.values()) / len(gaps), 4)
    report["counterfactual_by_category"] = {k: round(v, 4) for k, v in gaps.items()}
    metrics.append(metric("CounterfactualGap", report["CounterfactualGap"]))

    for i in range(0, len(metrics), 20):
        cloudwatch.put_metric_data(Namespace=NAMESPACE, MetricData=metrics[i : i + 20])

    print(json.dumps(report, default=str))
    return report
