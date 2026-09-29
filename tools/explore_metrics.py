from collections import Counter

import boto3
from boto3.dynamodb.conditions import Key

table = boto3.resource("dynamodb").Table("bias-api-metrics")
items = table.query(KeyConditionExpression=Key("pk").eq("AGG#api"))["Items"]

total = Counter()
for it in items:
    total.update(
        {k: int(v) for k, v in it.items() if k not in ("pk", "sk", "expires_at")}
    )

print("requests:", total["n"], "| positive rate:", round(total["pos"] / total["n"], 3))
print("p_biased histogram:", [total[f"p_bin_{i}"] for i in range(10)])
for g in ("gender", "race", "religion", "age", "lgbtq"):
    if total[f"grp_{g}_n"]:
        print(g, round(total[f"grp_{g}_pos"] / total[f"grp_{g}_n"], 3))
