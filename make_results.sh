#!/bin/bash

# Usage:
# bash make_results.sh            -> default outputs/
# bash make_results.sh outputs_x

OUT_DIR=${1:-outputs}

python - <<EOF
import os
import json
import csv

out_dir = "${OUT_DIR}"
out_csv = os.path.join(out_dir, "results.csv")

rows = []

# ---------- load ----------
for fname in sorted(os.listdir(out_dir)):
    if not fname.endswith(".json"):
        continue

    with open(os.path.join(out_dir, fname), "r") as f:
        data = json.load(f)

    acc = data["acc_list"]
    t1, t2, t3, t4 = acc
    tmean = sum(acc) / len(acc)

    clip = data["target_domain"]["clipart"]
    quick = data["target_domain"]["quickdraw"]

    c1, c2, c3, c4 = clip["task_acc"]
    q1, q2, q3, q4 = quick["task_acc"]

    cmean = clip["mean_acc"]
    qmean = quick["mean_acc"]
    tdmean = (cmean + qmean) / 2

    rows.append({
        "file": fname,
        "clipart_mean": cmean,
        "quickdraw_mean": qmean,
        "target_domain_mean": tdmean,
        "train_domain_mean": tmean,
        "train_domain1": t1,
        "train_domain2": t2,
        "train_domain3": t3,
        "train_domain4": t4,
        "clipart_class1": c1,
        "clipart_class2": c2,
        "clipart_class3": c3,
        "clipart_class4": c4,
        "quickdraw_class1": q1,
        "quickdraw_class2": q2,
        "quickdraw_class3": q3,
        "quickdraw_class4": q4,
    })

# ---------- rank (WITHOUT reordering rows) ----------
sorted_by_target = sorted(
    rows,
    key=lambda x: x["target_domain_mean"],
    reverse=True
)

rank_map = {
    r["file"]: i + 1
    for i, r in enumerate(sorted_by_target)
}

for r in rows:
    r["rank"] = rank_map[r["file"]]

# ---------- write CSV ----------
fields = [
    "file",
    "clipart_mean",
    "quickdraw_mean",
    "target_domain_mean",
    "rank",
    "train_domain_mean",
    "train_domain1",
    "train_domain2",
    "train_domain3",
    "train_domain4",
    "clipart_class1",
    "clipart_class2",
    "clipart_class3",
    "clipart_class4",
    "quickdraw_class1",
    "quickdraw_class2",
    "quickdraw_class3",
    "quickdraw_class4",
]

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(fields)

    for r in rows:
        writer.writerow([
            r["file"],
            f"{r['clipart_mean']:.2f}",
            f"{r['quickdraw_mean']:.2f}",
            f"{r['target_domain_mean']:.2f}",
            r["rank"],
            f"{r['train_domain_mean']:.2f}",
            f"{r['train_domain1']:.2f}",
            f"{r['train_domain2']:.2f}",
            f"{r['train_domain3']:.2f}",
            f"{r['train_domain4']:.2f}",
            f"{r['clipart_class1']:.2f}",
            f"{r['clipart_class2']:.2f}",
            f"{r['clipart_class3']:.2f}",
            f"{r['clipart_class4']:.2f}",
            f"{r['quickdraw_class1']:.2f}",
            f"{r['quickdraw_class2']:.2f}",
            f"{r['quickdraw_class3']:.2f}",
            f"{r['quickdraw_class4']:.2f}",
        ])

# ---------- print TOP-5 ----------
print("\\n===== TOP 5 (Target Domain Mean) =====")
for r in sorted_by_target[:5]:
    print(
        f"Rank {rank_map[r['file']]:>2} | "
        f"{r['file']} | "
        f"{r['target_domain_mean']:.2f} "
        f"(Clipart {r['clipart_mean']:.2f}, Quickdraw {r['quickdraw_mean']:.2f})"
    )
print("=====================================")
EOF