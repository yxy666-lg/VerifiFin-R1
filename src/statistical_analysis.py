#!/usr/bin/env python3
"""对逐题配对结果计算 bootstrap 置信区间和 McNemar 精确检验。"""

import argparse
import json
import math
import random


def paired_bootstrap(rows, baseline_key, treatment_key, samples=10000, seed=42):
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        picked = [rows[rng.randrange(len(rows))] for _ in rows]
        deltas.append(sum(r[treatment_key] - r[baseline_key] for r in picked) / len(picked))
    deltas.sort()
    return deltas[int(samples * 0.025)], deltas[int(samples * 0.975)]


def mcnemar_exact(rows, baseline_key, treatment_key):
    improved = sum((not r[baseline_key]) and r[treatment_key] for r in rows)
    regressed = sum(r[baseline_key] and (not r[treatment_key]) for r in rows)
    n = improved + regressed
    if n == 0:
        return improved, regressed, 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(improved, regressed) + 1)) / (2 ** n)
    return improved, regressed, min(1.0, 2 * tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL，每行含两个布尔正确性字段")
    parser.add_argument("--baseline_key", default="baseline_correct")
    parser.add_argument("--treatment_key", default="bon_correct")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with open(args.input, encoding="utf-8-sig") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = [r for r in rows if args.baseline_key in r and args.treatment_key in r]
    if not rows:
        raise SystemExit("没有可比较的逐题记录")
    base = sum(bool(r[args.baseline_key]) for r in rows) / len(rows)
    treatment = sum(bool(r[args.treatment_key]) for r in rows) / len(rows)
    low, high = paired_bootstrap(rows, args.baseline_key, args.treatment_key,
                                 args.bootstrap_samples, args.seed)
    improved, regressed, p_value = mcnemar_exact(rows, args.baseline_key, args.treatment_key)
    print(json.dumps({
        "n": len(rows), "baseline_accuracy": base, "treatment_accuracy": treatment,
        "delta_pp": (treatment - base) * 100,
        "delta_95ci_pp": [low * 100, high * 100],
        "improved": improved, "regressed": regressed, "mcnemar_exact_p": p_value,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
