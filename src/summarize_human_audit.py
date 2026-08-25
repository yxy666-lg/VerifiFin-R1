#!/usr/bin/env python3
"""汇总已填写的冲突人工复核表；未复核样本不会进入分母。"""

import argparse
import csv
import json
from collections import Counter


VALID = {"judge_correct", "rule_correct", "both_wrong", "ambiguous"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/conflict_audit.csv")
    args = parser.parse_args()
    with open(args.input, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    reviewed = [r for r in rows if r.get("human_verdict") in VALID]
    verdicts = Counter(r["human_verdict"] for r in reviewed)
    error_types = Counter(r.get("primary_error_type") or "unclassified" for r in reviewed)
    report = {
        "total_conflicts": len(rows), "reviewed": len(reviewed),
        "review_coverage": round(len(reviewed) / len(rows) * 100, 2) if rows else 0,
        "verdict_counts": verdicts, "primary_error_types": error_types,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
