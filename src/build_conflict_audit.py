#!/usr/bin/env python3
"""导出规则验证器与 LLM Judge 的冲突样本，供人工复核与错误归因。"""

import argparse
import csv
import json
import os
import re
from collections import Counter


def error_tags(text):
    tags = []
    rules = {
        "percentage": r"%|％|百分点|税率|利率|折扣率",
        "unit": r"万元|亿元|元|万|亿|倍|天|头|股",
        "tax_rule": r"增值税|消费税|土地增值税|所得税|契税|税额|税率",
        "unsupported_assumption": r"假设|可能|反推|常见版本|隐含|题目.*不足|未明确|笔误",
        "rounding": r"四舍五入|约为|精确到|保留.*位",
        "formula_chain": r"=.*=.*=|×|÷|\*|/",
        "option_mismatch": r"选项.*不|不符合选项|不在选项|对应选项",
    }
    for name, pattern in rules.items():
        if re.search(pattern, text, re.I | re.S):
            tags.append(name)
    return tags or ["other"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/prm/fin_prm_verified.jsonl")
    parser.add_argument("--output_csv", default="reports/conflict_audit.csv")
    parser.add_argument("--output_json", default="reports/conflict_summary.json")
    args = parser.parse_args()
    rows, tag_counts, directions = [], Counter(), Counter()
    with open(args.input, encoding="utf-8") as f:
        for trajectory_id, line in enumerate(f):
            item = json.loads(line)
            for step_index, step in enumerate(item.get("steps", []), 1):
                rule_label, judge_label = step.get("rule_label"), step.get("label")
                if rule_label is None:
                    continue
                judge_correct, rule_correct = judge_label == 1, rule_label == 2
                if judge_correct == rule_correct:
                    continue
                direction = "judge_positive_rule_negative" if judge_correct else "judge_negative_rule_positive"
                tags = error_tags(step.get("step", ""))
                directions[direction] += 1
                tag_counts.update(tags)
                rows.append({
                    "audit_id": f"T{trajectory_id:05d}-S{step_index:02d}",
                    "question": item.get("question", ""),
                    "step": step.get("step", ""),
                    "judge_label": judge_label,
                    "rule_label": rule_label,
                    "direction": direction,
                    "auto_tags": "|".join(tags),
                    "rule_calcs": json.dumps(step.get("rule_calcs", []), ensure_ascii=False),
                    "human_verdict": "",  # judge_correct / rule_correct / both_wrong / ambiguous
                    "primary_error_type": "",
                    "review_note": "",
                })
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "disagreements": len(rows),
        "direction_counts": directions,
        "heuristic_tag_counts": tag_counts,
        "important_caveat": "冲突不等于 Judge 错误；必须填写 human_verdict 后才能计算真实性能。",
        "allowed_human_verdicts": ["judge_correct", "rule_correct", "both_wrong", "ambiguous"],
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
