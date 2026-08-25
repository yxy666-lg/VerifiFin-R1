#!/usr/bin/env python3
"""从冲突表生成50条模型辅助初审；结果不是人工金标。"""

import argparse
import csv
import json


def review(row):
    tags = set(filter(None, row.get("auto_tags", "").split("|")))
    direction = row["direction"]
    if direction == "judge_negative_rule_positive":
        if "unsupported_assumption" in tags or "option_mismatch" in tags:
            return {
                "assistant_verdict": "judge_likely_correct",
                "assistant_confidence": "high",
                "assistant_reason": "算式局部成立，但步骤包含题外假设、按选项反推或题目信息不足；规则通过不能证明整步逻辑正确。",
                "needs_expert_review": "no",
                "review_evidence": "unsupported assumption / option reverse-engineering",
            }
        return {
            "assistant_verdict": "ambiguous",
            "assistant_confidence": "low",
            "assistant_reason": "规则仅证明局部算式成立；是否采用正确公式或金融规则需要领域知识确认。",
            "needs_expert_review": "yes",
            "review_evidence": "local arithmetic only",
        }
    # Judge 正、规则负：优先视为高风险候选，但链式等式和百分号可能触发解析错误。
    if "formula_chain" in tags or "percentage" in tags:
        return {
            "assistant_verdict": "parser_or_arithmetic_conflict",
            "assistant_confidence": "medium",
            "assistant_reason": "规则检测到至少一个失败算式，但链式等式或百分号可能导致抽取错位；需要按原式复算后才能区分模型算错与解析器误报。",
            "needs_expert_review": "yes",
            "review_evidence": "rule_calcs contains failed expression",
        }
    return {
        "assistant_verdict": "rule_likely_correct",
        "assistant_confidence": "medium",
        "assistant_reason": "Judge判正但确定性规则未通过，属于高风险伪正确候选。",
        "needs_expert_review": "yes",
        "review_evidence": "judge positive / rule negative",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/conflict_audit.csv")
    parser.add_argument("--output", default="reports/assisted_review_50.csv")
    parser.add_argument("--summary", default="reports/assisted_review_50_summary.json")
    args = parser.parse_args()
    with open(args.input, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    positive = [r for r in rows if r["direction"] == "judge_positive_rule_negative"]
    negative = [r for r in rows if r["direction"] == "judge_negative_rule_positive"]
    negative.sort(key=lambda r: (
        "unsupported_assumption" not in r["auto_tags"] and "option_mismatch" not in r["auto_tags"],
        r["audit_id"],
    ))
    selected = positive + negative[:max(0, 50 - len(positive))]
    reviewed = []
    for row in selected:
        out = dict(row)
        # 明确删除空的人工作答栏，避免把本结果误传为人工金标。
        for key in ("human_verdict", "primary_error_type", "review_note"):
            out.pop(key, None)
        out.update(review(row))
        reviewed.append(out)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=reviewed[0].keys())
        writer.writeheader()
        writer.writerows(reviewed)
    summary = {
        "review_type": "model_assisted_not_human_gold",
        "selected": len(reviewed),
        "all_judge_positive_rule_negative": len(positive),
        "sampled_judge_negative_rule_positive": len(reviewed) - len(positive),
        "verdict_counts": {}, "expert_review_required": 0,
    }
    for row in reviewed:
        verdict = row["assistant_verdict"]
        summary["verdict_counts"][verdict] = summary["verdict_counts"].get(verdict, 0) + 1
        summary["expert_review_required"] += row["needs_expert_review"] == "yes"
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
