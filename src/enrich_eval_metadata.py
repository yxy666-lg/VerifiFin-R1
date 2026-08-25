#!/usr/bin/env python3
"""为自研金融推理集补充可复现的题型、难度、可验算性和陷阱标签。"""

import argparse
import hashlib
import json
import re

from calc_verifier import verify_response_steps


CATEGORIES = [
    ("taxation", r"税|增值税|所得税|消费税"),
    ("bond_and_fixed_income", r"债券|久期|到期收益率"),
    ("corporate_valuation", r"DCF|估值|市盈率|P/E|自由现金流"),
    ("risk_and_portfolio", r"CAPM|夏普|β|VaR|投资组合|标准差"),
    ("corporate_finance", r"WACC|NPV|资本成本|项目投资|融资"),
    ("financial_statement", r"流动比率|速动比率|ROE|杜邦|资产负债率|周转率|毛利率|净利率|报表"),
    ("derivatives", r"期权|期货|远期|互换"),
    ("compliance", r"合规|监管|洗钱|内幕|准则|规定"),
]


def category(question):
    for name, pattern in CATEGORIES:
        if re.search(pattern, question, re.I):
            return name
    return "general_finance"


def traps(question):
    found = []
    for name, pattern in {
        "percentage": r"%|率|百分点", "unit": r"万元|亿元|元|倍|天",
        "multi_metric": r"和|分别|分解", "decision": r"判断|是否|高估|低估|应该",
        "time_value": r"贴现|现值|未来|年期", "missing_information": r"信息不足|无法|缺少",
    }.items():
        if re.search(pattern, question):
            found.append(name)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/eval/finance_reasoning.jsonl")
    parser.add_argument("--output", default="data/eval/finance_reasoning_enriched.jsonl")
    args = parser.parse_args()
    output = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            question, steps = item["question"], item.get("steps", [])
            verification = verify_response_steps(steps)
            complexity = len(steps) + len(re.findall(r"[+\-×÷*/]", " ".join(steps)))
            difficulty = "hard" if complexity >= 9 else "medium" if complexity >= 5 else "easy"
            enriched = dict(item)
            enriched["metadata"] = {
                "question_id": hashlib.sha1(question.encode()).hexdigest()[:12],
                "category": category(question),
                "difficulty": difficulty,
                "gold_steps": len(steps),
                "verifiable": verification["total_calcs"] > 0,
                "verifiable_calcs": verification["total_calcs"],
                "error_traps": traps(question),
                "metadata_source": "rule_v1_needs_human_review",
            }
            output.append(enriched)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {len(output)} enriched records to {args.output}")


if __name__ == "__main__":
    main()
