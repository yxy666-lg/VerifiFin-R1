#!/usr/bin/env python3
"""
Fin-R1 可验证过程监督数据生成（路线 A 的核心）

用规则引擎（calc_verifier）自动给 PRM 轨迹的每一步打标签，
替代/校正 LLM-as-Judge 的标注，产出"可验证"的金融 PRM 数据。

背景（面试讲点）：
  LLM-judge 带标准答案逐步骤标注有 11.34% 的"伪正确"数值盲区（结论对但算式错被标正确）。
  本模块把规则引擎从"分析脚本"升级为"数据生成管线"：
    - 对每个含算式的步骤，用 Python 安全求值器验算，自动打 label（0=错 / 2=对）
    - 与 judge 标签做一致性对比，量化 judge 的盲区与误判
    - 输出一个可验证的 PRM 数据子集（可用于未来训练，无人工标注）

用法：
  python src/build_verified_prm.py \
    --data data/prm/finance_prm.jsonl \
    --output data/prm/fin_prm_verified.jsonl
"""

import argparse
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calc_verifier import extract_calcs, safe_eval  # noqa: E402


# 规则引擎标签约定（与 PRM 三分类对齐）：
#   2 = 正确（所有算式验算通过）
#   0 = 错误（至少一个算式算错）
#   None = 不可验（该步无算式，纯概念/定义/推理，规则引擎不覆盖）
# 注意：规则引擎只覆盖"计算"维度。逻辑/概念错误不在其能力内——这是它的诚实边界。
RULE_CORRECT = 2
RULE_WRONG = 0


def verify_step_by_rules(step_text: str) -> dict:
    """用规则引擎验算一步。返回：
    {"has_calc": bool, "rule_label": int|None, "calcs": [...]}"""
    calcs = extract_calcs(step_text)
    if not calcs:
        return {"has_calc": False, "rule_label": None, "calcs": []}

    results = []
    all_ok = True
    for expr, claimed_raw in calcs:
        try:
            computed = safe_eval(expr)
            claimed = safe_eval(claimed_raw)
            ok = abs(computed - claimed) <= 0.01 * max(1.0, abs(claimed))
        except Exception:
            # 解析失败（如复杂函数式），不算"算错"，标记不可判定
            ok = None
        results.append({"expr": expr, "computed": None, "claimed": claimed_raw, "ok": ok})
        if ok is False:
            all_ok = False

    # 只要有一个明确算错 → 该步错（金融"一步错全错"特性）
    if all_ok:
        return {"has_calc": True, "rule_label": RULE_CORRECT, "calcs": results}
    return {"has_calc": True, "rule_label": RULE_WRONG, "calcs": results}


def build(args):
    items = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line.strip()))
            if len(items) >= args.max_trajectories:
                break

    out_items = []
    # 汇总统计
    total_steps = 0
    verifyable = 0            # 含算式可验的步骤
    rule_wrong = 0            # 规则引擎判定算错的步骤
    judge_agree = 0           # judge 与 rule 一致（judge正确&rule正确 + judge非正确&rule错）
    judge_correct_rule_wrong = 0   # judge标正确但rule算错（核心盲区）
    judge_correct_rule_correct = 0 # judge标正确且rule算对
    judge_wrong_rule_correct = 0   # judge标非正确但rule算对（judge误杀）
    no_calc = 0               # 无可验算式（纯概念步）

    for it in items:
        question = it.get("question", "")
        steps = it.get("steps", [])
        if not steps:
            continue
        new_steps = []
        for s in steps:
            text = s.get("step", "")
            judge_label = s.get("label")
            v = verify_step_by_rules(text)
            total_steps += 1
            if v["has_calc"]:
                verifyable += 1
                rule_label = v["rule_label"]
                # judge 三分类 -> 二分类：label==1 视为正确，-1/0 视为非正确
                judge_binary = (judge_label == 1)
                rule_binary = (rule_label == RULE_CORRECT)
                if rule_label == RULE_WRONG:
                    rule_wrong += 1
                if judge_binary == rule_binary:
                    judge_agree += 1
                if judge_binary and not rule_binary:
                    judge_correct_rule_wrong += 1
                if judge_binary and rule_binary:
                    judge_correct_rule_correct += 1
                if not judge_binary and rule_binary:
                    judge_wrong_rule_correct += 1
                # 写回：附上 rule_label 与验算明细
                s["rule_label"] = rule_label
                s["rule_calcs"] = v["calcs"]
                s["label_source"] = "rule" if rule_label == RULE_WRONG else s.get("label_source", "judge")
            else:
                no_calc += 1
                s["rule_label"] = None
                s["label_source"] = s.get("label_source", "judge")
            new_steps.append(s)
        it["steps"] = new_steps
        out_items.append(it)

    # 写入可验证数据集（保留全部轨迹，rule_label 作为额外信号）
    with open(args.output, "w", encoding="utf-8") as f:
        for it in out_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # 统计输出
    stats = {
        "trajectories": len(out_items),
        "total_steps": total_steps,
        "verifyable_steps": verifyable,
        "verifyable_rate": round(verifyable / total_steps * 100, 2) if total_steps else None,
        "no_calc_steps": no_calc,
        "rule_wrong_steps": rule_wrong,
        "rule_calc_accuracy": round(
            (verifyable - rule_wrong) / verifyable * 100, 2) if verifyable else None,
        # judge 一致性（只统计可验步骤）
        "judge_agree_rate": round(judge_agree / verifyable * 100, 2) if verifyable else None,
        "judge_correct_rule_correct": judge_correct_rule_correct,
        "judge_correct_rule_wrong": judge_correct_rule_wrong,
        "judge_pseudo_correct_rate": round(
            judge_correct_rule_wrong / (judge_correct_rule_correct + judge_correct_rule_wrong) * 100, 2)
        if (judge_correct_rule_correct + judge_correct_rule_wrong) else None,
        "judge_wrong_rule_correct": judge_wrong_rule_correct,
    }
    with open(args.output.replace(".jsonl", "_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n=== 可验证过程监督数据生成（{stats['trajectories']} 条轨迹）===")
    print(f"总步骤: {total_steps}")
    print(f"可验步骤: {verifyable}（{stats['verifyable_rate']}%），无算式概念步: {no_calc}")
    print(f"规则引擎判定算错: {rule_wrong}，可验步骤算式正确率: {stats['rule_calc_accuracy']}%")
    print(f"\n-- judge 一致性（仅可验步骤）--")
    print(f"judge与规则一致率: {stats['judge_agree_rate']}%")
    print(f"judge标正确且规则算对: {judge_correct_rule_correct}")
    print(f"judge标正确但规则算错（伪正确盲区）: {judge_correct_rule_wrong}")
    print(f"伪正确率: {stats['judge_pseudo_correct_rate']}%")
    print(f"judge标非正确但规则算对（judge误杀）: {judge_wrong_rule_correct}")
    print(f"\n数据已写: {args.output}")
    print(f"统计已写: {args.output.replace('.jsonl', '_stats.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fin-R1 可验证过程监督数据生成")
    parser.add_argument("--data", default="data/prm/finance_prm.jsonl")
    parser.add_argument("--output", default="data/prm/fin_prm_verified.jsonl")
    parser.add_argument("--max_trajectories", type=int, default=2000,
                        help="最多处理轨迹数（默认2000，与 calc_verifier 口径一致）")
    args = parser.parse_args()
    build(args)
