#!/usr/bin/env python3
"""
Fin-R1 计算步骤规则引擎校验（金融专属差异化）

用 Python 安全地验算模型推理步骤中的算式（如 5000×15% = 750、16950÷1.13 = 15000），
找出"文字漂亮但数值算错"的步骤——这是 LLM-Judge 的盲区，也是医疗场景做不到的。

用法：
  # 校验单个步骤
  python src/calc_verifier.py --step "第2步：净利润=5000×15%=750万"

  # 扫描 PRM 数据，统计计算错误（尤其被 LLM-Judge 标"正确"但数值错的）
  python src/calc_verifier.py --analyze data/prm/finance_prm.jsonl

面试讲点：
  金融计算步骤可用规则引擎验算到 ground truth，弥补 LLM-Judge 对数值错误的盲区，
  这是金融场景独有的可验证性（医疗的临床判断无法用计算器验证）。
"""

import argparse
import ast
import json
import operator
import re

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def safe_eval(expr: str) -> float:
    """安全计算算术表达式（只允许数字/运算符/括号/百分号，无任意代码执行）"""
    expr = expr.strip()
    if not expr:
        raise ValueError("空表达式")
    # 把百分号转成分数（15% -> (15/100.0)）
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100.0)", expr)
    node = ast.parse(expr, mode="eval").body
    return float(_eval_node(node))


def _eval_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand)
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


def normalize_calc(text: str) -> str:
    """归一化中文计算文本：统一符号、去千分位、去单位"""
    t = text.replace("×", "*").replace("x", "*").replace("X", "*")
    t = t.replace("÷", "/").replace("／", "/").replace("＋", "+").replace("－", "-")
    t = t.replace("，", ",").replace("（", "(").replace("）", ")")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)          # 去千分位逗号
    t = re.sub(r"(?<=\d)\s+(?=\d{3})", "", t)        # 去空格千分位（"1 000" -> "1000"）
    # 去复合单位密度（"万元/年"、"元/件"、"万元/人"）——先于单单位，避免残留 "/年"
    t = re.sub(r"(?<=\d)(?:万元|亿元|元|万|亿)/(?:年|月|日|人|件|股|个|吨|次|份|米|平方米|㎡)", "", t)
    t = re.sub(r"(?<=\d)\s*(万元|亿元|元|万|亿|倍|股|点|吨|个)", "", t)  # 去单单位
    return t


# 计算链 + 声称结果（"a op b ... = result"，result 可带 %）。
# 关键设计：
#   1. 声称值 (claim) 是 `=` 后紧跟的数字，且后面**不是运算符**才算终结
#      —— 这样 "= 85，85+5" 只认第一个 85（逗号分隔两段）；
#   2. 链式中途等号（"= 150000 × 0.2 = 30000"）：`=` 后如果跟的是"完整中间链段"
#      （数字 op 数字 ...），说明还没到最终结果，继续并入，直到遇纯数字声称值。
CALC_RESULT_RE = re.compile(
    r"(-?\d+(?:\.\d+)?%?(?:\s*[+\-*/(]\s*-?\d+(?:\.\d+)?%?)+)"           # 首段计算链
    r"(?:\s*(?:=|等于|为|得)\s*-?\d+(?:\.\d+)?%?(?:\s*[+\-*/(]\s*-?\d+(?:\.\d+)?%?)+)*"  # 跨中间链段
    r"\s*(?:=|等于|为|得|约|结果为)\s*"
    r"(-?\d+(?:\.\d+)?%?)(?!\s*[+\-*×÷/])"
)


def extract_calcs(text: str):
    """提取 (表达式, 声称结果) 对"""
    norm = normalize_calc(text)
    found = []
    for m in CALC_RESULT_RE.finditer(norm):
        found.append((m.group(1).strip(), m.group(2).replace(",", "").strip()))
    return found


def verify_step(step_text: str, tolerance: float = 0.01):
    """校验一步里的算式。返回 [(expr, computed, claimed, ok)]"""
    results = []
    for expr, claimed_raw in extract_calcs(step_text):
        try:
            computed = safe_eval(expr)
            claimed = safe_eval(claimed_raw)
            ok = abs(computed - claimed) <= tolerance * max(1.0, abs(claimed))
            results.append({"expr": expr, "computed": computed, "claimed": claimed, "ok": ok})
        except Exception as e:
            results.append({"expr": expr, "computed": None, "claimed": claimed_raw, "ok": None, "error": str(e)[:40]})
    return results


def verify_response_steps(steps: list) -> dict:
    """校验一条轨迹的所有步骤，返回汇总"""
    total_calcs = 0
    correct_calcs = 0
    wrong_steps = []
    for s in steps:
        text = s.get("step", "") if isinstance(s, dict) else str(s)
        v = verify_step(text)
        step_calc = len(v)
        step_ok = sum(1 for r in v if r.get("ok"))
        total_calcs += step_calc
        correct_calcs += step_ok
        if v and step_ok < step_calc:
            wrong_steps.append({"step": text[:60], "detail": [r for r in v if not r.get("ok")]})
    return {
        "total_calcs": total_calcs,
        "correct_calcs": correct_calcs,
        "calc_accuracy": round(correct_calcs / total_calcs * 100, 2) if total_calcs else None,
        "wrong_steps": wrong_steps[:5],
    }


def analyze_prm(data_path: str, max_trajectories: int = 2000) -> dict:
    """扫描 PRM 数据，统计计算错误，尤其"LLM-Judge 标正确但数值算错"的步骤"""
    items = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line.strip()))
            if len(items) >= max_trajectories:
                break

    total_calcs = 0
    correct_calcs = 0
    judged_correct_with_calc = 0
    judge_correct_but_calc_wrong = 0

    for it in items:
        for s in it.get("steps", []):
            label = s.get("label")
            v = verify_step(s.get("step", ""))
            if not v:
                continue
            step_calc = len(v)
            step_ok = sum(1 for r in v if r.get("ok"))
            total_calcs += step_calc
            correct_calcs += step_ok
            if label == 1:
                judged_correct_with_calc += step_calc
                judge_correct_but_calc_wrong += (step_calc - step_ok)

    return {
        "trajectories": len(items),
        "total_calcs": total_calcs,
        "calc_accuracy": round(correct_calcs / total_calcs * 100, 2) if total_calcs else None,
        "judge_labeled_correct_calcs": judged_correct_with_calc,
        "judge_correct_but_calc_wrong": judge_correct_but_calc_wrong,
        "calc_error_rate_among_judge_correct": round(
            judge_correct_but_calc_wrong / judged_correct_with_calc * 100, 2
        ) if judged_correct_with_calc else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Fin-R1 计算步骤规则引擎校验")
    parser.add_argument("--step", default=None, help="校验单个步骤文本")
    parser.add_argument("--analyze", default=None, help="扫描 PRM 数据统计计算错误")
    parser.add_argument("--max_trajectories", type=int, default=2000, help="分析时最多处理轨迹数")
    args = parser.parse_args()

    if args.step:
        results = verify_step(args.step)
        for r in results:
            status = "✅" if r.get("ok") else ("❌" if r.get("ok") is False else "⚠️")
            print(f"{status} {r['expr']} = {r['computed']}（声称 {r['claimed']}）")
        if not results:
            print("未提取到算式（文本里没有 a op b = result 结构）")
        return

    if args.analyze:
        stats = analyze_prm(args.analyze, args.max_trajectories)
        print(f"\n=== 计算步骤规则引擎校验（{stats['trajectories']} 条轨迹）===")
        print(f"可验算式总数: {stats['total_calcs']}")
        print(f"算式正确率: {stats['calc_accuracy']}%")
        print(f"被 LLM-Judge 标'正确'且含算式的步骤: {stats['judge_labeled_correct_calcs']}")
        print(f"其中数值算错（Judge 漏判）: {stats['judge_correct_but_calc_wrong']}")
        print(f"Judge 标'正确'但数值错的占比: {stats['calc_error_rate_among_judge_correct']}%")
        print("\n结论：这就是规则引擎能发现、而 LLM-Judge 会漏掉的数值错误——金融场景独有的可验证性。")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
