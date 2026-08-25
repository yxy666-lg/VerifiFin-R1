#!/usr/bin/env python3
"""
Fin-R1 PRM 标注质量抽检脚本

读取 data/prm/finance_prm.jsonl（data_prep_prm.py --mode annotate 的产物），输出：
  1. 数据量统计（总数 / 有答案 / 无答案）
  2. 步骤标签分布（+1 / 0 / -1 占比）
  3. 每条轨迹的步骤数分布
  4. judge 置信度分布
  5. 异常检查：
     - 全 +1 的轨迹（可能"结论错但全对"）
     - 全 -1 的轨迹（可能是垃圾轨迹）
     - 「结论对不上标准答案但步骤全 +1」——judge 漏网毒数据（重点）
     - 无步骤 / 空步骤的轨迹
  6. 抽样打印若干条（默认 10 条）供人工查看

用法：
  python src/check_prm_quality.py --data data/prm/finance_prm.jsonl --sample 10
  python src/check_prm_quality.py --data data/prm/finance_prm.jsonl --suspicious_only
"""

import argparse
import json
import re
from collections import Counter

LABEL_NAMES = {1: "正确(+1)", 0: "部分(0)", -1: "错误(-1)"}


def answer_matches_text(answer: str, text: str) -> bool:
    """判断标准答案是否出现在文本里（字母/数字整词匹配，粗略用于异常检测）"""
    answer = str(answer).strip()
    if not answer:
        return True  # 无答案不判
    # 字母答案（单选 A 或多选 ABC）：每个答案字母都应在文本里出现（容错"、"/空格分隔）
    m = re.fullmatch(r"[A-E]+", answer)
    if m:
        return all(
            re.search(r"(?<![A-Z])" + ch + r"(?![A-Z])", text)
            for ch in m.group()
        )
    # 数字答案：任一数字整词匹配即可
    for num in re.finditer(r"-?\d+(?:\.\d+)?", answer):
        if re.search(r"(?<!\d)" + re.escape(num.group()) + r"(?!\d)", text):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="PRM 标注质量抽检")
    parser.add_argument("--data", default="data/prm/finance_prm.jsonl", help="标注结果文件")
    parser.add_argument("--sample", type=int, default=10, help="抽样打印条数")
    parser.add_argument("--suspicious_only", action="store_true",
                        help="只打印可疑轨迹（结论对不上答案但步骤全对）")
    args = parser.parse_args()

    items = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    if not items:
        print("空文件，没有数据")
        return

    n = len(items)
    with_ans = sum(1 for it in items if it.get("answer"))
    print(f"═══ 基本统计 ═══")
    print(f"总轨迹: {n} 条 | 有标准答案: {with_ans} | 无答案: {n - with_ans}")

    # 标签 / 步骤数 / 置信度
    label_counter = Counter()
    confs = []
    steps_per_traj = []
    for it in items:
        steps = it.get("steps", [])
        steps_per_traj.append(len(steps))
        for s in steps:
            label_counter[s.get("label")] += 1
            confs.append(s.get("confidence", 0))

    total_steps = sum(steps_per_traj)
    print(f"\n═══ 标签分布（共 {total_steps} 步）═══")
    for lb in (1, 0, -1):
        cnt = label_counter.get(lb, 0)
        pct = cnt / total_steps * 100 if total_steps else 0
        print(f"  {LABEL_NAMES.get(lb, lb)}: {cnt} 步 ({pct:.1f}%)")

    print(f"\n═══ 步骤数分布 ═══")
    step_dist = Counter(steps_per_traj)
    for k in sorted(step_dist):
        print(f"  {k} 步: {step_dist[k]} 条")

    if confs:
        avg_conf = sum(confs) / len(confs)
        low_conf = sum(1 for c in confs if c < 0.5)
        print(f"\n═══ judge 置信度 ═══")
        print(f"  平均: {avg_conf:.3f} | <0.5 的步骤: {low_conf} ({low_conf/len(confs)*100:.1f}%)")

    # 异常检查
    print(f"\n═══ 异常检查 ═══")
    no_steps = [it for it in items if not it.get("steps")]
    all_pos = [it for it in items if it.get("steps") and all(s.get("label") == 1 for s in it["steps"])]
    all_neg = [it for it in items if it.get("steps") and all(s.get("label") == -1 for s in it["steps"])]

    print(f"  无步骤轨迹: {len(no_steps)} 条")
    print(f"  全 +1 轨迹: {len(all_pos)} 条 ({len(all_pos)/n*100:.1f}%)  ← 若过高说明 judge 太宽松")
    print(f"  全 -1 轨迹: {len(all_neg)} 条 ({len(all_neg)/n*100:.1f}%)  ← 可能是垃圾轨迹")

    # 结论对不上答案但全 +1（重点）
    suspicious = []
    for it in all_pos:
        ans = it.get("answer", "")
        final = it.get("final_answer", "") + it.get("trajectory", "")
        if ans and not answer_matches_text(ans, final):
            suspicious.append(it)
    print(f"  ⚠️ 结论对不上标准答案但全 +1: {len(suspicious)} 条 ({len(suspicious)/max(n,1)*100:.1f}%)"
          f" ← judge 漏网毒数据，建议人工复核或丢弃")

    # 抽样打印
    if args.suspicious_only:
        show = suspicious
        if not show:
            print("\n无可疑轨迹，很好。")
            return
    else:
        show = items[: args.sample]

    print(f"\n═══ 抽样 {len(show)} 条 ═══")
    for i, it in enumerate(show[: args.sample]):
        print(f"\n--- 样本 {i+1} ---")
        print(f"Q: {it['question'][:80]}")
        if it.get("answer"):
            print(f"标准答案: {it['answer']}")
        print(f"难度: {it.get('difficulty', '?')}")
        for s in it.get("steps", []):
            print(f"  [{LABEL_NAMES.get(s.get('label'), '?')} (conf={s.get('confidence', 0):.2f})] {s.get('step', '')[:60]}")


if __name__ == "__main__":
    main()
