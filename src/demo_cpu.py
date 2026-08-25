#!/usr/bin/env python3
"""无需GPU的面试演示：展示数据规模、规则验证和冲突审计摘要。"""

import json
import os

from calc_verifier import verify_step


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    with open(os.path.join(ROOT, path), encoding="utf-8-sig") as f:
        return json.load(f)


def main():
    verified = load("data/prm/fin_prm_verified_stats.json")
    conflicts = load("reports/conflict_summary.json")
    assisted = load("reports/assisted_review_50_summary.json")
    example = "净利润 = 5000 × 15% = 750 万元"
    print("=== VerifiFin-R1 CPU Demo ===")
    print(f"审计轨迹/步骤: {verified['trajectories']} / {verified['total_steps']}")
    print(f"可验证步骤: {verified['verifyable_steps']} ({verified['verifyable_rate']}%)")
    print(f"规则—Judge冲突: {conflicts['disagreements']}")
    print(f"模型辅助初审: {assisted['selected']}，仍需专业复核: {assisted['expert_review_required']}")
    print(f"示例算式: {example}")
    print(json.dumps(verify_step(example), ensure_ascii=False, indent=2))
    print("注意：规则只证明局部算术，不证明整步金融逻辑。")


if __name__ == "__main__":
    main()
