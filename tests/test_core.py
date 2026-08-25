import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reward_finance import FinanceReward, extract_reasoning_steps
from statistical_analysis import mcnemar_exact, paired_bootstrap


class RewardTests(unittest.TestCase):
    def test_extract_steps_excludes_preamble(self):
        text = "先分析。第1步：收入=100。第2步：成本=60。综上利润40。"
        steps = extract_reasoning_steps(text)
        self.assertEqual(len(steps), 2)
        self.assertFalse(steps[0].startswith("先分析"))

    def test_answer_reward(self):
        reward = FinanceReward(enable_soft_judge=False, answer_lookup={"题目": "B"})
        self.assertEqual(reward.answer_reward("题目", "最终答案是B")[0], 1.0)
        self.assertEqual(reward.answer_reward("题目", "最终答案是A")[0], 0.0)

    def test_repetition_is_penalized(self):
        response = "第1步：计算收入。第2步：计算收入。"
        self.assertGreater(FinanceReward.hacking_penalty(response)[0], 0)

    def test_statistics(self):
        rows = [{"a": False, "b": True}, {"a": True, "b": True}, {"a": True, "b": False}]
        improved, regressed, p = mcnemar_exact(rows, "a", "b")
        self.assertEqual((improved, regressed), (1, 1))
        self.assertGreaterEqual(p, 0)
        self.assertEqual(len(paired_bootstrap(rows, "a", "b", samples=100, seed=1)), 2)


if __name__ == "__main__":
    unittest.main()
