#!/usr/bin/env python3
"""
Fin-R1 GRPO Reward 函数

复合 Reward = 答案正确性 + 上下文 PRM/Judge + 规则验算 + 低权重格式 - 投机惩罚

面试亮点：
  1. 硬约束防止推理格式崩塌（不输出步骤编号、不写计算式直接扣分）
  2. 软约束评估语义质量（金融逻辑是否正确、结论是否有据可查）
  3. 软约束可接入训好的 PRM 模型（本地、快、无 API 成本），也可用 LLM-as-a-Judge
  4. 复合分归一化到 [0,1]，硬约束权重与软约束同级，真正发挥作用

接口说明：
  __call__ 兼容 TRL / Unsloth GRPOTrainer 的两种调用方式：
    reward_fn(completions, prompts=..., **kwargs)
    reward_fn(prompts, completions, **kwargs)
  二者都会把问题（prompt）通过 kwargs 传进来，本类会自动提取。

使用方式（配合 TRL GRPOTrainer，见 src/grpo_train.py）：
  from reward_finance import FinanceReward
  reward_fn = FinanceReward(enable_soft_judge=True, answer_lookup=answers)
  # TRL 会自动调用 reward_fn(completions=..., **kwargs)
"""

import json
import os
import re
from difflib import SequenceMatcher
from typing import Optional


# ---------- 硬约束规则 ----------
# 注意：所有硬约束针对"带推理过程的长回答"，纯字母答案（如"选A"）会全部不通过，
# 这是有意为之——GRPO 的 prompt 都是推理题，我们希望模型先推理再给答案。
HARD_CONSTRAINTS = [
    {
        "name": "步骤编号",
        "pattern": r"第\d+步[：:]",
        "weight": 0.15,
        "desc": "推理过程必须用「第N步」编号",
    },
    {
        "name": "计算过程",
        "pattern": r"(?:\d+\s*[+\-×÷*/]\s*\d+|=\s*-?\d+(?:\.\d+)?)",
        "weight": 0.15,
        "desc": "必须出现带运算符的计算式（如 5000×15%）或等号结果",
    },
    {
        "name": "计算单位",
        "pattern": r"\d+(?:\.\d+)?\s*(万元|亿元|元|万|亿|%|％|倍|股|点|吨|个)",
        "weight": 0.10,
        "desc": "计算结果需带单位（元/万元/%/倍等）",
    },
    {
        "name": "最终结论",
        "pattern": r"(综上|综上所述|最终结论|结论|答案是|最终)",
        "weight": 0.10,
        "desc": "必须有明确的总结/结论",
    },
    {
        "name": "引用来源",
        "pattern": r"(根据|依据|按|参考|规定|准则|公告)",
        "weight": 0.05,
        "desc": "涉及判断时应引用具体准则或数据来源",
    },
]

# 硬约束满分（用于归一化）
MAX_HARD_SCORE = sum(c["weight"] for c in HARD_CONSTRAINTS)

# ---------- LLM-as-a-Judge 评估 Prompt ----------
SOFT_JUDGE_PROMPT = """你是一位金融专家评审。请对以下回答的金融专业质量打分（0-100 分）。

评分维度（各 25 分）：
1. 金融逻辑：推理逻辑是否严密，是否遵循金融学基本原理
2. 计算准确性：如有计算，过程和结果是否正确（有标准答案时以标准答案为准）
3. 专业深度：是否使用了专业的金融术语和分析框架
4. 实用性：结论是否具有实际指导意义

问题：{question}
回答：{response}
{answer_section}
请输出 JSON：
{{"finance_logic": <0-25>, "calc_accuracy": <0-25>, "professional_depth": <0-25>, "practicality": <0-25>, "overall": "<一句话总结>"}}

只输出 JSON，不需要其他文字。"""


class FinanceReward:
    """金融 GRPO 复合 Reward 函数"""

    def __init__(
        self,
        hard_weight: float = 0.10,
        soft_weight: float = 0.55,
        answer_weight: float = 0.25,
        verifier_weight: float = 0.10,
        hacking_penalty_weight: float = 0.15,
        api_key: Optional[str] = None,
        api_base: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",  # 评分用便宜模型
        enable_soft_judge: bool = True,
        answer_lookup: Optional[dict] = None,
        prm_model=None,
        prm_tokenizer=None,
        debug: bool = False,
    ):
        total = hard_weight + soft_weight + answer_weight + verifier_weight
        if total <= 0:
            raise ValueError("reward 权重之和必须大于 0")
        hard_weight, soft_weight = hard_weight / total, soft_weight / total
        answer_weight, verifier_weight = answer_weight / total, verifier_weight / total

        self.hard_weight = hard_weight
        self.soft_weight = soft_weight
        self.answer_weight = answer_weight
        self.verifier_weight = verifier_weight
        self.hacking_penalty_weight = hacking_penalty_weight
        self.enable_soft_judge = enable_soft_judge
        self.debug = debug
        self.answer_lookup = answer_lookup or {}

        # 软约束后端：优先 PRM 模型（本地），否则 LLM-as-a-Judge
        self.prm_model = prm_model
        self.prm_tokenizer = prm_tokenizer
        # TRL 的 GRPOTrainer 需要 reward func 有 __name__（用于记录打分器名字）
        self.__name__ = "finance_reward"
        # judge client 惰性创建（本地没装 openai 也能 import 本模块 / 跑硬约束自测）
        self._api_key = api_key
        self._api_base = api_base
        self.model = model
        self.client = None

    # ---------- 软约束后端选择 ----------
    def _soft(self, question: str, response: str):
        """返回 (score 0~1, detail)"""
        if self.prm_model is not None:
            return self._prm_soft_reward(question, response)
        if not self.enable_soft_judge:
            return 0.5, {"note": "soft judge disabled"}
        return self._judge_soft_reward(question, response)

    def _prm_soft_reward(self, question: str, response: str) -> tuple[float, dict]:
        """用训好的 PRM（LoRA 三分类：0=错误 1=部分 2=正确）给每步打分。
        优化：一个回答的所有步骤合并成一次批量前向，避免每步一次前向（GRPO 里慢 5-6 倍）。"""
        import torch
        steps = extract_reasoning_steps(response)
        if not steps:
            return 0.5, {"note": "no steps extracted"}

        # 4bit/accelerate 派发下 model.device 可能不稳，用第一个参数取设备最稳
        device = next(iter(self.prm_model.parameters())).device
        prm_question = normalize_question(question)
        texts, prefix = [], []
        for step in steps:
            history = "\n".join(prefix) if prefix else "（无）"
            texts.append(f"问题：{prm_question}\n\n此前推理：\n{history}\n\n待评估步骤：\n{step}")
            prefix.append(step)
        enc = self.prm_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            logits = self.prm_model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 2].float().cpu().tolist()
        # bottom-k 聚合聚焦薄弱步骤，避免堆砌简单正确步骤稀释关键错误。
        k = max(1, (len(probs) + 1) // 2)
        score = sum(sorted(probs)[:k]) / k
        return float(score), {"backend": "prm", "steps": len(steps), "bottom_k": k}

    def _judge_soft_reward(self, question: str, response: str) -> tuple[float, dict]:
        """LLM-as-a-Judge 打分（带可选标准答案）"""
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self._api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=self._api_base,
            )

        answer = self._lookup_answer(question)
        answer_section = f"\n标准答案：{answer}" if answer else ""
        prompt = SOFT_JUDGE_PROMPT.format(
            question=question, response=response, answer_section=answer_section
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512,
            )
            raw = res.choices[0].message.content

            # 提取 JSON
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if json_match:
                judge_result = json.loads(json_match.group())
            else:
                judge_result = {}

            # 归一化到 0-1
            finance_logic = judge_result.get("finance_logic", 12) / 25.0
            calc_accuracy = judge_result.get("calc_accuracy", 12) / 25.0
            professional_depth = judge_result.get("professional_depth", 12) / 25.0
            practicality = judge_result.get("practicality", 12) / 25.0

            soft_score = (finance_logic + calc_accuracy + professional_depth + practicality) / 4.0
            return soft_score, judge_result

        except Exception as e:
            print(f"[SoftReward] API 错误: {e}")
            return 0.5, {"error": str(e)}

    def _lookup_answer(self, question: str) -> str:
        """从 answer_lookup 里按 prompt 找标准答案（精确优先，前缀兜底）"""
        if question in self.answer_lookup:
            return self.answer_lookup[question]
        q50 = question[:50]
        for k, v in self.answer_lookup.items():
            if k[:50] == q50:
                return v
        return ""

    # ---------- 硬约束 ----------
    def hard_reward(self, response: str) -> tuple[float, dict]:
        """
        硬约束打分（格式检查），返回归一化到 [0,1] 的分数
        """
        scores = {}
        total = 0.0

        for constraint in HARD_CONSTRAINTS:
            match = re.search(constraint["pattern"], response)
            score = constraint["weight"] if match else 0.0
            scores[constraint["name"]] = {
                "score": score,
                "passed": bool(match),
                "desc": constraint["desc"],
            }
            total += score

        # 归一化：硬约束贡献 ∈ [0,1]
        normalized = total / MAX_HARD_SCORE
        return normalized, scores

    def answer_reward(self, question: str, response: str) -> tuple[float, dict]:
        """有标准答案时严格校验；无标准答案时不伪造监督，返回中性分。"""
        gold = self._lookup_answer(question)
        if not gold:
            return 0.5, {"available": False}
        gold_up = str(gold).strip().upper()
        if re.fullmatch(r"[A-E]+", gold_up):
            matches = re.findall(r"(?:答案|选择|选项|结论)[^A-E]{0,8}([A-E]+)", response.upper())
            correct = bool(matches) and matches[-1] == gold_up
        else:
            gold_numbers = re.findall(r"-?\d+(?:\.\d+)?", str(gold))
            correct = bool(gold_numbers) and all(
                re.search(r"(?<!\d)" + re.escape(n) + r"(?!\d)", response) for n in gold_numbers
            )
        return float(correct), {"available": True, "correct": correct}

    @staticmethod
    def verifier_reward(response: str) -> tuple[float, dict]:
        """只在可验算步骤上给确定性分；无算式时返回中性分。"""
        try:
            from calc_verifier import verify_response_steps
            result = verify_response_steps(extract_reasoning_steps(response))
            total = result.get("total_calcs", 0)
            if not total:
                return 0.5, {"available": False}
            score = result.get("correct_calcs", 0) / total
            return float(score), {"available": True, **result}
        except Exception as exc:
            return 0.5, {"available": False, "error": str(exc)}

    @staticmethod
    def hacking_penalty(response: str) -> tuple[float, dict]:
        """惩罚重复步骤、异常冗长和多次互相冲突的最终答案。"""
        steps = extract_reasoning_steps(response)
        duplicate_pairs = 0
        for i, left in enumerate(steps):
            for right in steps[i + 1:]:
                duplicate_pairs += SequenceMatcher(None, left, right).ratio() >= 0.88
        repeated_conclusions = len(re.findall(r"(?:答案是|最终答案|最终结论)", response))
        penalty = min(1.0, duplicate_pairs * 0.25 + max(0, repeated_conclusions - 1) * 0.2)
        if len(response) > 4000:
            penalty = min(1.0, penalty + 0.25)
        return penalty, {"duplicate_pairs": duplicate_pairs, "conclusions": repeated_conclusions}

    # ---------- 对外接口（TRL / Unsloth 兼容） ----------
    def __call__(self, *args, **kwargs) -> list[float]:
        """
        兼容 TRL GRPOTrainer 的两种调用方式：
          reward_fn(completions=..., prompts=..., ...)
          reward_fn(prompts, completions, ...)
        """
        completions = kwargs.get("completions")
        if completions is None:
            if len(args) >= 2 and isinstance(args[1], list):
                # 约定：位置参数按 (prompts, completions) 传入
                completions = args[1]
            elif len(args) >= 1 and isinstance(args[0], list) and not all(isinstance(x, str) for x in args[0]):
                completions = args[0]

        questions = self._extract_prompts(kwargs, args)

        if completions is None:
            raise TypeError("无法定位 completions：请以关键字传入，或按 (prompts, completions) 传参")
        if not questions:
            raise TypeError("无法定位 prompts：请以关键字 prompts= 传入")

        # 若只有 1 个 prompt 对应多个 completions（group 采样），广播
        if len(questions) == 1 and len(completions) > 1:
            questions = questions * len(completions)
        elif len(questions) != len(completions):
            # 尽量对齐；实在对不上就截短，避免 zip 静默丢数据
            questions = (questions * (len(completions) // len(questions) + 1))[:len(completions)]

        rewards = []
        for q, resp in zip(questions, completions):
            hard_s, hard_detail = self.hard_reward(resp)
            soft_s, soft_detail = self._soft(q, resp)
            answer_s, answer_detail = self.answer_reward(q, resp)
            verifier_s, verifier_detail = self.verifier_reward(resp)
            penalty, penalty_detail = self.hacking_penalty(resp)

            total = (
                self.hard_weight * hard_s
                + self.soft_weight * soft_s
                + self.answer_weight * answer_s
                + self.verifier_weight * verifier_s
                - self.hacking_penalty_weight * penalty
            )
            total = max(0.0, min(1.0, total))

            if self.debug:
                print(f"[Reward] {total:.3f} | answer={answer_s:.3f} process={soft_s:.3f} "
                      f"verify={verifier_s:.3f} format={hard_s:.3f} penalty={penalty:.3f}")

            rewards.append(total)

        return rewards

    @staticmethod
    def _extract_prompts(kwargs: dict, args: tuple) -> list[str]:
        """从 kwargs 里提取 prompt 列表（兼容 prompts / prompt / messages / 位置参数）"""
        if "prompts" in kwargs and kwargs["prompts"]:
            return list(kwargs["prompts"])
        if "prompt" in kwargs:
            p = kwargs["prompt"]
            return [p] if isinstance(p, str) else list(p)
        if "messages" in kwargs and kwargs["messages"]:
            # messages: list of chat lists → 取每条最后一条 user 消息
            out = []
            for msgs in kwargs["messages"]:
                q = ""
                for m in reversed(msgs):
                    if m.get("role") == "user":
                        q = m.get("content", "")
                        break
                out.append(q)
            return out
        # 位置参数 (prompts, completions)
        for a in args:
            if isinstance(a, list) and len(a) > 0 and all(isinstance(x, str) for x in a):
                return list(a)
        return []


def extract_reasoning_steps(response: str) -> list[str]:
    """提取带编号步骤，不把步骤前导语或最终总结误当成独立步骤。"""
    matches = re.findall(r"第\d+步[：:]\s*(.*?)(?=第\d+步[：:]|\Z)", response, re.DOTALL)
    return [m.strip() for m in matches if m.strip()]


def normalize_question(prompt: str) -> str:
    """GRPO prompt 含格式指令时，仅保留“问题：”后的真实题干以匹配 PRM 训练分布。"""
    return prompt.rsplit("\n\n问题：", 1)[-1].strip()


# ---------- 示例：自测 ----------
if __name__ == "__main__":
    reward_fn = FinanceReward(enable_soft_judge=False, debug=True)

    test_q = "某公司年营收 5000 万，净利润率 15%，请计算净利润。"
    test_r = (
        "第1步：确定已知条件。营收 5000 万，净利润率 15%。\n"
        "第2步：计算净利润。净利润 = 营收 × 净利润率 = 5000 × 15% = 750 万\n"
        "第3步：验证。15% × 5000 = 750，计算正确。\n"
        "综上，该公司年净利润为 750 万元。"
    )
    bad_r = "选B"

    print("== 接口测试：关键字调用 ==")
    score = reward_fn(completions=[test_r, bad_r], prompts=[test_q, test_q])
    for s in score:
        print(f"Reward: {s:.3f}")

    print("== 接口测试：位置参数调用 ==")
    score2 = reward_fn([test_q, test_q], [test_r, bad_r])
    for s in score2:
        print(f"Reward: {s:.3f}")

    print("== 硬约束细节 ==")
    _, det = reward_fn.hard_reward(test_r)
    for k, v in det.items():
        print(f"  {k}: passed={v['passed']}")
