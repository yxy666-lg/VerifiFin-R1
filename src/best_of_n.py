#!/usr/bin/env python3
"""
Fin-R1 Best-of-N + PRM 重排实验

每个问题生成 N 个回答（采样），用 PRM 打分，选分数最高的回答。
对比「单次生成（贪心）」vs「Best-of-N + PRM 重排」的答案准确率。
目的：独立验证 PRM 的价值（不依赖 GRPO）。

关键改进（v2）：
  1. 修复 reward 调用顺序 bug：位置参数约定是 (prompts, completions)，prompts 在前。
     v1 传成 (completions, prompts)，PRM 拿题目当回答打分，全部恒为 0.5。
  2. 断点续跑：每题结果实时写入 <output>.progress.jsonl，中断后重跑自动跳过已做过的题。
  3. 批量 PRM 打分：所有候选的所有步骤合并成一次前向，替代逐候选调用。

用法（在云端干净环境 grpo_clean）：
  python src/best_of_n.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --model results/grpo/final \
    --prm_model results/prm/final \
    --test_file data/eval/finance_reasoning.jsonl \
    --n 8 \
    --output results/best_of_n.json

提速建议（先小规模验证再跑满）：
  --max_questions 20 --n 4 --max_new_tokens 256   # 快速冒烟
  跑满：--n 8 --max_new_tokens 512（每题约 700s，48 题约 10h，可断点续跑）
"""

import argparse
import json
import os
import re
import random
import sys
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import check_answer  # noqa: E402
from reward_finance import FinanceReward  # noqa: E402


def load_model(model_path, base_model):
    """加载模型：支持 LoRA adapter 或完整模型"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        from peft import PeftModel
        print(f"[BoN] 检测到 LoRA adapter，基座: {base_model}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
            device_map="auto",  # 关键：不加会加载到 CPU，7B 生成慢到像卡死
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            device_map="auto",
        )
    model.eval()
    return model, tokenizer


def load_prm(base_model, prm_dir):
    """加载 PRM 打分器（LoRA 三分类器），4bit 省显存"""
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=3, torch_dtype=torch.bfloat16,
        trust_remote_code=True, load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, prm_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate_n(model, tokenizer, prompt, n, max_new_tokens, temperature=0.8):
    """一次生成 n 个回答（采样）；n=1 时贪心"""
    device = next(model.parameters()).device  # device_map="auto" 时 model.device 可能不可靠
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=(n > 1),
            temperature=temperature if n > 1 else None,
            num_return_sequences=n,
        )
    texts = []
    for i in range(n):
        texts.append(tokenizer.decode(outputs[i][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip())
    return texts


def batch_prm_score(prm_model, prm_tokenizer, question, responses):
    """批量给多个回答打分：所有候选的所有步骤合并成一次前向，返回每个回答的平均正确概率。
    使用 (问题+历史前缀+当前步骤) 打 PRM，不再走 FinanceReward（避开其 (prompts, completions)
    传参约定，也省掉硬约束等无关逻辑）。"""
    step_texts, resp_indices = [], []
    for i, resp in enumerate(responses):
        steps = re.findall(r"第\d+步[：:]\s*(.*?)(?=第\d+步[：:]|\Z)", resp, re.DOTALL)
        prefix = []
        for s in (s.strip() for s in steps if s.strip()):
            history = "\n".join(prefix) if prefix else "（无）"
            step_texts.append(f"问题：{question}\n\n此前推理：\n{history}\n\n待评估步骤：\n{s}")
            resp_indices.append(i)
            prefix.append(s)
    if not step_texts:
        return [0.5] * len(responses)

    device = next(iter(prm_model.parameters())).device
    enc = prm_tokenizer(
        step_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        logits = prm_model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[:, 2].cpu().tolist()  # label 2 = 正确

    grouped = defaultdict(list)
    for idx, p in zip(resp_indices, probs):
        grouped[idx].append(p)
    scores = []
    for i in range(len(responses)):
        values = sorted(grouped[i])
        k = max(1, (len(values) + 1) // 2)
        scores.append(sum(values[:k]) / k if values else 0.5)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Fin-R1 Best-of-N + PRM 重排")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model", required=True, help="模型路径（GRPO final 或 SFT）")
    parser.add_argument("--prm_model", required=True, help="PRM 模型路径")
    parser.add_argument("--test_file", default="data/eval/finance_reasoning.jsonl", help="评测文件")
    parser.add_argument("--n", type=int, default=8, help="每个问题采样数")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="生成长度")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_questions", type=int, default=100, help="最多处理题数")
    parser.add_argument("--output", default="results/best_of_n.json")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer = load_model(args.model, args.base_model)
    prm_model, prm_tokenizer = load_prm(args.base_model, args.prm_model)

    items = []
    with open(args.test_file, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line.strip()))
    items = items[:args.max_questions]
    print(f"[BoN] {len(items)} 题，N={args.n}，max_new_tokens={args.max_new_tokens}")

    # 断点续跑：已完成的题写进 progress 文件，重跑时跳过
    progress_path = args.output.replace(".json", ".progress.jsonl")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    done_qs = set()
    if os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_qs.add(json.loads(line.strip())["question"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"[BoN] 断点续跑：跳过已完成的 {len(done_qs)} 题")

    records = []
    if os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    pf = open(progress_path, "a", encoding="utf-8")

    try:
        for item in tqdm(items, desc="Best-of-N"):
            question = item["question"]
            gold = str(item.get("answer", "")).strip()
            # 只对可量化答案计分（数字/选项字母），与 evaluate.py 一致
            is_quant = bool(re.search(r"\d", gold)) or bool(re.fullmatch(r"[A-E]+", gold.upper()))
            if not gold or not is_quant:
                continue
            if question in done_qs:
                continue

            prompt = (f"{question}\n\n请对以下金融问题进行详细逐步推理，每一步用「第N步：」开头，"
                      f"涉及计算写出完整计算式，最后给出明确结论。")

            # 基线：单次贪心
            base = generate_n(model, tokenizer, prompt, 1, args.max_new_tokens)[0]
            # Best-of-N：采样 N 个，PRM 打分取最高
            cands = generate_n(model, tokenizer, prompt, args.n, args.max_new_tokens, args.temperature)
            scores = batch_prm_score(prm_model, prm_tokenizer, question, cands)
            best = cands[int(max(range(len(scores)), key=lambda i: scores[i]))]
            sampled = cands[0]
            random_pick = cands[random.randrange(len(cands))]

            base_ok = check_answer(gold, base)
            bon_ok = check_answer(gold, best)
            sampled_ok = check_answer(gold, sampled)
            random_ok = check_answer(gold, random_pick)
            candidate_correct = [check_answer(gold, c) for c in cands]

            # 每题落盘（断点续跑 + 可看单题对错）
            record = {
                "question": question,
                "gold": gold,
                "baseline_correct": bool(base_ok),
                "sampled_first_correct": bool(sampled_ok),
                "random_correct": bool(random_ok),
                "bon_correct": bool(bon_ok),
                "oracle_correct": any(candidate_correct),
                "scores": [round(float(s), 4) for s in scores],
            }
            pf.write(json.dumps(record, ensure_ascii=False) + "\n")
            pf.flush()
            done_qs.add(question)
            records.append(record)
    finally:
        pf.close()

    scored = len(records)
    if scored == 0:
        raise SystemExit("没有可量化的题，无法评测")

    def accuracy(key):
        return round(sum(bool(r.get(key)) for r in records) / scored * 100, 2)

    results = {
        "method": f"Best-of-{args.n} + PRM 重排",
        "questions": scored,
        "greedy_accuracy": accuracy("baseline_correct"),
        "sampled_first_accuracy": accuracy("sampled_first_correct"),
        "random_rerank_accuracy": accuracy("random_correct"),
        "prm_rerank_accuracy": accuracy("bon_correct"),
        "oracle_pass_at_n": accuracy("oracle_correct"),
    }
    results["prm_vs_random_delta_pp"] = round(
        results["prm_rerank_accuracy"] - results["random_rerank_accuracy"], 2
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[BoN] 结果（{scored} 题，可量化答案）:")
    print(f"  贪心: {results['greedy_accuracy']}% | 首个采样: {results['sampled_first_accuracy']}%")
    print(f"  随机重排: {results['random_rerank_accuracy']}% | PRM重排: {results['prm_rerank_accuracy']}%")
    print(f"  Oracle pass@{args.n}: {results['oracle_pass_at_n']}%")
    print(f"  结果保存: {args.output}")
    print(f"  单题明细: {progress_path}")


if __name__ == "__main__":
    main()
