#!/usr/bin/env python3
"""
Fin-R1 自我改进数据生成（升级2：Expert Iteration / ReST 思路）

作用：
  用训练好的模型对 GRPO prompt 采样 → PRM 给每一步打分 → 保留 top N% 高质量轨迹
  导出为补充 SFT 数据，再做一轮训练（= 标准 GRPO 之后的自我改进对比实验）

用法：
  # ① 先用 grpo_train.py 跑标准 GRPO，得到 results/grpo/final
  # ② 生成 + 筛选自我改进数据
  python src/self_improve.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --model results/grpo/final \
    --prm_model results/prm/final \
    --prompt_file data/grpo/prompts.jsonl \
    --top_ratio 0.5 \
    --output data/self_improve/augmented.jsonl
  # ③ 用 augmented.jsonl 做几轮 SFT（LoRA 继续），再用新模型跑第二轮 GRPO

注意：
  - 本脚本只做"生成+打分+筛选+导出"，不重复训练。
  - 建议先跑标准 GRPO 出基线，再跑本流程做对比（简历上的 ablation）。
"""

import argparse
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path, base_model):
    """加载模型：支持完整模型或 LoRA adapter"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        from peft import PeftModel
        print(f"[SelfImprove] 检测到 LoRA adapter，基座: {base_model}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    model.eval()
    return model, tokenizer


def load_prm(base_model, prm_dir):
    """加载 PRM 打分器（LoRA 三分类器）"""
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=3, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, prm_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def prm_score(prm_model, prm_tokenizer, question, response):
    """上下文 PRM bottom-k 聚合，避免冗余正确步骤稀释关键错误。"""
    steps = re.findall(r"第\d+步[：:]\s*(.*?)(?=第\d+步[：:]|\Z)", response, re.DOTALL)
    steps = [s.strip() for s in steps if s.strip()]
    if not steps:
        return 0.5
    probs, prefix = [], []
    for s in steps:
        history = "\n".join(prefix) if prefix else "（无）"
        text = f"问题：{question}\n\n此前推理：\n{history}\n\n待评估步骤：\n{s}"
        enc = prm_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(prm_model.device)
        with torch.no_grad():
            logits = prm_model(**enc).logits[0]
        probs.append(torch.softmax(logits, dim=-1)[2].item())  # label 2 = 正确
        prefix.append(s)
    k = max(1, (len(probs) + 1) // 2)
    return sum(sorted(probs)[:k]) / k


def main():
    parser = argparse.ArgumentParser(description="Fin-R1 自我改进数据生成")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model", required=True, help="第一轮 GRPO 模型路径")
    parser.add_argument("--prm_model", required=True, help="PRM 模型路径")
    parser.add_argument("--prompt_file", default="data/grpo/prompts.jsonl", help="GRPO prompt 集")
    parser.add_argument("--top_ratio", type=float, default=0.5, help="保留分数最高的比例")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="采样长度")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_prompts", type=int, default=2000, help="最多处理多少条 prompt")
    parser.add_argument("--output", default="data/self_improve/augmented.jsonl")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.base_model)
    prm_model, prm_tokenizer = load_prm(args.base_model, args.prm_model)

    prompts = []
    with open(args.prompt_file, encoding="utf-8") as f:
        for line in f:
            it = json.loads(line.strip())
            prompts.append(it["prompt"])
    prompts = prompts[: args.max_prompts]
    print(f"[SelfImprove] {len(prompts)} 条 prompt，top_ratio={args.top_ratio}")

    scored = []
    for q in tqdm(prompts, desc="生成+打分"):
        try:
            inputs = tokenizer(q, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=True, temperature=args.temperature, top_p=0.95,
                )
            resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            if not resp:
                continue
            score = prm_score(prm_model, prm_tokenizer, q, resp)
            scored.append({"prompt": q, "response": resp, "score": score})
        except Exception as e:
            print(f"  错误: {e}")

    if not scored:
        raise SystemExit("没有生成到任何有效样本")

    scored.sort(key=lambda x: x["score"], reverse=True)
    keep_n = max(1, int(len(scored) * args.top_ratio))
    kept = scored[:keep_n]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for it in kept:
            entry = {
                "messages": [
                    {"role": "user", "content": it["prompt"]},
                    {"role": "assistant", "content": it["response"]},
                ],
                "prm_score": it["score"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    scores = [s["score"] for s in scored]
    print(f"\n[SelfImprove] 生成 {len(scored)} 条，保留 top {keep_n} 条")
    print(f"  分数范围: {min(scores):.3f} ~ {max(scores):.3f}，平均 {sum(scores)/len(scores):.3f}")
    print(f"  输出: {args.output}")
    print("  下一步：用这个文件做几轮 SFT，再用新模型跑第二轮 GRPO")


if __name__ == "__main__":
    main()
