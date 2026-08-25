#!/usr/bin/env python3
"""
Fin-R1 评测脚本

功能：
  1. FinEval 标准评测（选择题准确率，用 val split，训练只用 dev，无泄漏）
  2. FinCorpus 留出集评测（跳过 SFT 用过的前 30000 条考试题，计数口径与 SFT/PRM 一致）
  3. 自研金融推理集评测（答案准确率 + 过程正确率）
  4. 与 SFT-only 模型对比，量化 GRPO 对齐的增益

使用方式：
  # FinEval 评测（支持 LoRA adapter，需 --base_model）
  python src/evaluate.py --model results/grpo/final --base_model Qwen/Qwen2.5-7B-Instruct --benchmark fineval

  # SFT vs GRPO 对比
  python src/evaluate.py --model results/grpo/final --baseline results/sft/checkpoint-xxx \
      --base_model Qwen/Qwen2.5-7B-Instruct --benchmark all

  # 金融推理过程正确率（可选 LLM judge 打分；也可换成训好的 PRM 模型）
  python src/evaluate.py --model results/grpo/final --base_model Qwen/Qwen2.5-7B-Instruct \
      --benchmark reasoning --judge_model gpt-4o-mini --judge_api_key xxx
"""

import argparse
import glob
import gzip
import json
import os
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------- FinEval 评测 ----------
def evaluate_fineval(model, tokenizer, split: str = "val", limit: int = None) -> dict:
    """
    FinEval 选择题准确率评测（val split）
    训练只用 dev，评测用 val，二者无重叠，数字可信。
    """
    print(f"[FinEval] 加载评测集（split={split}）...")
    import huggingface_hub
    zip_path = huggingface_hub.hf_hub_download(
        repo_id="SUFE-AIFLM-Lab/FinEval",
        filename="FinEval.zip",
        repo_type="dataset",
    )

    correct = 0
    total = 0
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})

    with zipfile.ZipFile(zip_path) as z:
        split_csvs = [n for n in z.namelist() if n.startswith(f"{split}/") and n.endswith(".csv")]
        print(f"[FinEval] {len(split_csvs)} 个 {split} CSV")

        for name in split_csvs:
            with z.open(name) as f:
                df = pd.read_csv(f)
            if "Unnamed: 7" in df.columns:
                df = df.rename(columns={"Unnamed: 7": "explanation"})

            # 用文件名取科目，避免依赖 zip 目录层级
            category = Path(name).stem.replace(f"_{split}", "")

            rows = list(df.iterrows())
            for _, row in tqdm(rows, desc=f"FinEval-{category}", leave=False):
                question = str(row.get("question", "")).strip()
                answer = str(row.get("answer", "")).strip()
                if not question or not answer:
                    continue

                options = []
                for opt in ["A", "B", "C", "D"]:
                    val = row.get(opt, "")
                    if pd.notna(val) and str(val).strip():
                        options.append(f"{opt}. {str(val).strip()}")
                if options:
                    question = question + "\n" + "\n".join(options)

                prompt = f"{question}\n\n请直接给出答案选项（A/B/C/D），不需要解释。"
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

                pred = extract_option(response)
                if check_answer(answer, pred):
                    correct += 1
                    by_category[category]["correct"] += 1
                total += 1
                by_category[category]["total"] += 1

                if limit and total >= limit:
                    break
            if limit and total >= limit:
                break

    acc = correct / total if total > 0 else 0
    results = {
        "benchmark": f"FinEval-{split}",
        "total": total,
        "correct": correct,
        "accuracy": round(acc * 100, 2),
        "by_category": {
            cat: {
                "accuracy": round(d["correct"] / d["total"] * 100, 2) if d["total"] else 0,
                "samples": d["total"],
            }
            for cat, d in sorted(by_category.items())
        },
    }
    return results


def evaluate_fincorpus_heldout(
    model, tokenizer, limit: int = 500, fincorpus_file: str = None,
    judge_model: str = None, judge_api_key: str = None,
    judge_api_base: str = "https://api.deepseek.com/v1",
) -> dict:
    """
    FinCorpus 留出集评测（无数据泄露）
    SFT 用了 fin_exam 前 30000 条考试题，这里用第 30000 条之后的题做评测。
    计数口径与 data_prep_sft / data_prep_prm 保持一致（只数"有答案标记"的考试题）。
    fincorpus_file：直接指定 fin_exam.jsonl.gz 路径（缓存缺失时用）。

    输出三组指标：
      1. accuracy           端到端准确率（对比 SFT 基线）
      2. step_format_rate   步骤格式合规率（检验是否任何输入都强行套模板）
      3. process_accuracy   过程正确率（judge 逐步骤核对标准答案，可选）
    让模型先输出逐步推理再给答案，才能同时测格式与过程。
    """
    print("[FinCorpus] 加载留出评测集...")

    gz_files = []
    if fincorpus_file and os.path.exists(fincorpus_file):
        gz_files = [fincorpus_file]
        print(f"  使用指定文件: {fincorpus_file}")
    else:
        cache_pattern = os.path.expanduser(
            "~/.cache/huggingface/hub/datasets--Duxiaoman-DI--FinCorpus/snapshots/*/data/fin_exam.jsonl.gz"
        )
        gz_files = glob.glob(cache_pattern)
        if not gz_files:
            blob_pattern = os.path.expanduser("~/.cache/huggingface/hub/datasets--Duxiaoman-DI--FinCorpus/blobs/*")
            all_blobs = glob.glob(blob_pattern)
            gz_files = [f for f in all_blobs if 50 * 1024 * 1024 < os.path.getsize(f) < 200 * 1024 * 1024]

    if not gz_files:
        print("  FinCorpus 缓存未找到，跳过")
        return {"benchmark": "FinCorpus-heldout", "samples": 0, "note": "cache not found"}

    correct = 0
    total = 0
    format_ok = 0
    judge_correct_steps = 0
    judge_total_steps = 0
    skip_count = 30000  # 跳过 SFT 用过的前 30000 条考试题

    # 第一步：读取所有留出题（跳过前 30000），收集 (question, answer)
    held_out = []
    with gzip.open(gz_files[0], "rt", encoding="utf-8") as f:
        exam_seen = 0
        for line in f:
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            text = item.get("text", "")
            answer_match = re.search(r"答案[：:]\s*([A-E]+(?:[、,，]?[A-E])*)", text)
            if not answer_match:
                continue  # 非考试题不计入，与 SFT 口径一致
            exam_seen += 1
            if exam_seen <= skip_count:
                continue
            answer = "".join(re.findall(r"[A-E]", answer_match.group(1)))
            answer_pos = answer_match.start()
            question = text[:answer_pos].strip()
            held_out.append((question, answer))
            if len(held_out) >= 100000:  # 保险上限，防止异常
                break
    print(f"  留出题总数: {len(held_out)}，随机抽样 {min(limit, len(held_out))} 条")

    # 第二步：固定种子随机抽样（可复现、抗顺序偏差，替代"前 N 条"切片）
    random.seed(42)
    sampled = random.sample(held_out, min(limit, len(held_out)))

    judge_client = None
    if judge_model:
        if not judge_api_key:
            print("[警告] 提供了 --judge_model 但未提供 --judge_api_key，FinCorpus 过程正确率不计算")
        else:
            from openai import OpenAI
            judge_client = OpenAI(api_key=judge_api_key, base_url=judge_api_base)

    pbar = tqdm(total=len(sampled), desc="FinCorpus-heldout", leave=False)
    for question, answer in sampled:
        # 让模型先逐步推理再给答案，才能同时测格式与过程
        prompt = f"{question}\n\n请对以下金融问题进行逐步推理，每一步用「第N步：」开头，最后给出答案选项（A/B/C/D）。"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # 1. 端到端准确率
        pred = extract_option(response)
        if check_answer(answer, pred):
            correct += 1
        total += 1

        # 2. 步骤格式合规率（检验是否真的按"第N步"推理，还是强行套模板）
        steps = re.findall(r"第\d+步[：:](.+?)(?=第\d+步[：:]|\Z)", response, re.DOTALL)
        steps = [s.strip() for s in steps if s.strip()]
        if steps:
            format_ok += 1

        # 3. 过程正确率（judge 逐步骤核对标准答案）
        if judge_client and steps:
            try:
                judge_prompt = PROCESS_JUDGE_PROMPT.format(
                    answer=answer,
                    steps="\n".join(f"第{i+1}步：{s}" for i, s in enumerate(steps)),
                )
                res = judge_client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0.0,
                    max_tokens=512,
                )
                m = re.search(r"\[[\s\S]*\]", res.choices[0].message.content)
                if m:
                    verdicts = json.loads(m.group())
                    judge_total_steps += len(verdicts)
                    judge_correct_steps += sum(1 for v in verdicts if v.get("correct") == 1)
            except Exception as e:
                print(f"  [Judge] 错误: {e}")

        pbar.update(1)
    pbar.close()

    acc = correct / total if total > 0 else 0
    results = {
        "benchmark": "FinCorpus-heldout",
        "total": total,
        "correct": correct,
        "accuracy": round(acc * 100, 2),
        "step_format_rate": round(format_ok / total * 100, 2) if total else 0,
        "note": "前30000条考试题用于SFT训练，此评测集为未训练数据",
    }
    if judge_client and judge_total_steps:
        results["process_accuracy"] = round(judge_correct_steps / judge_total_steps * 100, 2)
    return results


# ---------- 自研金融推理评测 ----------
PROCESS_JUDGE_PROMPT = """你是一位金融推理评分专家。请逐条判断生成步骤是否正确。

【标准答案】{answer}

生成的推理步骤：
{steps}

评分要求：每一步输出 1（正确）或 0（错误），以【标准答案】为基准。
如果某步导致最终结论偏离标准答案，该步及其后续步骤都应为 0。

请输出 JSON 数组：[{{"step_index": 0, "correct": 1}}, {{"step_index": 1, "correct": 0}}, ...]
只输出 JSON，不需要其他文字。"""


def evaluate_finance_reasoning(
    model, tokenizer, test_file: str, judge_model: str = None, judge_api_key: str = None,
    judge_api_base: str = "https://api.deepseek.com/v1",
) -> dict:
    """
    自研金融推理评测
    测试集格式：data/eval/finance_reasoning.jsonl
    每行：{"question": "...", "steps": [...], "answer": "..."}

    指标：
      - answer_accuracy  答案正确率（严格匹配，非子串）
      - step_format_rate 步骤格式符合率（诚实命名：只反映"是否按第N步输出"，不代表过程正确）
      - process_accuracy 过程正确率（需 LLM judge / PRM 打分，才是真正的"每步推理正确率"）
    """
    print(f"[金融推理] 加载测试集: {test_file}")
    if not Path(test_file).exists():
        print(f"  文件不存在，跳过")
        return {"benchmark": "FinanceReasoning", "samples": 0, "note": "test file not found"}

    items = []
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line.strip()))

    format_correct = 0          # 按第N步格式输出的题目数
    format_total = 0            # 推理题总数
    judge_correct_steps = 0     # judge 判定正确的步骤数
    judge_total_steps = 0       # judge 判定的步骤总数
    answer_correct = 0
    answer_scored = 0           # 答案"可量化"的题数（含数字/选项字母）
    answer_total = len(items)   # 题总数（含无数字的主观/合规题）

    prompt_template = (
        "请对以下金融问题进行详细逐步推理，用「第N步：」开头，最后给出结论。\n\n"
        "问题：{question}"
    )

    # 可选的 LLM judge（用于真正的过程正确率）
    judge_client = None
    if judge_model:
        if not judge_api_key:
            print("[警告] 提供了 --judge_model 但未提供 --judge_api_key，过程正确率将不计算")
        else:
            from openai import OpenAI
            judge_client = OpenAI(api_key=judge_api_key, base_url=judge_api_base)

    for item in tqdm(items, desc="金融推理评测"):
        question = item["question"]
        gold_answer = str(item.get("answer", "")).strip()
        prompt = prompt_template.format(question=question)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # 提取步骤
        steps = re.findall(r"第\d+步[：:](.+?)(?=第\d+步[：:]|\Z)", response, re.DOTALL)
        steps = [s.strip() for s in steps if s.strip()]

        # 步骤格式符合率（只反映是否按格式输出，诚实命名）
        format_total += 1
        format_correct += 1 if steps else 0

        # 过程正确率（judge 逐步骤打分；这才是"每步推理正确率"）
        if judge_client and steps:
            try:
                judge_prompt = PROCESS_JUDGE_PROMPT.format(
                    answer=gold_answer if gold_answer else "（无标准答案，按金融常识判断）",
                    steps="\n".join(f"第{i+1}步：{s}" for i, s in enumerate(steps)),
                )
                res = judge_client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0.0,
                    max_tokens=512,
                )
                m = re.search(r"\[[\s\S]*\]", res.choices[0].message.content)
                if m:
                    verdicts = json.loads(m.group())
                    judge_total_steps += len(verdicts)
                    judge_correct_steps += sum(1 for v in verdicts if v.get("correct") == 1)
            except Exception as e:
                print(f"  [Judge] 错误: {e}")

        # 答案正确率：只对"可量化"答案（含数字或选项字母）严格匹配计分；
        # 主观/合规题没有数字，数字匹配必然判 0 分，会不公平拖低指标，
        # 这类题交由 process_accuracy（judge/PRM 打分）评估。
        if gold_answer:
            is_quantifiable = bool(re.search(r"\d", gold_answer)) or bool(re.fullmatch(r"[A-E]+", gold_answer.strip().upper()))
            if is_quantifiable:
                answer_scored += 1
                if check_answer(gold_answer, response):
                    answer_correct += 1

    results = {
        "benchmark": "FinanceReasoning",
        "samples": answer_total,
        "answer_accuracy": round(answer_correct / answer_scored * 100, 2) if answer_scored else 0,
        "answer_scored": answer_scored,
        "qualitative_unscored": answer_total - answer_scored,
        "step_format_rate": round(format_correct / format_total * 100, 2) if format_total else 0,
    }
    if judge_client and judge_total_steps:
        results["process_accuracy"] = round(judge_correct_steps / judge_total_steps * 100, 2)
    return results


# ---------- SFT vs GRPO 对比 ----------
def compare_models(sft_results: dict, grpo_results: dict, output_file: str):
    """生成 SFT vs GRPO 对比报告"""
    report = {
        "comparison": {
            "models": ["SFT-only", "SFT+GRPO"],
            "metrics": {},
        }
    }

    for key in sft_results:
        if key in ("benchmark", "by_category", "samples"):
            continue
        sft_val = sft_results.get(key, 0)
        grpo_val = grpo_results.get(key, 0)
        delta = round(grpo_val - sft_val, 2) if isinstance(sft_val, (int, float)) else "N/A"
        report["comparison"]["metrics"][key] = {
            "sft": sft_val,
            "grpo": grpo_val,
            "delta": delta,
        }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n对比报告保存: {output_file}")
    return report


# ---------- 工具函数 ----------
def extract_option(response: str) -> str:
    """从回答中提取选项字母（支持单选/多选如 AB）"""
    m = re.search(r"(?:选项|答案|选|选择)\s*[是为]*\s*([A-D]+)", response)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?<![A-Z])[A-D]+(?![A-Z])", response)
    if m:
        return m.group(0).upper()
    return ""


def number_contained(gold: str, response: str) -> bool:
    """数字答案做整词匹配，避免 '4400' 误判成包含在 '44000' 里"""
    for m in re.finditer(r"-?\d+(?:\.\d+)?", gold):
        num = m.group()
        if re.search(r"(?<!\d)" + re.escape(num) + r"(?!\d)", response):
            return True
    return False


def check_answer(gold: str, pred_or_response: str) -> bool:
    """答案匹配：字母答案比选项；数字答案整词匹配"""
    gold = str(gold).strip().upper()
    if re.fullmatch(r"[A-E]+", gold):
        # 训练时让模型输出"正确答案是 A"，评测时让它直接给字母，两种都兼容
        pred = pred_or_response if re.fullmatch(r"[A-E]+", pred_or_response.strip()) else extract_option(pred_or_response)
        return bool(pred) and pred == gold
    return number_contained(gold, pred_or_response)


def load_model(model_path: str, base_model: str = None):
    """
    加载模型和 tokenizer
    - 如果是 LoRA adapter（目录里有 adapter_config.json），需要先加载 base_model 再挂 adapter
    - 否则当作完整模型加载
    """
    print(f"加载模型: {model_path}")
    is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if is_adapter:
        if not base_model:
            raise ValueError(
                f"{model_path} 是 LoRA adapter，必须用 --base_model 指定基座模型。"
                "（例如 Qwen/Qwen2.5-7B-Instruct）"
            )
        print(f"  检测到 LoRA adapter，基座: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError("评测 LoRA 需要 peft 库: pip install peft") from e

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Fin-R1 评测")
    parser.add_argument("--model", required=True, help="模型路径（完整模型 或 LoRA adapter）")
    parser.add_argument("--base_model", default=None, help="基座模型路径（--model 是 LoRA adapter 时必填）")
    parser.add_argument("--baseline", default=None, help="SFT baseline 路径（用于对比）")
    parser.add_argument("--benchmark", nargs="+", default=["fineval", "fincorpus"], choices=["fineval", "fincorpus", "reasoning", "all"])
    parser.add_argument("--output_dir", default="results/grpo", help="输出目录")
    parser.add_argument("--test_file", default="data/eval/finance_reasoning.jsonl", help="自研推理测试集")
    parser.add_argument("--fincorpus_file", default=None, help="FinCorpus fin_exam.jsonl.gz 直接路径（缓存缺失时用）")
    parser.add_argument("--limit", type=int, default=None, help="限制评测样本数（调试用）")
    parser.add_argument("--judge_model", default=None, help="过程正确率的 LLM judge 模型（如 deepseek-chat）")
    parser.add_argument("--judge_api_key", default=None, help="LLM judge 的 API key")
    parser.add_argument("--judge_api_base", default="https://api.deepseek.com/v1", help="LLM judge 的 API base")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model, tokenizer = load_model(args.model, args.base_model)

    results_grpo = {}

    benchmarks = args.benchmark if "all" not in args.benchmark else ["fineval", "fincorpus", "reasoning"]

    if "fineval" in benchmarks:
        fineval_results = evaluate_fineval(model, tokenizer, limit=args.limit)
        results_grpo.update(fineval_results)
        print(f"\n[FinEval] 准确率: {fineval_results['accuracy']}%")

    if "fincorpus" in benchmarks:
        fc_results = evaluate_fincorpus_heldout(
            model, tokenizer, limit=(args.limit or 500), fincorpus_file=args.fincorpus_file,
            judge_model=args.judge_model, judge_api_key=args.judge_api_key, judge_api_base=args.judge_api_base,
        )
        results_grpo.update(fc_results)
        print(f"\n[FinCorpus留出集] 准确率: {fc_results.get('accuracy', 'N/A')}%")

    if "reasoning" in benchmarks:
        reasoning_results = evaluate_finance_reasoning(
            model, tokenizer, args.test_file,
            judge_model=args.judge_model, judge_api_key=args.judge_api_key,
            judge_api_base=args.judge_api_base,
        )
        results_grpo.update(reasoning_results)
        print(f"\n[金融推理] 答案准确率: {reasoning_results.get('answer_accuracy', 'N/A')}%")
        print(f"[金融推理] 步骤格式率: {reasoning_results.get('step_format_rate', 'N/A')}%")
        if "process_accuracy" in reasoning_results:
            print(f"[金融推理] 过程正确率: {reasoning_results['process_accuracy']}%")

    # 保存评测结果
    result_path = os.path.join(args.output_dir, "evaluation_grpo.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results_grpo, f, ensure_ascii=False, indent=2)
    print(f"\n评测结果保存: {result_path}")

    # SFT vs GRPO 对比
    if args.baseline:
        print(f"\n加载 SFT baseline: {args.baseline}")
        sft_model, sft_tokenizer = load_model(args.baseline, args.base_model)

        # 重新跑 baseline 评测（复用相同评测集）
        results_sft = {}
        if "fineval" in benchmarks:
            results_sft.update(evaluate_fineval(sft_model, sft_tokenizer, limit=args.limit))
        if "fincorpus" in benchmarks:
            results_sft.update(evaluate_fincorpus_heldout(
                sft_model, sft_tokenizer, limit=(args.limit or 500), fincorpus_file=args.fincorpus_file,
                judge_model=args.judge_model, judge_api_key=args.judge_api_key, judge_api_base=args.judge_api_base))
        if "reasoning" in benchmarks:
            results_sft.update(evaluate_finance_reasoning(
                sft_model, sft_tokenizer, args.test_file,
                judge_model=args.judge_model, judge_api_key=args.judge_api_key,
                judge_api_base=args.judge_api_base,
            ))

        compare_path = os.path.join(args.output_dir, "comparison_sft_vs_grpo.json")
        report = compare_models(results_sft, results_grpo, compare_path)

        print("\n========== SFT vs GRPO 对比 ==========")
        for metric, vals in report["comparison"]["metrics"].items():
            print(f"  {metric}: SFT={vals['sft']} → GRPO={vals['grpo']} (Δ={vals['delta']})")

    print("\n✅ 评测完成！")


if __name__ == "__main__":
    main()
