#!/usr/bin/env python3
"""
Fin-R1 PRM（过程监督）数据构建脚本

功能：
  1. 从 FinEval dev + 自研金融推理题 + FinCorpus fin_exam 生成 CoT 推理轨迹
  2. 用 LLM-as-a-Judge 标注每步正确性（**必须带标准答案**，否则标签不可信）
  3. 输出 PRM 训练/评测数据

核心思路：
  - 用 GPT-4 / DeepSeek API 生成"问题 → CoT 步骤 → 最终答案"
  - 对 CoT 的每一步用 LLM-as-a-Judge 判断：正确 / 部分正确 / 错误
  - judge 以【标准答案】为基准，任何偏离标准答案的步骤都不能给满分
  - 人肉抽检约 5-10%（关键样本全检）

使用方式：
  # 生成推理轨迹（新数据：内部携带标准答案）
  python src/data_prep_prm.py --mode generate --api_key YOUR_KEY --output_dir data/prm

  # 旧轨迹回填答案（不重新生成，直接复用已有 trajectories.jsonl）
  python src/data_prep_prm.py --mode backfill --trajectories data/prm/trajectories.jsonl --output_dir data/prm

  # 标注（LLM-as-a-Judge，带标准答案）
  python src/data_prep_prm.py --mode annotate --trajectories data/prm/trajectories.jsonl --output_dir data/prm

  # 从标注结果构建 GRPO prompt 集
  python src/data_prep_prm.py --mode grpo_prompts --trajectories data/prm/finance_prm.jsonl --output_dir data/grpo

输出格式（每条数据）：
{
  "question": "...",
  "answer": "A",                 # 标准答案（来自题目源）
  "trajectory": "...",            # 完整 CoT 文本
  "final_answer": "...",          # 最终结论段
  "difficulty": "medium",         # easy / medium / hard
  "steps": [
    {"step": "第一步：确定折旧方法...", "label": 1, "confidence": 0.95, "reason": "..."},
    {"step": "第二步：计算年折旧额...", "label": 1, "confidence": 0.88, "reason": "..."},
    {"step": "第三步：汇总...", "label": -1, "confidence": 0.72, "reason": "..."}
  ]
  # label: 1=正确, 0=部分正确, -1=错误
}
"""

import argparse
import glob
import gzip
import json
import os
import re
import time
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------- 非金融内容过滤 ----------
# FinCorpus fin_exam 混入了大量非金融考题（航海/物理/化学等），
# 这些会稀释"金融推理"数据的可信度，生成轨迹前先过滤。
NON_FINANCE_KEYWORDS = [
    "船舶", "港口", "航海", "航道", "灯塔", "船员", "货物装载",
    "气象", "气压", "风速", "降水量",
    "化学", "物理", "电路", "电压", "电阻", "发动机", "轴承", "机械",
    "医疗", "药物", "病症", "解剖",
    "外语", "编程", "代码", "语法", "单词",
]


def is_finance_relevant(text: str) -> bool:
    """粗粒度过滤明显非金融的题目（保守策略：宁缺毋滥）"""
    return not any(kw in text for kw in NON_FINANCE_KEYWORDS)


# ---------- 金融推理题 Prompt ----------
COT_GENERATION_PROMPT = """你是一位金融专家。请对以下金融问题给出详细的逐步推理过程。

要求：
1. 每一步推理独立成段，用「第N步：」开头
2. 涉及计算的步骤，必须写出完整的计算式和单位
3. 涉及判断的步骤，必须引用具体数据或准则
4. 最后一步给出最终结论

问题：{question}

请开始你的逐步推理："""


PRM_ANNOTATION_PROMPT = """你是一位金融推理评分专家。下面是一条金融推理轨迹，请对每一步单独评分。

评分标准：
- 2 分：推理完全正确，逻辑严密，计算无误
- 1 分：推理方向正确但有轻微瑕疵（如数据引用不全、逻辑跳跃）
- 0 分：推理有误（如计算错误、错误引用、逻辑矛盾）
- -1 分：推理严重错误或完全无关

【标准答案】{answer}

问题：
{question}

推理轨迹：
{trajectory}

评分要点：
1. 必须以【标准答案】为基准。任何导致最终结论偏离标准答案的关键步骤，都不能给 2 分
2. 中间计算错误的步骤，即使最终结论"歪打正着"，也要按实际计算正确性给分
3. 请对照轨迹原文逐一评分，不要凭空补全或臆测缺失的步骤
4. reason 请控制在 20 字以内，简洁说明评分依据（避免输出过长被截断）

请输出 JSON 数组，每步一个评分：
[{{"step_index": 0, "score": 2, "reason": "..."}}, {{"step_index": 1, "score": 1, "reason": "..."}}, ...]

只输出 JSON，不需要其他文字。"""


def extract_steps(trajectory: str) -> list[str]:
    """从推理轨迹中按「第N步」拆分"""
    steps = re.split(r"(?=第\d+步[：:])", trajectory)
    steps = [s.strip() for s in steps if s.strip()]
    # 如果没匹配到「第N步」格式，按段落拆分
    if not steps:
        steps = [p.strip() for p in trajectory.split("\n\n") if p.strip()]
    return steps


def extract_final_answer(steps: list[str]) -> str:
    """最终结论 = 最后一步（或轨迹末尾的结论段）"""
    if not steps:
        return ""
    # 结论通常以"最终结论/综上/答案是"开头，取含结论词的最后一段
    for s in reversed(steps):
        if re.search(r"(最终结论|综上|综上所述|答案是|因此|所以)", s):
            return s
    return steps[-1]


def generate_trajectories(
    items: list[dict],
    api_key: str,
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-4o",
    max_retries: int = 3,
) -> list[dict]:
    """
    用 GPT-4 / DeepSeek 生成推理轨迹
    items: [{"question", "answer", "source"}, ...]
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=api_base, timeout=120)
    results = []

    for item in tqdm(items, desc="生成推理轨迹"):
        q = item["question"]
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": COT_GENERATION_PROMPT.format(question=q)}
                    ],
                    temperature=0.7,
                    max_tokens=2048,
                )
                trajectory = response.choices[0].message.content
                steps = extract_steps(trajectory)
                results.append({
                    "question": q,
                    "answer": item.get("answer", ""),
                    "source": item.get("source", ""),
                    "trajectory": trajectory,
                    "steps": steps,
                })
                break
            except Exception as e:
                print(f"  API 错误 (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)
        else:
            print(f"  跳过（重试耗尽）: {q[:50]}...")

        time.sleep(0.5)  # rate limit

    return results


def annotate_steps(
    trajectories: list[dict],
    api_key: str,
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-4o",
    max_retries: int = 5,
    skip_no_answer: bool = True,
    out_fh=None,
) -> list[dict]:
    """
    用 LLM-as-a-Judge 对每一步标注正确性（带标准答案）
    skip_no_answer=True 时，无标准答案的轨迹直接跳过（它们的标签不可信，标了也会被丢弃）
    out_fh：传入已打开的(追加)文件句柄时，每标注完一条立即写入并 flush，
           中途崩溃/中断也不会丢失已完成的进度（配合断点续跑）。
    max_retries=5：DeepSeek 偶发慢窗口/超时，提高重试次数避免丢数据。
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=api_base, timeout=120)
    results = []
    missing_answer = 0

    for item in tqdm(trajectories, desc="PRM 标注"):
        question = item["question"]
        trajectory = item["trajectory"]
        raw_steps = item["steps"]
        answer = item.get("answer", "")

        if not answer:
            missing_answer += 1
            if skip_no_answer:
                continue
            print(f"  [警告] 无标准答案，标注不可靠: {question[:40]}...")

        prompt = PRM_ANNOTATION_PROMPT.format(
            question=question, trajectory=trajectory,
            answer=answer if answer else "（未知，请按推理自洽性判断）",
        )

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,  # 评分任务用低温
                    max_tokens=3072,  # 足够容纳长 JSON，避免输出被截断导致丢轨迹
                )
                scores_raw = response.choices[0].message.content

                # 提取 JSON
                json_match = re.search(r"\[[\s\S]*\]", scores_raw)
                if not json_match:
                    raise ValueError(f"Response 中无 JSON 数组: {scores_raw[:200]}")
                scores = json.loads(json_match.group())

                # 将 score 映射到 label
                annotated_steps = []
                for s in scores:
                    idx = s.get("step_index")
                    # step_index 缺失或越界 → 跳过，避免全部堆到第 0 步
                    if idx is None or not isinstance(idx, int) or not (0 <= idx < len(raw_steps)):
                        continue
                    score = s.get("score", 0)
                    label = 1 if score >= 2 else (0 if score == 1 else -1)
                    annotated_steps.append({
                        "step": raw_steps[idx],
                        "label": label,          # 1=正确, 0=部分正确, -1=错误
                        "confidence": abs(score) / 2.0,
                        "reason": s.get("reason", ""),
                    })

                entry = {
                    "question": question,
                    "answer": answer,
                    "trajectory": trajectory,
                    "final_answer": extract_final_answer(raw_steps),
                    "steps": annotated_steps,
                    "difficulty": classify_difficulty(question),
                }
                # 边标边写：崩溃/中断不丢已完成的进度
                if out_fh is not None:
                    out_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    out_fh.flush()
                results.append(entry)
                break
            except Exception as e:
                print(f"  标注错误 (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)
        else:
            print(f"  跳过（重试耗尽）: {question[:50]}...")

        time.sleep(0.5)

    if missing_answer:
        print(f"[警告] 共 {missing_answer}/{len(trajectories)} 条轨迹缺少标准答案，建议先用 --mode backfill 回填。")
    return results


def classify_difficulty(question: str) -> str:
    """根据问题长度和关键词简单分类难度"""
    length = len(question)
    hard_keywords = ["计算", "推导", "模型", "分析", "比较", "论述"]
    easy_keywords = ["定义", "简述", "列举", "什么是", "请解释"]

    hard_count = sum(1 for kw in hard_keywords if kw in question)
    easy_count = sum(1 for kw in easy_keywords if kw in question)

    # 先看"硬"信号：含计算/推导关键词就不可能是 easy
    if hard_count >= 2 or (hard_count >= 1 and length > 150):
        return "hard"
    if hard_count >= 1:
        return "medium"
    if easy_count >= 2:
        return "easy"
    if length < 50:
        return "easy"
    return "medium"


def load_questions(max_exam: int = 10000) -> list[dict]:
    """
    加载金融推理题来源（返回 question + 标准答案 + source）：
      1. FinCorpus fin_exam（有答案+解析的考试题，优先）
      2. FinEval dev（有 explanation 的）
      3. 自研题库（data/prm/questions.jsonl，可带 answer）
    """
    questions = []

    # 1. FinCorpus fin_exam（从缓存直接读 .gz）
    print("加载 FinCorpus fin_exam...")
    try:
        cache_pattern = os.path.expanduser(
            "~/.cache/huggingface/hub/datasets--Duxiaoman-DI--FinCorpus/snapshots/*/data/fin_exam.jsonl.gz"
        )
        gz_files = glob.glob(cache_pattern)
        if not gz_files:
            blob_pattern = os.path.expanduser(
                "~/.cache/huggingface/hub/datasets--Duxiaoman-DI--FinCorpus/blobs/*"
            )
            all_blobs = glob.glob(blob_pattern)
            gz_files = [f for f in all_blobs if 10 * 1024 * 1024 < os.path.getsize(f) < 200 * 1024 * 1024]

        if gz_files:
            count = 0
            with gzip.open(gz_files[0], "rt", encoding="utf-8") as f:
                for line in f:
                    if count >= max_exam:
                        break
                    try:
                        item = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    text = item.get("text", "")

                    # 只取有答案和解析、且属于金融的题
                    answer_match = re.search(r"答案[：:]\s*([A-E]+(?:[、,，]?[A-E])*)", text)
                    has_explanation = re.search(r"分析解释[：:]\s*\S+", text)
                    if not (answer_match and has_explanation and len(text) > 30):
                        continue
                    if not is_finance_relevant(text):
                        continue

                    # 提取完整答案字母（多选题如"A、B、C"不能只取 A）
                    answer_letters = "".join(re.findall(r"[A-E]", answer_match.group(1)))
                    if not answer_letters:
                        continue

                    # 用匹配位置切题，避免题面里出现"答案"字样导致截断
                    answer_pos = answer_match.start()
                    question = text[:answer_pos].strip()
                    if len(question) > 20:
                        questions.append({
                            "question": question,
                            "answer": answer_letters,
                            "source": "fin_corpus",
                        })
                        count += 1
            print(f"  FinCorpus: {count} 条")
        else:
            print("  FinCorpus 缓存未找到")
    except Exception as e:
        print(f"  FinCorpus 加载失败: {e}")

    # 2. FinEval dev（有 explanation 的 170 条）
    print("加载 FinEval dev...")
    try:
        import huggingface_hub
        zip_path = huggingface_hub.hf_hub_download(
            repo_id="SUFE-AIFLM-Lab/FinEval",
            filename="FinEval.zip",
            repo_type="dataset",
        )
        with zipfile.ZipFile(zip_path) as z:
            dev_csvs = [n for n in z.namelist() if n.startswith("dev/") and n.endswith(".csv")]
            for name in dev_csvs:
                with z.open(name) as f:
                    df = pd.read_csv(f)
                for _, row in df.iterrows():
                    q = str(row.get("question", "")).strip()
                    answer = str(row.get("answer", "")).strip()
                    explanation = str(row.get("explanation", "")).strip()
                    if not (explanation and explanation != "nan" and len(q) > 20 and answer):
                        continue
                    # 拼上选项
                    options = []
                    for opt in ["A", "B", "C", "D"]:
                        val = row.get(opt, "")
                        if pd.notna(val) and str(val).strip():
                            options.append(f"{opt}. {str(val).strip()}")
                    if options:
                        q = q + "\n" + "\n".join(options)
                    questions.append({
                        "question": q,
                        "answer": answer,
                        "source": "fineval_dev",
                    })
        print(f"  FinEval dev: 已加载")
    except Exception as e:
        print(f"  FinEval 加载失败: {e}")

    # 3. 自研题库（可带 answer）
    self_path = Path("data/prm/questions.jsonl")
    if self_path.exists():
        with open(self_path, encoding="utf-8") as f:
            for line in f:
                it = json.loads(line.strip())
                q = it.get("question", "")
                if q:
                    questions.append({
                        "question": q,
                        "answer": it.get("answer", ""),
                        "source": "self_built",
                    })
        print(f"  自研题库: {self_path}")

    # 去重（基于题面前 50 字）
    seen = set()
    deduped = []
    for it in questions:
        key = it["question"][:50]
        if key not in seen:
            seen.add(key)
            deduped.append(it)

    print(f"总题量: {len(deduped)} 条")
    return deduped


def backfill_answers(traj_path: str, out_path: str) -> None:
    """
    给已有 trajectories.jsonl 回填标准答案（按题面前 50 字匹配），
    这样旧的轨迹不用重新花钱生成，补上答案后即可重新标注。
    """
    print(f"[Backfill] 加载题目源...")
    source = load_questions()
    lookup = {}
    for it in source:
        lookup[it["question"][:50]] = it["answer"]

    total = 0
    filled = 0
    with open(traj_path, encoding="utf-8") as f:
        lines = f.readlines()

    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            item = json.loads(line.strip())
            total += 1
            ans = lookup.get(item["question"][:50], "")
            if ans:
                filled += 1
            item["answer"] = ans
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[Backfill] 完成 {filled}/{total} 条已回填答案 → {out_path}")
    print("[Backfill] 回填后请重新运行 --mode annotate 覆盖标注。")


def build_grpo_prompts(prm_path: str, out_path: str) -> None:
    """
    从标注后的 PRM 数据构建 GRPO prompt 集。
    GRPO 需要的是"问题 → 采样推理"的 prompt 集（而不是标注数据），
    answer 字段用于 reward 校验，不会喂给模型。
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    count = 0
    with open(prm_path, encoding="utf-8") as f:
        with open(out_path, "w", encoding="utf-8") as out:
            for line in f:
                item = json.loads(line.strip())
                if not item.get("steps"):
                    continue  # 无有效标注的轨迹不进入 GRPO
                prompt = (
                    "请对以下金融问题进行详细的逐步推理，每一步用「第N步：」开头，"
                    "涉及计算必须写出完整计算式和单位，最后给出明确结论。\n\n"
                    f"问题：{item['question']}"
                )
                out.write(json.dumps({
                    "prompt": prompt,
                    "answer": item.get("answer", ""),
                }, ensure_ascii=False) + "\n")
                count += 1
    print(f"[GRPO Prompts] {count} 条 → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fin-R1 PRM 数据构建")
    parser.add_argument("--mode", choices=["generate", "backfill", "annotate", "full", "grpo_prompts"], required=True,
                        help="generate=生成轨迹, backfill=回填答案, annotate=标注, full=全流程, grpo_prompts=构建GRPO prompt集")
    parser.add_argument("--api_key", default=None, help="OpenAI / DeepSeek API key（或设环境变量 OPENAI_API_KEY）")
    parser.add_argument("--api_base", default="https://api.openai.com/v1",
                        help="API base URL（DeepSeek: https://api.deepseek.com/v1）")
    parser.add_argument("--model", default="gpt-4o", help="模型名")
    parser.add_argument("--output_dir", default="data/prm", help="输出目录")
    parser.add_argument("--trajectories", default=None, help="推理轨迹文件（backfill / annotate / grpo_prompts 模式需要）")
    parser.add_argument("--max_questions", type=int, default=10000, help="最大题数")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # backfill 不需要 API key
    if args.mode == "backfill":
        traj_path = args.trajectories or os.path.join(args.output_dir, "trajectories.jsonl")
        out_path = os.path.join(args.output_dir, "trajectories_answered.jsonl")
        backfill_answers(traj_path, out_path)
        return

    if args.mode == "grpo_prompts":
        prm_path = args.trajectories or os.path.join(args.output_dir, "finance_prm.jsonl")
        out_path = os.path.join("data/grpo", "prompts.jsonl")
        build_grpo_prompts(prm_path, out_path)
        return

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ 请设置 API Key：--api_key 或环境变量 OPENAI_API_KEY")
        return

    if args.mode in ("generate", "full"):
        items = load_questions()
        items = items[:args.max_questions]

        trajectories = generate_trajectories(items, api_key, args.api_base, args.model)

        traj_path = os.path.join(args.output_dir, "trajectories.jsonl")
        with open(traj_path, "w", encoding="utf-8") as f:
            for t in trajectories:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"\n轨迹保存: {traj_path} ({len(trajectories)} 条)")

    if args.mode in ("annotate", "full"):
        traj_path = args.trajectories or os.path.join(args.output_dir, "trajectories.jsonl")
        trajectories = []
        with open(traj_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    trajectories.append(json.loads(line))

        # 断点续跑：跳过已标注过的问题（崩溃/中断后可无缝续跑）
        prm_path = os.path.join(args.output_dir, "finance_prm.jsonl")
        done = set()
        if os.path.exists(prm_path):
            with open(prm_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        done.add(json.loads(line.strip()).get("question", ""))
                    except json.JSONDecodeError:
                        continue
            to_annotate = [t for t in trajectories if t.get("question", "") not in done]
            print(f"[Resume] 已标注 {len(done)} 条，本次将标注剩余 {len(to_annotate)} 条")
        else:
            to_annotate = trajectories

        # 追加写 + 边标边写：崩溃/中断不丢已完成的进度，重启自动续
        with open(prm_path, "a", encoding="utf-8") as f:
            annotated = annotate_steps(
                to_annotate, api_key, args.api_base, args.model, out_fh=f
            )
        total = len(done) + len(annotated)
        print(f"\nPRM 数据保存: {prm_path} (本次新增 {len(annotated)} 条，累计 {total} 条)")

    print("\n✅ PRM 数据构建完成！")


if __name__ == "__main__":
    main()
