#!/usr/bin/env python3
"""
Fin-R1 PRM 模型训练（LoRA 三分类器）

把"问题 + 历史步骤 + 当前步骤"喂给模型，预测当前步骤是：
  0 = 错误 / 1 = 部分正确 / 2 = 正确

数据源：data/prm/finance_prm.jsonl（data_prep_prm.py 标注产物）
样本构造：每条轨迹的每一步 = 一条样本：
  text = question + 历史步骤前缀 + 当前 step
  label = label(-1/0/1) → (0/1/2)

用途：训练完成后作为 GRPO 的 reward 模型（本地、快、无 API 成本），
      也可用于评测时的过程正确率打分（替代 LLM judge）。

用法：
  python src/prm_train.py \
    --data data/prm/finance_prm.jsonl \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --output_dir results/prm

  # GRPO 用 PRM 做 reward
  python src/grpo_train.py ... --prm_model results/prm/checkpoint-xxx
"""

import argparse
import json
import os

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# label 映射：PRM 标注的 label (-1 错误 / 0 部分 / 1 正确) → 分类器 label (0/1/2)
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
NUM_LABELS = 3
LABEL_NAMES = ["错误", "部分正确", "正确"]

# LoRA 目标模块（Qwen2 结构）
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def format_prm_input(question: str, prefix_steps: list[str], current_step: str) -> str:
    """PRM 输入包含完整推理前缀，避免脱离上下文判断单步。"""
    history = "\n".join(prefix_steps) if prefix_steps else "（无）"
    return (
        f"问题：{question}\n\n"
        f"此前推理：\n{history}\n\n"
        f"待评估步骤：\n{current_step}"
    )


def load_samples(data_path: str, max_samples: int = 100000) -> list[dict]:
    """按轨迹展开样本；group_id 用于按题切分，杜绝步骤级泄漏。"""
    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            question = item.get("question", "")
            steps = item.get("steps", [])
            if not steps:
                continue
            prefix = []
            group_id = str(item.get("question_id") or item.get("id") or question).strip()
            for s in steps:
                step_text = s.get("step", "")
                label = s.get("label")
                if not step_text or label is None or label not in LABEL_MAP:
                    if step_text:
                        prefix.append(step_text)
                    continue
                samples.append({
                    "text": format_prm_input(question, prefix, step_text),
                    "label": LABEL_MAP[label],
                    "group_id": group_id,
                })
                prefix.append(step_text)
                if len(samples) >= max_samples:
                    return samples
    return samples


def tokenize_function(tokenizer, examples, max_length: int):
    enc = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    enc["labels"] = examples["label"]
    return enc


def _expected_calibration_error(probs, labels, bins: int = 10) -> float:
    confidence = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += mask.mean() * abs((predictions[mask] == labels[mask]).mean() - confidence[mask].mean())
    return float(ece)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    preds = np.argmax(logits, axis=-1)
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    one_hot = np.eye(NUM_LABELS)[labels]
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1))),
        "ece": _expected_calibration_error(probs, labels),
    }


def group_split(samples: list[dict], val_size: float, seed: int) -> tuple[list[dict], list[dict]]:
    """按 question/trajectory 分组切分，保证同一道题只属于一个集合。"""
    import random
    groups = sorted({s["group_id"] for s in samples})
    random.Random(seed).shuffle(groups)
    n_val = max(1, round(len(groups) * val_size))
    val_groups = set(groups[:n_val])
    train = [s for s in samples if s["group_id"] not in val_groups]
    val = [s for s in samples if s["group_id"] in val_groups]
    assert not ({s["group_id"] for s in train} & {s["group_id"] for s in val})
    return train, val


class WeightedTrainer(Trainer):
    """带类别权重的 Trainer：缓解 PRM 数据 +1 占比过高导致的类别不平衡"""

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # 权重 dtype 对齐 logits，避免 BFloat16/Float 不一致（云上 autocast 会兜底，这里显式对齐更稳）
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device).to(logits.dtype))
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(description="Fin-R1 PRM 模型训练")
    parser.add_argument("--data", default="data/prm/finance_prm.jsonl", help="PRM 标注数据")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct", help="基座模型")
    parser.add_argument("--output_dir", default="results/prm", help="输出目录")
    parser.add_argument("--max_length", type=int, default=512, help="单步最大长度")
    parser.add_argument("--max_samples", type=int, default=100000, help="最大样本数")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--batch_size", type=int, default=4, help="per device batch（7B 默认 4 防 OOM）")
    parser.add_argument("--grad_accum", type=int, default=4, help="gradient accumulation")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--val_size", type=float, default=0.05, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = load_samples(args.data, max_samples=args.max_samples)
    if len(samples) < 10:
        raise SystemExit(f"PRM 样本太少（{len(samples)}），请先跑 data_prep_prm.py --mode annotate")

    label_dist = np.bincount([s["label"] for s in samples], minlength=3)
    print(f"样本数: {len(samples)}，标签分布(错误/部分/正确): {label_dist.tolist()}")

    # 类别权重：缓解 +1(正确)占比过高(~88%)导致的类别不平衡，
    # 否则分类器会学到"几乎什么都判正确"的懒惰先验。
    from sklearn.utils.class_weight import compute_class_weight
    label_arr = np.array([s["label"] for s in samples])
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=label_arr)
    print(f"类别权重(错误/部分/正确): {[round(float(w), 2) for w in class_weights]}")

    train_samples, val_samples = group_split(samples, args.val_size, args.seed)
    print(f"按题分组切分: train={len(train_samples)} steps / val={len(val_samples)} steps，题目零重叠")
    train_ds_raw = Dataset.from_list(train_samples)
    eval_ds_raw = Dataset.from_list(val_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=NUM_LABELS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.id2label = {i: l for i, l in enumerate(LABEL_NAMES)}
    model.config.label2id = {l: i for i, l in enumerate(LABEL_NAMES)}
    model.config.pad_token_id = tokenizer.pad_token_id  # batch>1 需要 pad token

    # LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = train_ds_raw.map(
        lambda e: tokenize_function(tokenizer, e, args.max_length),
        batched=True,
        remove_columns=train_ds_raw.column_names,
    )
    eval_ds = eval_ds_raw.map(
        lambda e: tokenize_function(tokenizer, e, args.max_length),
        batched=True,
        remove_columns=eval_ds_raw.column_names,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        report_to=[],
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

    # 打印验证集准确率
    eval_res = trainer.evaluate()
    print(f"\n✅ PRM 训练完成！验证集准确率: {eval_res.get('eval_accuracy', 'N/A')}")
    print(f"   模型保存: {os.path.join(args.output_dir, 'final')}")
    print("   在 GRPO 中作为 reward 使用: python src/grpo_train.py --prm_model <此目录>")


if __name__ == "__main__":
    main()
