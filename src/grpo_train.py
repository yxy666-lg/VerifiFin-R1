#!/usr/bin/env python3
"""
Fin-R1 GRPO 对齐训练脚本（TRL GRPOTrainer）

前置条件：
  1. SFT 训练完成（results/sft/checkpoint-xxx，LoRA adapter）
  2. PRM 数据构建完成（data/prm/finance_prm.jsonl）
  3. 已生成 GRPO prompt 集：
     python src/data_prep_prm.py --mode grpo_prompts --trajectories data/prm/finance_prm.jsonl

用法：
  # Reward = LLM-as-a-Judge（先跑，快速验证管线）
  python src/grpo_train.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --sft_adapter results/sft/checkpoint-xxx \
    --prompt_file data/grpo/prompts.jsonl \
    --output_dir results/grpo \
    --api_key sk-xxx --api_base https://api.deepseek.com/v1 --judge_model deepseek-chat

  # Reward = 训好的 PRM 模型（本地、快、无 API 成本，推荐）
  python src/grpo_train.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --sft_adapter results/sft/checkpoint-xxx \
    --prompt_file data/grpo/prompts.jsonl \
    --output_dir results/grpo \
    --prm_model results/prm/final

Reward：FinanceReward（硬约束 + 软约束），软约束优先用 PRM 模型，否则 LLM-as-a-Judge。
"""

import argparse
import json
import os

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from reward_finance import FinanceReward

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_prompts(prompt_file: str) -> tuple[Dataset, dict]:
    """
    加载 GRPO prompt 集。
    返回 (dataset, answer_lookup)
    answer_lookup 只用于 reward 校验，不会喂给模型。
    """
    ds = load_dataset("json", data_files=prompt_file)["train"]

    answer_lookup = {}
    with open(prompt_file, encoding="utf-8") as f:
        for line in f:
            it = json.loads(line.strip())
            prompt = it.get("prompt", "")
            if prompt:
                answer_lookup[prompt] = it.get("answer", "")
    # TRL 只需要 prompt 列，多余的 answer 列可能干扰，去掉（答案已进 answer_lookup 供 reward 用）
    ds = ds.select_columns(["prompt"])
    print(f"[GRPO] prompt 集: {len(ds)} 条，{sum(1 for v in answer_lookup.values() if v)} 条带标准答案")
    return ds, answer_lookup


def load_prm_model(base_model: str, prm_dir: str, load_in_4bit: bool = True):
    """加载训好的 PRM（LoRA 三分类器）用于 reward。
    PRM 只做前向打分(无梯度)，4bit 省显存不影响精度，避免和策略模型一起 OOM。"""
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=3,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=load_in_4bit,
    )
    model = PeftModel.from_pretrained(model, prm_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # PRM 分类器批量打分(batch>1)必须设 pad_token_id，否则 Qwen2ForSequenceClassification 报错
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


class DynamicKLGRPOTrainer(GRPOTrainer):
    """动态 KL：TRL 在 __init__ 里缓存了 self.beta，loss 用 self.beta 算 KL。
    回调改 args.beta 无效，必须在每步 compute_loss 前更新 self.beta。"""

    def __init__(self, kl_schedule, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_schedule = kl_schedule  # callable(step) -> beta

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.kl_schedule is not None:
            new_beta = self.kl_schedule(self.state.global_step)
            if abs(self.beta - new_beta) > 1e-9:
                print(f"[KL] step {self.state.global_step}: beta {self.beta} -> {new_beta}")
            self.beta = new_beta
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


def main():
    parser = argparse.ArgumentParser(description="Fin-R1 GRPO 对齐训练")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct", help="基座模型")
    parser.add_argument("--sft_adapter", default=None, help="SFT LoRA adapter 路径（从 SFT 继续）")
    parser.add_argument("--prompt_file", default="data/grpo/prompts.jsonl", help="GRPO prompt 集")
    parser.add_argument("--output_dir", default="results/grpo", help="输出目录")
    parser.add_argument("--prm_model", default=None, help="PRM 模型路径（推荐，作为 reward 后端）")
    parser.add_argument("--api_key", default=None, help="LLM judge API key（--prm_model 为空时用）")
    parser.add_argument("--api_base", default="https://api.openai.com/v1", help="LLM judge API base")
    parser.add_argument("--judge_model", default="gpt-4o-mini", help="LLM judge 模型")
    parser.add_argument("--enable_soft_judge", action="store_true", default=False,
                        help="没有 PRM 模型时，是否启用 LLM judge 软约束")
    parser.add_argument("--format_weight", type=float, default=0.10)
    parser.add_argument("--process_weight", type=float, default=0.55)
    parser.add_argument("--answer_weight", type=float, default=0.25)
    parser.add_argument("--verifier_weight", type=float, default=0.10)
    parser.add_argument("--hacking_penalty_weight", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=2, help="per device train batch")
    parser.add_argument("--grad_accum", type=int, default=4, help="gradient accumulation")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="每组采样数（group size）。GRPO 会加载参考模型+PRM，40GB 显存下 8 会 OOM，默认 4")
    parser.add_argument("--max_prompt_length", type=int, default=1024, help="prompt 最大长度")
    parser.add_argument("--max_completion_length", type=int, default=512,
                        help="回答最大长度（金融推理一般 300-500 token，512 够用且省显存）")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    parser.add_argument("--beta", type=float, default=0.04, help="KL 散度系数（constant 模式用）")
    parser.add_argument("--beta_schedule", choices=["constant", "dynamic"], default="constant",
                        help="dynamic=前一半低 beta 探索，后一半高 beta 稳定收敛")
    parser.add_argument("--beta_explore", type=float, default=0.02, help="动态 KL 探索期 beta")
    parser.add_argument("--beta_stable", type=float, default=0.08, help="动态 KL 收敛期 beta")
    parser.add_argument("--beta_switch_ratio", type=float, default=0.5, help="动态 KL 切换点（步数占比）")
    parser.add_argument("--max_steps", type=int, default=-1, help="最大步数（-1 = 按 epochs）")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="GRPO 每步很慢(生成8个回答+训练)，6683 prompts 下 1 epoch≈5-6h。"
                             "先用 --max_steps 300 验证管线，再放全量")
    parser.add_argument("--load_in_4bit", action="store_true", default=False,
                        help="基座 4bit 量化（省显存）。默认 bf16：4bit+TRL GRPO 有 pad_token 配置传播 bug，bf16 更稳")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--shutdown_when_done", action="store_true",
                        help="训练成功完成后自动关机(AutoDL,过夜跑全量用,避免空转计费)。试跑别加这个。")
    args = parser.parse_args()

    if not args.prm_model and not args.enable_soft_judge:
        raise SystemExit(
            "必须二选一：--prm_model（推荐）或 --enable_soft_judge（LLM judge，需 --api_key）"
        )
    if args.enable_soft_judge and not (args.api_key or os.environ.get("OPENAI_API_KEY")):
        raise SystemExit("--enable_soft_judge 需要 --api_key 或环境变量 OPENAI_API_KEY")

    # ---------- 加载基座 + SFT adapter ----------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id  # 显式对齐 pad/id

    print(f"[GRPO] 加载基座: {args.base_model}（load_in_4bit={args.load_in_4bit}）")
    # 不用 device_map="auto"：交给 TRL/accelerate 统一放 GPU，避免 dispatch 干扰 LoRA 梯度
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        load_in_4bit=args.load_in_4bit,
    )

    # 4bit 量化必须调用 prepare_model_for_kbit_training，否则参数 requires_grad=False，
    # 训练时 loss 连不上梯度报 "element 0 of tensors does not require grad"
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if args.sft_adapter:
        print(f"[GRPO] 从 SFT adapter 继续: {args.sft_adapter}")
        model = PeftModel.from_pretrained(model, args.sft_adapter)
    else:
        print("[GRPO] 无 SFT adapter，新建 LoRA（建议先 SFT 再 GRPO）")
        lora_config = LoraConfig(
            r=64, lora_alpha=128, target_modules=LORA_TARGET_MODULES, lora_dropout=0.05, bias="none",
        )
        model = get_peft_model(model, lora_config)
    model.config.pad_token_id = tokenizer.pad_token_id      # 训练 forward 需要
    model.generation_config.pad_token_id = tokenizer.pad_token_id  # 生成(采样)需要
    model.train()
    # 显式确保 LoRA 参数 requires_grad，否则 GRPO 的 loss 连不上梯度
    trainable = sum(1 for n, p in model.named_parameters() if p.requires_grad)
    print(f"[GRPO] 可训练参数: {trainable} 个（>0 才算 LoRA 生效）")
    if trainable == 0:
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"[GRPO] 兜底后可训练参数: {trainable} 个")
    model.print_trainable_parameters()

    # ---------- prompt 集 + reward ----------
    ds, answer_lookup = load_prompts(args.prompt_file)
    if len(ds) == 0:
        raise SystemExit("GRPO prompt 集为空，请先跑 data_prep_prm.py --mode grpo_prompts")

    if args.prm_model:
        print(f"[GRPO] reward 后端：PRM 模型 {args.prm_model}")
        prm_model, prm_tokenizer = load_prm_model(args.base_model, args.prm_model)
        reward_fn = FinanceReward(
            prm_model=prm_model, prm_tokenizer=prm_tokenizer, answer_lookup=answer_lookup,
            hard_weight=args.format_weight, soft_weight=args.process_weight,
            answer_weight=args.answer_weight, verifier_weight=args.verifier_weight,
            hacking_penalty_weight=args.hacking_penalty_weight,
        )
    else:
        print(f"[GRPO] reward 后端：LLM-as-a-Judge ({args.judge_model})")
        reward_fn = FinanceReward(
            api_key=args.api_key, api_base=args.api_base, model=args.judge_model,
            enable_soft_judge=True, answer_lookup=answer_lookup,
            hard_weight=args.format_weight, soft_weight=args.process_weight,
            answer_weight=args.answer_weight, verifier_weight=args.verifier_weight,
            hacking_penalty_weight=args.hacking_penalty_weight,
        )

    # ---------- GRPO 配置 ----------
    # 不同 trl 版本的 GRPOConfig 参数有差异（如 top_p 并非所有版本都有），
    # 这里运行时检测签名，自动过滤掉当前版本不支持的参数，避免版本漂移崩溃。
    import inspect
    _supported = inspect.signature(GRPOConfig.__init__).parameters
    _config_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        learning_rate=args.lr,
        beta=args.beta,                 # KL 散度系数
        temperature=0.8,
        top_p=0.95,
        max_steps=args.max_steps,
        num_train_epochs=args.num_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_steps=args.save_steps,
        bf16=True,
        gradient_checkpointing=True,
        seed=42,
    )
    _dropped = [k for k in _config_kwargs if k not in _supported]
    if _dropped:
        print(f"[GRPO] 当前 trl 不支持以下参数，已忽略: {_dropped}")
    _config_kwargs = {k: v for k, v in _config_kwargs.items() if k in _supported}
    training_args = GRPOConfig(**_config_kwargs)

    # ---------- 动态 KL（升级1：先探索后收敛） ----------
    # 注意：TRL 缓存 self.beta，必须用 DynamicKLGRPOTrainer 子类在每步更新，
    #       普通 callback 改 args.beta 是无效的（已核实 TRL 源码）。
    trainer_cls = GRPOTrainer
    if args.beta_schedule == "dynamic":
        steps_per_epoch = max(
            1, len(ds) // max(1, training_args.per_device_train_batch_size * training_args.num_generations)
        )
        total_steps = steps_per_epoch * training_args.num_train_epochs
        switch_step = int(total_steps * args.beta_switch_ratio)

        def kl_schedule(step):
            return args.beta_explore if step < switch_step else args.beta_stable

        trainer_cls = DynamicKLGRPOTrainer
        print(f"[GRPO] 动态 KL: 前 {switch_step} 步 beta={args.beta_explore}（探索），"
              f"之后 beta={args.beta_stable}（收敛），共约 {total_steps} 步")

    trainer_kwargs = dict(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=ds,
    )
    # tokenizer / processing_class 兼容（不同 trl 版本命名不同）
    import inspect
    _tp = inspect.signature(GRPOTrainer.__init__).parameters
    if "tokenizer" in _tp:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in _tp:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        print("[GRPO] 警告: GRPOTrainer 不认 tokenizer/processing_class，已跳过（可能影响文本处理）")
    if args.beta_schedule == "dynamic":
        trainer_kwargs["kl_schedule"] = kl_schedule
    trainer = trainer_cls(**trainer_kwargs)

    # 最后一道保险：确保 trainer 持有的模型也有 pad_token_id（Qwen 基座默认 None）
    trainer.model.config.pad_token_id = tokenizer.pad_token_id
    trainer.model.generation_config.pad_token_id = tokenizer.pad_token_id

    print("[GRPO] 开始训练...")
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))

    print(f"\n✅ GRPO 训练完成！模型保存: {os.path.join(args.output_dir, 'final')}")
    print("   评测: python src/evaluate.py --model results/grpo/final --base_model <基座> --benchmark all")

    # 过夜跑全量：训练成功完成后自动关机，避免空转计费
    if args.shutdown_when_done:
        import time
        print("训练完成，5 秒后自动关机（AutoDL），数据已保存，重新开机即可继续评测。")
        time.sleep(5)
        os.system("shutdown")


if __name__ == "__main__":
    main()
