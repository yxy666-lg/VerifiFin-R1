# VerifiFin-R1

面向金融推理的过程监督与可验证强化学习工具箱。仓库按以下流程组织代码：

`SFT -> PRM 数据 -> PRM 三分类器 -> Rule/PRM Reward -> GRPO -> Best-of-N`

项目基于 Qwen2.5-7B-Instruct，重点解决金融多步推理中“答案碰巧正确、过程错误难定位、单一 LLM Judge 不稳定”的问题。仓库提供数据审计、过程奖励、受限金融公式 DSL、统计检验和推理重排代码。

## 证据状态

- 整理 22,462 条 SFT 样本，构建 7,839 条候选推理轨迹；修复答案截断与标签污染后保留 6,683 条过程监督轨迹。
- 在 2,000 条轨迹、10,177 个步骤上运行算式审计；594 个步骤可被规则解析，发现 112 个 Rule-Judge 冲突案例。
- 旧版统一评测：FinEval-val（1,151 题）SFT 为 73.15%，GRPO 为 73.76%。这些结果仅作为已跑通训练闭环的探索性证据。
- Rule/PRM混合Reward、新版GRPO和Best-of-N重排已经提供实现入口，但当前公开仓库没有附带对应模型权重、完整训练数据与同口径结果文件，因此不把设计目标或非公开实验数字写成已复现结论。

上述历史规模与结果来自项目实验记录；仓库公开的是代码、审计摘要和人工编写样例，不分发受许可约束的原始数据、API轨迹或模型权重。

## 已公开实现

| 模块 | 功能 |
|---|---|
| `src/data_prep_sft.py` | 构建 SFT 数据并隔离 FinEval dev/val |
| `src/data_prep_prm.py` | 生成、回填与标注过程监督轨迹 |
| `src/audit_dataset.py` | 检查重复、标签分布和评测泄漏 |
| `src/prm_train.py` | 训练“错误/部分正确/正确”LoRA 三分类 PRM |
| `src/finance_dsl.py` | 以受限 DSL 验证常见金融计算，不执行任意 Python |
| `src/calc_verifier.py` | 从显式算式中发现 Rule-Judge 冲突 |
| `src/reward_finance.py` | 答案、格式、规则与 PRM 的组合奖励 |
| `src/grpo_train.py` | 基于 TRL 的 GRPO 训练入口 |
| `src/best_of_n.py` | 采样 N 个候选并使用 PRM 重排 |
| `src/statistical_analysis.py` | McNemar 精确检验与配对 Bootstrap |

## 快速开始

### CPU：测试确定性组件

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-cpu.txt
python -m unittest discover -s tests -v
python src/finance_dsl.py --expression \
  "compound_interest(principal=10000, rate=5%, periods=2) = 11025"
```

### GPU：训练链路

完整 GPU 依赖与版本说明见 `configs/requirements.txt`。准备有合法使用权的数据后：

```bash
python src/data_prep_sft.py --task fineval fin_corpus template --output_dir data/sft
python src/data_prep_prm.py --mode grpo_prompts --trajectories data/prm/finance_prm.jsonl
python src/prm_train.py --data data/prm/finance_prm.jsonl \
  --base_model Qwen/Qwen2.5-7B-Instruct
python src/grpo_train.py --base_model Qwen/Qwen2.5-7B-Instruct \
  --sft_adapter results/sft/final --prm_model results/prm/final \
  --prompt_file data/grpo/prompts.jsonl
```

Best-of-N 示例：

```bash
python src/best_of_n.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --model results/grpo/final \
  --prm_model results/prm/final \
  --test_file data/eval/finance_reasoning.jsonl \
  --n 8 --output results/best_of_n.json
```

## 数据与复现边界

本仓库不分发 FinEval、FinCorpus 原始题目、完整衍生训练集、API 生成轨迹或模型权重。`data/examples/` 仅包含人工编写的格式样例。请从数据集官方渠道取得授权后运行构建脚本，并保持训练集与最终评测集隔离。

推荐评测协议：

1. 按 `question_id` 或 `trajectory_id` 分组切分 PRM 数据，禁止同题跨集合。
2. Base、SFT、GRPO 使用同一 FinEval-val 协议；报告分子、分母与解码参数。
3. 多随机种子结果报告均值与标准差；逐题比较使用 McNemar 或配对 Bootstrap。
4. 规则验证器只覆盖可解析算式，不把“无法解析”等价为“正确”。

更多背景见 [PROJECT_FROM_ZERO.md](PROJECT_FROM_ZERO.md)，研究边界见 [reports/RESEARCH_REPORT.md](reports/RESEARCH_REPORT.md)。

## License

代码采用 MIT License。第三方模型、数据集和生成内容仍受各自条款约束，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
