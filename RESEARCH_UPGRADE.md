# Fin-R1 研究版升级说明

## 已实现

- 上下文 PRM：输入包含问题、历史步骤和当前步骤。
- 无泄漏切分：PRM 按题目/轨迹分组切分，模型选择指标改为 macro-F1。
- PRM 指标：accuracy、balanced accuracy、macro-F1、Brier、ECE。
- Hybrid Reward：答案 0.25、过程 0.55、规则验算 0.10、格式 0.10，并减去投机惩罚。
- 过程聚合：用 bottom-k 步骤分数替代简单平均，降低冗余正确步骤的稀释效应。
- 公平 Best-of-N：greedy、sampled-first、random rerank、PRM rerank、Oracle pass@N。
- 统计分析：paired bootstrap 95% CI 与 McNemar exact test。
- 数据审计：统计重复、标签分布、训练—评测重叠和评测集指纹。
- 规则指标修正：伪正确率分母改为全部 Judge 判正的可验步骤。

## 零重训使用方式（当前推荐）

不需要重新租卡，也不要把新版方法写成已经取得提升。保留现有 checkpoint 和旧实验数字，先完成以下 CPU 工作：

1. `python src/audit_dataset.py --fail_on_eval_overlap`
2. 运行 `src/calc_verifier.py` 和 `src/build_verified_prm.py`，整理 Judge 漏判/误杀案例。
3. 从现有逐题输出中制作错误类型表；若没有逐题输出，就只使用仓库已经保存的数据分析结果。
4. 把上下文 PRM、Hybrid Reward、严格消融明确标为“已实现的后续方案，尚未复跑”。

只有在以后获得免费 GPU、学校算力或面试明确要求论文级实验时，才考虑重新训练 PRM/GRPO。`configs/experiments.yaml` 是届时的实验计划，不是当前交付要求。

当前自研评测集只有 48 题。在不追加 GPU 预算的前提下，可以人工扩充和校验题目，但在完成之前应把结果称为探索性结果。

已生成 `reports/assisted_review_50.csv` 完成50条模型辅助初审；其中13条仍需专业复核。该文件不是人工金标。
