#!/usr/bin/env python3
"""
Fin-R1 SFT 数据准备脚本
功能：从 HuggingFace 下载金融数据集，清洗并格式化为 Qwen2.5 指令格式

数据源：
  1. FinEval dev — 中文金融考试/选择题（**只训 dev，val/test 留给评测，避免数据泄漏**）
  2. FinCorpus fin_exam — 金融考试题（只保留带答案+解析的考试题，非考试文本直接丢弃）
  3. 自研金融指令对 — 财报问答、研报分析、合规审查
输出：data/sft/finance_sft_train.jsonl / finance_sft_val.jsonl

使用方式：
  python src/data_prep_sft.py \
    --task fin_corpus \
    --output_dir data/sft \
    --train_ratio 0.95
"""

import argparse
import glob
import gzip
import json
import os
import random
import re
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------- 金融领域 System Prompt 模板 ----------
FINANCE_SYSTEM_PROMPT = (
    "你是一位专业的金融分析师，精通财务报表分析、宏观经济研究和合规审查。"
    "请基于金融专业知识，给出准确、严谨、步骤清晰的回答。"
    "如果涉及计算，请展示完整的推导过程。"
)

# ---------- Qwen2.5 ChatML 格式 ----------
# format_qwen 直接构造 messages dict，ChatML 模板由 LLaMA-Factory 的 template: qwen 处理


def format_qwen(instruction: str, output: str) -> dict:
    """格式化为 Qwen2.5 兼容的训练数据"""
    return {
        "messages": [
            {"role": "system", "content": FINANCE_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]
    }


def load_fineval(split: str = "dev") -> list[dict]:
    """
    加载 FinEval 中文金融评测数据
    来源：huggingface.co/datasets/SUFE-AIFLM-Lab/FinEval

    ⚠️ 只加载 dev 作为训练数据；val/test 必须留给评测脚本，否则验收数字毫无意义。

    FinEval 的 CSV 列名不统一（dev 有 explanation，val 有 Unnamed: 7，test 无答案），
    所以用 pandas 逐个读 CSV 并统一列名。

    格式化策略：
    - 有 explanation：输出"正确答案是 X。[选项全文]。\n\n解析：[explanation]"
    - 无 explanation：输出"正确答案是 X。[选项全文]。"（至少学到答案内容，不只一个字母）
    """
    print(f"[FinEval] 下载中（split={split}）...")
    records = []

    import huggingface_hub
    zip_path = huggingface_hub.hf_hub_download(
        repo_id="SUFE-AIFLM-Lab/FinEval",
        filename="FinEval.zip",
        repo_type="dataset",
    )

    with zipfile.ZipFile(zip_path) as z:
        csv_names = [n for n in z.namelist() if n.startswith(f"{split}/") and n.endswith(".csv")]
        print(f"[FinEval] 共 {len(csv_names)} 个 {split} CSV 文件")

        for name in tqdm(csv_names, desc=f"FinEval {split}"):
            with z.open(name) as f:
                df = pd.read_csv(f)

            if "Unnamed: 7" in df.columns:
                df = df.rename(columns={"Unnamed: 7": "explanation"})

            for _, row in df.iterrows():
                question = str(row.get("question", "")).strip()
                answer = str(row.get("answer", "")).strip()

                if not question or not answer:
                    continue

                # 收集选项 A/B/C/D（兼容部分科目有 E 选项）
                options_map = {}
                options_text = []
                for opt in ["A", "B", "C", "D", "E"]:
                    val = row.get(opt, "")
                    if pd.notna(val) and str(val).strip():
                        options_map[opt] = str(val).strip()
                        options_text.append(f"{opt}. {options_map[opt]}")
                if options_text:
                    question = question + "\n" + "\n".join(options_text)

                explanation = str(row.get("explanation", "")).strip()

                # 构造完整答案
                answer_text = options_map.get(answer, answer)
                if explanation and explanation != "nan" and len(explanation) > 5:
                    full_answer = f"正确答案是 {answer}. {answer_text}\n\n解析：{explanation}"
                else:
                    full_answer = f"正确答案是 {answer}. {answer_text}\n\n该题考查金融专业知识，{answer}选项为正确表述。"

                records.append(format_qwen(question, full_answer))

    print(f"[FinEval] 完成，{len(records)} 条")
    return records


def load_fin_corpus(min_length: int = 50, max_length: int = 10000) -> list[dict]:
    """
    加载 FinCorpus 金融考试数据
    来源：huggingface.co/datasets/Duxiaoman-DI/FinCorpus

    FinCorpus 用自定义加载脚本，datasets 库已弃用，所以直接读缓存里的 .gz 文件。
    数据格式（fin_exam.jsonl.gz）：
      {"text": "题目\\nA、选项\\nB、选项\\n答案：X\\n分析解释：...", "meta": {"source": "fin_exam"}}

    ⚠️ 只保留带"答案 + 分析解释"的考试题；其余（公告/文章）没有标准答案，
       原来的硬编码假摘要会教模型瞎编，已移除，非考试文本直接丢弃。
    """
    print("[FinCorpus] 加载中...")
    records = []
    max_count = 30000

    # 找缓存里的 .gz 文件
    cache_pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--Duxiaoman-DI--FinCorpus/snapshots/*/data/*.gz"
    )
    gz_files = glob.glob(cache_pattern)

    if not gz_files:
        # Windows 路径兼容
        cache_pattern = os.path.expanduser(
            "~/.cache/huggingface/hub/datasets--Duxiaoman-DI--FinCorpus/blobs/*"
        )
        all_blobs = glob.glob(cache_pattern)
        # 只取大于 10MB 的 blob（排除小的 metadata 文件）
        gz_files = [f for f in all_blobs if os.path.getsize(f) > 10 * 1024 * 1024]

    if not gz_files:
        print("[FinCorpus] 缓存中未找到 .gz 文件，跳过")
        print("  提示：请先用 datasets 库下载 FinCorpus（Duxiaoman-DI/FinCorpus）生成缓存。")
        return records

    print(f"[FinCorpus] 找到 {len(gz_files)} 个数据文件")
    gz_files.sort(key=lambda f: 0 if "exam" in f.lower() else 1)

    count = 0
    for gz_path in gz_files:
        if count >= max_count:
            break

        fname = os.path.basename(gz_path)
        print(f"[FinCorpus] 读取 {fname}...")

        try:
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                for line in tqdm(f, desc=f"FinCorpus {fname[:20]}"):
                    if count >= max_count:
                        break

                    try:
                        item = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue

                    text = item.get("text", "")
                    if not text or len(text) < min_length or len(text) > max_length:
                        continue

                    # 只保留考试题（有答案标记），非考试文本直接丢弃
                    answer_match = re.search(r"答案[：:]\s*([A-E]+(?:[、,，]?[A-E])*)", text)
                    if not answer_match:
                        continue
                    # 提取完整答案字母（多选题如"A、B、C"不能只取 A）
                    answer_letters = "".join(re.findall(r"[A-E]", answer_match.group(1)))
                    if not answer_letters:
                        continue
                    explanation_match = re.search(r"分析解释[：:]\s*(.+)", text, re.DOTALL)

                    # 用正则匹配位置切题，避免题面里出现"答案"字样导致截断
                    answer_pos = answer_match.start()
                    question_part = text[:answer_pos].strip()
                    answer_display = "、".join(answer_letters)  # 展示格式 A、B、C

                    if explanation_match:
                        explanation = explanation_match.group(1).strip()
                        full_answer = f"正确答案是 {answer_display}\n\n解析：{explanation}"
                    else:
                        full_answer = f"正确答案是 {answer_display}"

                    records.append(format_qwen(question_part, full_answer))
                    count += 1

        except Exception as e:
            print(f"[FinCorpus] 读取 {fname} 失败: {e}")
            continue

    print(f"[FinCorpus] 完成，{len(records)} 条")
    return records


def generate_template_data(n: int = 5000) -> list[dict]:
    """
    模板增强：用金融知识模板生成额外的 SFT 数据
    覆盖：财务比率、估值方法、风险管理、货币政策、投资分析等

    ⚠️ 模板只有 ~20 个，随机前缀只会产生少量变体。
       main() 里已改为按"问题+答案"去重，模板数据会被压缩到几十条真实变体，
       避免模型背诵重复段落。
    """
    print("[模板增强] 生成金融指令数据...")
    templates = [
        ("请解释什么是流动比率，并说明其计算方法和判断标准。",
         "流动比率是衡量企业短期偿债能力的财务指标。\n\n计算方法：流动比率 = 流动资产 ÷ 流动负债\n\n判断标准：\n- 流动比率 > 2：短期偿债能力较强\n- 流动比率 1~2：偿债能力一般，需关注\n- 流动比率 < 1：存在短期偿债风险\n\n该指标反映企业用短期资产偿还短期负债的能力，是银行信贷评估的重要参考。"),

        ("请解释什么是速动比率，与流动比率有什么区别？",
         "速动比率是流动比率的改良版，排除了存货的影响。\n\n计算方法：速动比率 = （流动资产 - 存货）÷ 流动负债\n\n与流动比率的区别：\n1. 流动比率包含存货，速动比率不包含\n2. 存货变现速度较慢，排除后更能反映即时偿债能力\n3. 速动比率 > 1 通常被认为偿债能力良好\n\n在银行信贷评估中，速动比率比流动比率更保守、更严格。"),

        ("请解释什么是资产负债率，并说明其财务意义。",
         "资产负债率反映企业负债占总资产的比例。\n\n计算方法：资产负债率 = 总负债 ÷ 总资产 × 100%\n\n财务意义：\n- 比率越高，财务杠杆越大，偿债压力越大\n- 比率越低，财务结构越保守，但可能资本利用率不高\n- 一般行业正常范围 40%~60%\n- 银行授信通常要求资产负债率不超过 70%\n\n该指标是评估企业长期偿债能力和财务风险的关键指标。"),

        ("请解释什么是净资产收益率（ROE），杜邦分析如何拆解？",
         "净资产收益率（ROE）反映股东权益的收益水平。\n\n计算方法：ROE = 净利润 ÷ 平均股东权益 × 100%\n\n杜邦分析三因子拆解：\nROE = 净利率 × 总资产周转率 × 权益乘数\n\n- 净利率 = 净利润 ÷ 营业收入（反映盈利能力）\n- 总资产周转率 = 营业收入 ÷ 总资产（反映运营效率）\n- 权益乘数 = 总资产 ÷ 股东权益（反映财务杠杆）\n\n杜邦分析将 ROE 拆解为盈利、运营、杠杆三个维度，帮助定位业绩驱动因素。"),

        ("请解释什么是市盈率（P/E），如何用它做股票估值？",
         "市盈率是股票估值最常用的指标之一。\n\n计算方法：P/E = 股价 ÷ 每股收益（EPS）\n\n含义：投资者愿意为每 1 元利润支付多少元。\n\n使用方法：\n1. 横向比较：与同行业公司 P/E 对比\n2. 纵向比较：与公司历史 P/E 对比\n3. 合理范围：A 股市场一般 10~30 倍\n4. P/E 低可能被低估，也可能业绩差\n\n局限性：\n- 不适用于亏损企业（EPS 为负）\n- 不适用于周期性行业\n- 未考虑成长性，需结合 PEG 指标"),

        ("请解释 CAPM 模型的公式和含义。",
         "CAPM（资本资产定价模型）描述风险与预期收益的关系。\n\n公式：E(Ri) = Rf + βi × [E(Rm) - Rf]\n\n参数含义：\n- E(Ri)：资产 i 的预期收益率\n- Rf：无风险利率（通常用国债收益率）\n- βi：资产 i 的系统性风险系数\n- E(Rm)：市场组合预期收益率\n- [E(Rm) - Rf]：市场风险溢价\n\n含义：投资者因承担系统性风险而获得额外收益。β > 1 表示比市场波动更大，β < 1 表示比市场更稳定。\n\n应用：用于股票估值、资本成本计算、投资组合管理。"),

        ("请解释什么是久期，它在债券风险管理中有什么作用？",
         "久期衡量债券价格对利率变化的敏感度。\n\n计算方法：Macaulay 久期 = Σ(t × CFt / (1+y)^t) / P\n\n其中 t 为期数，CFt 为每期现金流，y 为到期收益率，P 为债券价格。\n\n修正久期 = Macaulay 久期 / (1 + y)\n\n作用：\n1. 利率上升 1%，债券价格下跌约（修正久期 × 1%）\n2. 久期越长，利率风险越大\n3. 银行使用久期缺口管理利率风险\n4. 资产久期 > 负债久期，利率上升时净值下降\n\n久期是固定收益投资和银行利率风险管理的核心工具。"),

        ("请解释什么是 VaR（在险价值），如何计算？",
         "VaR 是衡量投资组合在给定置信水平和时间内最大可能损失的指标。\n\n定义：在 (1-α) 置信水平下，未来 N 天内损失不超过 VaR。\n\n常用 99% 置信水平，即 99% 的概率损失不超过 VaR。\n\n计算方法：\n1. 历史模拟法：用历史收益率分布直接计算\n2. 方差-协方差法：假设正态分布，VaR = μ + zα × σ\n3. 蒙特卡洛模拟：随机生成未来价格路径\n\n巴塞尔协议要求银行用 VaR 计量市场风险资本。\n\n局限性：\n- 只给出损失上限，不描述尾部\n- 假设可能不符合实际（如正态分布）\n- 需配合压力测试使用"),

        ("请解释什么是 M2 货币供应量，与 M1 有什么区别？",
         "M2 是广义货币供应量，反映社会总购买力。\n\n中国货币供应量分层：\n- M0 = 流通中的现金\n- M1 = M0 + 活期存款（狭义货币）\n- M2 = M1 + 定期存款 + 证券保证金等（广义货币）\n\n区别：\n1. M1 反映现实购买力，流动性最强\n2. M2 反映现实 + 潜在购买力\n3. M1 增速 > M2 增速：活期化，经济活跃\n4. M2 增速 > M1 增速：定期化，经济趋缓\n\n央行通过调节 M2 增速实施货币政策，M2 是观察流动性和通胀的重要指标。"),

        ("请解释 Basel III 对银行资本充足率的要求。",
         "Basel III 是巴塞尔银行监管委员会发布的国际银行监管标准。\n\n核心资本要求：\n1. 核心一级资本充足率 ≥ 4.5%\n2. 一级资本充足率 ≥ 6%\n3. 总资本充足率 ≥ 8%\n\n额外要求：\n- 资本留存缓冲 2.5%（核心一级）\n- 逆周期缓冲 0~2.5%\n- 系统重要性银行附加 1~3.5%\n\n合计：系统重要性银行核心一级资本充足率需达 8.5%~11%\n\n引入杠杆率 ≥ 3% 作为风险加权资本的补充。\n\n流动性要求：\n- 流动性覆盖率（LCR）≥ 100%\n- 净稳定资金比率（NSFR）≥ 100%\n\n中国银保监会在此基础上略有调整。"),

        ("请解释什么是净现值（NPV），如何用 NPV 做投资决策？",
         "净现值（NPV）是项目未来现金流的现值减去初始投资。\n\n计算方法：NPV = Σ CFt / (1+r)^t - 初始投资\n\n其中 CFt 为第 t 期现金流，r 为贴现率。\n\n决策规则：\n- NPV > 0：项目创造价值，应该投资\n- NPV = 0：项目不创造也不损毁价值\n- NPV < 0：项目损毁价值，不应该投资\n\n优点：\n1. 考虑货币时间价值\n2. 考虑全部现金流\n3. 与股东价值最大化目标一致\n\n缺点：\n1. 贴现率选取主观\n2. 对长期项目不确定性较大\n3. 不适合互斥项目规模差异大的比较"),

        ("请解释什么是 Black-Scholes 期权定价模型。",
         "Black-Scholes 模型是欧式期权定价的经典模型。\n\n看涨期权公式：\nC = S × N(d1) - K × e^(-rT) × N(d2)\n\n其中：\nd1 = [ln(S/K) + (r + σ²/2) × T] / (σ × √T)\nd2 = d1 - σ × √T\n\n参数：\n- S：标的资产现价\n- K：行权价\n- r：无风险利率\n- T：到期时间（年）\n- σ：波动率\n- N()：标准正态分布累积概率\n\n假设条件：\n1. 标的资产价格服从对数正态分布\n2. 无风险利率恒定\n3. 无分红\n4. 无交易成本\n5. 可以连续交易\n\n该模型是金融工程的里程碑，广泛应用于期权定价和风险管理。"),

        ("请解释什么是信用违约互换（CDS）。",
         "CDS 是信用衍生品，相当于对债券违约买保险。\n\n机制：\n- 买方定期支付保费\n- 卖方承诺：如果参考债券违约，赔偿买方损失\n- 不持有债券也可以买 CDS（裸卖空）\n\n定价因素：\n1. 参考实体的信用评级\n2. 债券到期时间\n3. 市场违约概率\n4. 回收率假设\n\n应用：\n1. 对冲信用风险\n2. 做空信用\n3. 套利交易\n\n2008 年金融危机中 CDS 被大量使用，雷曼兄弟和 AIG 的 CDS 头寸是危机放大器。"),

        ("请解释什么是夏普比率。",
         "夏普比率衡量每单位风险获得的超额收益。\n\n计算方法：夏普比率 = (Rp - Rf) / σp\n\n其中 Rp 为组合收益率，Rf 为无风险利率，σp 为组合标准差。\n\n含义：\n- 夏普比率 > 1：良好\n- 夏普比率 > 2：优秀\n- 夏普比率 < 0：不如无风险投资\n\n应用：\n1. 评价基金经理的风险调整收益\n2. 比较不同策略的风险效率\n3. 优化投资组合\n\n局限性：\n1. 假设收益正态分布\n2. 标准差衡量波动而非真正的下行风险\n3. 对尾部风险不敏感\n\nSortino 比率用下行偏差替代标准差，更适合衡量下行风险。"),

        ("请解释什么是杜邦分析法。",
         "杜邦分析法将 ROE 拆解为三个驱动因素。\n\n公式：ROE = 净利率 × 总资产周转率 × 权益乘数\n\n拆解：\n1. 净利率 = 净利润 / 营业收入 → 盈利能力\n2. 总资产周转率 = 营业收入 / 总资产 → 运营效率\n3. 权益乘数 = 总资产 / 股东权益 → 财务杠杆\n\n5 因子杜邦：\nROE = 税负担 × 利息负担 × 经营利润率 × 资产周转率 × 权益乘数\n\n应用：\n1. 定位 ROE 变动原因\n2. 比较同业盈利模式差异\n3. 评估盈利质量可持续性\n\n例子：\n- 高净利率低周转：奢侈品行业\n- 低净利率高周转：零售行业\n- 高杠杆：银行业\n\n杜邦分析帮助投资者看清 ROE 的真正驱动因素，避免被单一指标误导。"),

        ("请解释什么是 WACC（加权平均资本成本）。",
         "WACC 是企业全部资本的加权平均成本。\n\n公式：WACC = E/(D+E) × Re + D/(D+E) × Rd × (1-t)\n\n其中：\n- E：股权市值\n- D：债务市值\n- Re：股权成本\n- Rd：债务成本\n- t：税率\n\n股权成本 Re 通常用 CAPM 计算：\nRe = Rf + β × (Rm - Rf)\n\n债务成本 Rd 用税后成本：Rd × (1-t)\n（利息可抵税）\n\n应用：\n1. 作为 DCF 估值的贴现率\n2. 评估投资项目的门槛收益率\n3. 评估资本结构优化\n\n注意：\n1. 用市场价值而非账面价值\n2. 考虑不同来源资本的成本差异\n3. 项目风险不同时需要调整 WACC"),

        ("请解释什么是久期缺口，银行如何用它管理利率风险？",
         "久期缺口衡量银行资产和负债的利率敏感度差异。\n\n计算方法：\n久期缺口 = 资产久期 - 负债久期 × (负债/资产)\n\n管理策略：\n1. 久期缺口 > 0：利率上升时净值下降\n2. 久期缺口 < 0：利率下降时净值下降\n3. 久期缺口 = 0：利率中性\n\n银行操作：\n- 预期利率上升 → 缩短资产久期 / 延长负债久期\n- 预期利率下降 → 延长资产久期 / 缩短负债久期\n- 资产负债匹配管理（ALM）\n\n净值变动估算：\nΔ净值 ≈ -久期缺口 × Δ利率 × 总资产\n\n久期缺口管理是银行 ALM 部门的核心工具。"),

        ("请解释什么是回购协议（Repo）。",
         "回购协议是短期抵押借款，本质是「卖出+回购」。\n\n机制：\n1. 借款方将债券「卖」给贷款方\n2. 约定未来以更高价格「买回」\n3. 差价 = 利息成本\n\n分类：\n- 正回购：借入资金（卖出债券）\n- 逆回购：借出资金（买入债券）\n\n期限：\n- 隔夜回购（O/N）最常见\n- 定期回购：1周~3个月\n- 开放回购：无固定到期\n\n中国回购市场：\n- 上交所质押式回购（GC001/GC007）\n- 银行间 R-001/R-007\n- 利率反映银行间流动性松紧\n\n应用：\n1. 银行流动性管理\n2. 央行公开市场操作工具\n3. 债券杠杆套息\n\n2008 年次贷危机中，回购市场的「折价率」飙升反映了流动性枯竭。"),

        ("请解释什么是 IPO 询价机制。",
         "IPO 询价是中国 A 股新股发行的定价机制。\n\n流程：\n1. 初步询价：向机构投资者询价，剔除最高 10% 报价\n2. 确定发行价：加权平均后定价\n3. 网下配售：按报价和数量配售\n4. 网上申购：个人投资者按发行价申购\n\n2021 年注册制改革后：\n- 主板：间接定价（参考可比公司）\n- 创业板/科创板：直接定价或询价\n\n询价对象：\n- 公募基金、社保、QFII 等\n- 个人投资者暂不能参与询价\n\n改革方向：\n- 引入更多类型投资者\n- 加强报价约束\n- 防止「抱团报价」\n\n询价机制的核心是市场化定价，取代了原核准制下的 23 倍市盈率上限。"),
    ]

    records = []
    for _ in range(n):
        q, a = random.choice(templates)
        # 加一点变化避免完全重复
        if random.random() > 0.5:
            q = q.replace("请解释", "请详细解释")
        records.append(format_qwen(q, a))

    print(f"[模板增强] 完成，{len(records)} 条（去重后大幅减少）")
    return records


def load_self_generated(data_dir: str) -> list[dict]:
    """
    加载自研金融指令对（财报问答、研报分析、合规审查）
    格式：data/sft/self_generated/*.jsonl
    每行：{"instruction": "...", "output": "..."}
    """
    records = []
    dir_path = Path(data_dir)

    if not dir_path.exists():
        print(f"[自研数据] 目录不存在，跳过: {data_dir}")
        return records

    for fname in dir_path.glob("*.jsonl"):
        with open(fname, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                records.append(format_qwen(item["instruction"], item["output"]))

    print(f"[自研数据] 完成，{len(records)} 条")
    return records


def save_jsonl(records: list[dict], filepath: str) -> None:
    """保存为 JSONL"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"保存: {filepath} ({len(records)} 条)")


def main():
    parser = argparse.ArgumentParser(description="Fin-R1 SFT 数据准备")
    parser.add_argument(
        "--task",
        nargs="+",
        default=["fineval", "fin_corpus", "template"],
        choices=["fineval", "fin_corpus", "self_generated", "template", "all"],
        help="数据源选择",
    )
    parser.add_argument("--output_dir", default="data/sft", help="输出目录")
    parser.add_argument("--train_ratio", type=float, default=0.95, help="训练集比例")
    parser.add_argument("--self_generated_dir", default="data/sft/self_generated", help="自研数据目录")
    parser.add_argument("--template_n", type=int, default=5000, help="模板增强条数")
    parser.add_argument("--fineval_split", default="dev", help="FinEval 训练用 split（默认 dev，防泄漏）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    all_records: list[dict] = []

    # 按数据源加载
    tasks = args.task if "all" not in args.task else ["fineval", "fin_corpus", "template", "self_generated"]

    if "fineval" in tasks:
        all_records.extend(load_fineval(split=args.fineval_split))

    if "fin_corpus" in tasks:
        all_records.extend(load_fin_corpus())

    if "template" in tasks:
        all_records.extend(generate_template_data(args.template_n))

    if "self_generated" in tasks:
        all_records.extend(load_self_generated(args.self_generated_dir))

    # 去重（按"问题+答案"组合，避免相同答案配不同前缀的模板重复污染）
    seen = set()
    deduped = []
    for rec in all_records:
        key = (rec["messages"][1]["content"], rec["messages"][2]["content"])
        if key not in seen:
            seen.add(key)
            deduped.append(rec)
    print(f"去重: {len(all_records)} → {len(deduped)} 条")

    # 打乱
    random.shuffle(deduped)

    # 切分 train / val
    split_idx = int(len(deduped) * args.train_ratio)
    train_records = deduped[:split_idx]
    val_records = deduped[split_idx:]

    # 保存
    save_jsonl(train_records, os.path.join(args.output_dir, "finance_sft_train.jsonl"))
    save_jsonl(val_records, os.path.join(args.output_dir, "finance_sft_val.jsonl"))

    # 导出 dataset_info.json（LLaMA-Factory 需要；train 和 val 都注册，供 eval_dataset 使用）
    dataset_info = {
        "finance_sft": {
            "file_name": "finance_sft_train.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"},
        },
        "finance_sft_val": {
            "file_name": "finance_sft_val.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"},
        },
    }
    info_path = os.path.join(args.output_dir, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    print(f"dataset_info 保存: {info_path}")

    print(f"\n✅ SFT 数据准备完成！")
    print(f"   训练集: {len(train_records)} 条 → {os.path.join(args.output_dir, 'finance_sft_train.jsonl')}")
    print(f"   验证集: {len(val_records)} 条 → {os.path.join(args.output_dir, 'finance_sft_val.jsonl')}")


if __name__ == "__main__":
    main()
