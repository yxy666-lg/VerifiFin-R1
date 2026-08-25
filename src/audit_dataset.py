#!/usr/bin/env python3
"""在训练前审计 PRM/GRPO/评测数据的重复、标签分布与答案覆盖。"""

import argparse
import hashlib
import json
import re
from collections import Counter


def normalize(text):
    return re.sub(r"\s+", "", str(text)).lower()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def question_of(item):
    text = item.get("question") or item.get("prompt") or ""
    return text.rsplit("\n\n问题：", 1)[-1].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prm", default="data/prm/finance_prm.jsonl")
    parser.add_argument("--grpo", default="data/grpo/prompts.jsonl")
    parser.add_argument("--eval", default="data/eval/finance_reasoning.jsonl")
    parser.add_argument("--fail_on_eval_overlap", action="store_true")
    args = parser.parse_args()
    prm, grpo, evaluation = map(read_jsonl, (args.prm, args.grpo, args.eval))
    sets = {name: {normalize(question_of(x)) for x in rows} for name, rows in
            (("prm", prm), ("grpo", grpo), ("eval", evaluation))}
    labels = Counter(s.get("label") for row in prm for s in row.get("steps", []))
    report = {
        "counts": {"prm": len(prm), "grpo": len(grpo), "eval": len(evaluation)},
        "unique_questions": {k: len(v) for k, v in sets.items()},
        "overlap": {
            "prm_eval": len(sets["prm"] & sets["eval"]),
            "grpo_eval": len(sets["grpo"] & sets["eval"]),
            "prm_grpo": len(sets["prm"] & sets["grpo"]),
        },
        "prm_step_labels": {str(k): v for k, v in sorted(labels.items(), key=lambda x: str(x[0]))},
        "eval_with_answer": sum(bool(x.get("answer")) for x in evaluation),
        "fingerprint": hashlib.sha256("\n".join(sorted(sets["eval"])).encode()).hexdigest()[:16],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_eval_overlap and (report["overlap"]["prm_eval"] or report["overlap"]["grpo_eval"]):
        raise SystemExit("检测到训练数据与评测集重叠")


if __name__ == "__main__":
    main()
