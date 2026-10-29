"""Build verl-format parquet files: DAPO-Math-17k (deduplicated) for training, and MATH-500,
AIME 2024, AIME 2025, AMC 2023 for evaluation. All use the same \\boxed{} prompt and the same
reward (rlvr/reward.py), so training and eval numbers are comparable.

    python rlvr/prepare_data.py --out /nfs/data/rlvr

Needs `datasets` and access to huggingface.co (or HF_ENDPOINT pointing at a mirror).
"""
import argparse
import os

import pandas as pd
from datasets import load_dataset

PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
DAPO_HEAD = (
    "Solve the following math problem step by step. The last line of your response should be of "
    "the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
DAPO_TAIL = '\n\nRemember to put your answer on its own line after "Answer:".'

# name -> (hf path, config, split, problem column, answer column)
EVAL_SETS = {
    "math500": ("HuggingFaceH4/MATH-500", "default", "test", "problem", "answer"),
    "aime24": ("HuggingFaceH4/aime_2024", "default", "train", "problem", "answer"),
    "aime25": ("yentinglin/aime_2025", "default", "train", "problem", "answer"),
    "amc23": ("math-ai/amc23", "default", "test", "question", "answer"),
}


def row(source, split, idx, problem, answer):
    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": PROMPT.format(problem=problem.strip())}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer).strip()},
        "extra_info": {"split": split, "index": idx, "problem": problem.strip()},
    }


def strip_dapo(text):
    assert text.startswith(DAPO_HEAD) and text.endswith(DAPO_TAIL), text[:200]
    return text[len(DAPO_HEAD) : -len(DAPO_TAIL)]


def build_train(limit):
    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    seen, rows = set(), []
    for ex in ds:
        problem = strip_dapo(ex["prompt"][0]["content"])
        if problem in seen:  # the HF release repeats each problem ~100 times
            continue
        seen.add(problem)
        rows.append(row("dapo_math", "train", len(rows), problem, ex["reward_model"]["ground_truth"]))
        if limit and len(rows) >= limit:
            break
    return rows


def build_eval(name):
    path, config, split, pcol, acol = EVAL_SETS[name]
    ds = load_dataset(path, config, split=split)
    return [row(name, "test", i, ex[pcol], ex[acol]) for i, ex in enumerate(ds)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-limit", type=int, default=0, help="keep only the first N unique problems (0 = all)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train = build_train(args.train_limit)
    pd.DataFrame(train).to_parquet(f"{args.out}/train.parquet")
    print(f"train: {len(train)} unique problems")

    all_eval = []
    for name in EVAL_SETS:
        rows = build_eval(name)
        pd.DataFrame(rows).to_parquet(f"{args.out}/{name}.parquet")
        all_eval += rows
        print(f"{name}: {len(rows)}")
    # Small, quick-to-run set for in-training validation: AIME 24/25 + AMC 23.
    val = [r for r in all_eval if r["data_source"] != "math500"]
    pd.DataFrame(val).to_parquet(f"{args.out}/val.parquet")
    print(f"val (aime24+aime25+amc23): {len(val)}")


if __name__ == "__main__":
    main()
