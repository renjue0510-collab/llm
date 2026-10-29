"""pass@k curves with vLLM, one process per GPU (data parallel, no cross-GPU traffic).

    python rlvr/passk_eval.py --model /nfs/models/Qwen3-1.7B-Base \
        --data /nfs/data/rlvr/math500.parquet --n 256 --out /nfs/eval/base-1.7b/math500

Writes <out>/summary.json (pass@k for k = 1, 2, 4, ..., n, unbiased estimator) and
<out>/per_problem.jsonl (correct count per problem). --save-responses also keeps every sample
(needed later for the strategy / hidden-state analysis). A verl checkpoint must first be merged
to HF format: python -m verl.model_merger merge --backend fsdp --local_dir <ckpt>/actor --target_dir <hf>
"""
import argparse
import gzip
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reward import is_correct  # noqa: E402


def pass_at_k(n, c, k):
    """Unbiased estimator 1 - C(n-c, k) / C(n, k) (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def summarize(per_problem, n):
    ks = [2**i for i in range(int(math.log2(n)) + 1)]
    if ks[-1] != n:
        ks.append(n)
    return {
        "num_problems": len(per_problem),
        "n": n,
        "pass@k": {k: sum(pass_at_k(n, p["correct"], k) for p in per_problem) / len(per_problem) for k in ks},
        "frac_solved_by_none": sum(p["correct"] == 0 for p in per_problem) / len(per_problem),
    }


def worker(args, shard, num_shards):
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(args.data).iloc[shard::num_shards]
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template(list(p), add_generation_prompt=True, tokenize=False) for p in df["prompt"]]
    llm = LLM(args.model, dtype="bfloat16", gpu_memory_utilization=0.85, seed=args.seed + shard,
              max_model_len=args.max_prompt + args.max_tokens, enable_prefix_caching=True)
    params = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                            max_tokens=args.max_tokens, seed=args.seed + shard)
    outs = llm.generate(prompts, params)

    rows, resp_f = [], None
    if args.save_responses:
        resp_f = gzip.open(f"{args.out}/responses.shard{shard}.jsonl.gz", "wt")
    for (_, r), out in zip(df.iterrows(), outs):
        gt = r["reward_model"]["ground_truth"]
        flags = [is_correct(o.text, gt) for o in out.outputs]
        lens = [len(o.token_ids) for o in out.outputs]
        rows.append({"data_source": r["data_source"], "index": int(r["extra_info"]["index"]),
                     "ground_truth": gt, "correct": sum(flags), "mean_len": sum(lens) / len(lens),
                     "truncated": sum(o.finish_reason == "length" for o in out.outputs)})
        if resp_f:
            resp_f.write(json.dumps({"index": rows[-1]["index"], "ground_truth": gt,
                                     "responses": [o.text for o in out.outputs], "correct": flags}) + "\n")
    if resp_f:
        resp_f.close()
    with open(f"{args.out}/per_problem.shard{shard}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="a parquet from prepare_data.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-prompt", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--save-responses", action="store_true")
    ap.add_argument("--shard", type=int, default=-1, help=argparse.SUPPRESS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.shard >= 0:
        return worker(args, args.shard, args.num_gpus)

    procs = []
    for g in range(args.num_gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        procs.append(subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--shard", str(g)], env=env))
    if any(p.wait() for p in procs):
        sys.exit("a shard failed; see its log above")

    per_problem = []
    for g in range(args.num_gpus):
        path = f"{args.out}/per_problem.shard{g}.jsonl"
        with open(path) as f:
            per_problem += [json.loads(line) for line in f]
        os.remove(path)
    per_problem.sort(key=lambda p: p["index"])
    with open(f"{args.out}/per_problem.jsonl", "w") as f:
        for p in per_problem:
            f.write(json.dumps(p) + "\n")
    summary = summarize(per_problem, args.n)
    summary.update(model=args.model, data=args.data, temperature=args.temperature, max_tokens=args.max_tokens,
                   mean_len=sum(p["mean_len"] for p in per_problem) / len(per_problem),
                   truncated_frac=sum(p["truncated"] for p in per_problem) / (args.n * len(per_problem)))
    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["pass@k"], indent=2))


if __name__ == "__main__":
    main()
