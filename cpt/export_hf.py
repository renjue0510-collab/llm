"""Convert a train_fsdp.py checkpoint (DCP format) back into a Hugging Face model directory.

Runs on one process with no GPU needed; for 32B it needs about 130 GB of RAM.

  python cpt/export_hf.py --base /models/Qwen-32B-base \
      --ckpt /nfs/ckpt/cpt-32b/step-0001000 --out /nfs/models/qwen-32b-cpt-1000
"""

import argparse

import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="base HF model directory (config and tokenizer)")
    ap.add_argument("--ckpt", required=True, help="step-XXXXXXX checkpoint directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = ap.parse_args()

    config = AutoConfig.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32, trust_remote_code=True)
    # Only the model half of the checkpoint is read; optimizer state is skipped.
    state = {"model": model.state_dict()}
    dcp.load(state, checkpoint_id=args.ckpt, no_dist=True)
    model.load_state_dict(state["model"])

    if args.dtype == "bf16":
        model = model.to(torch.bfloat16)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base, trust_remote_code=True).save_pretrained(args.out)
    print(f"exported {args.ckpt} -> {args.out}")


if __name__ == "__main__":
    main()
