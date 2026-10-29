"""Continued pretraining of a Hugging Face causal LM with FSDP2 on a single multi-GPU node.

Designed for 32B-class models on 8 PCIe GPUs (no NVLink): pure data parallel with
FSDP2 sharding, bf16 compute, full activation checkpointing, and optional CPU
offload of parameters, gradients and optimizer state (--cpu-offload).

Launch with torchrun (see launch.sh):
  torchrun --standalone --nproc_per_node 8 cpt/train_fsdp.py \
      --model /models/Qwen-32B-base --data /data/packed/domain:0.8 /data/packed/general:0.2 \
      --out /nfs/ckpt/cpt-32b --train-tokens 1e9 --cpu-offload
"""

import argparse
import gc
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from transformers import AutoConfig, AutoModelForCausalLM


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF base model directory")
    ap.add_argument("--data", nargs="+", required=True,
                    help="packed data prefixes from pack_data.py, each optionally suffixed with :weight")
    ap.add_argument("--out", required=True, help="checkpoint directory (use shared/NFS storage)")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--micro-batch", type=int, default=4, help="sequences per GPU per micro-step")
    ap.add_argument("--global-batch-tokens", type=float, default=2 ** 21, help="tokens per optimizer step")
    ap.add_argument("--train-tokens", type=float, required=True, help="total tokens to train on")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early after this many steps (smoke tests)")
    ap.add_argument("--lr", type=float, default=2e-5, help="peak LR, ~1/10 of the base model's pretraining peak")
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup-ratio", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--cpu-offload", action="store_true", help="offload params/grads/optimizer to CPU RAM")
    ap.add_argument("--reduce-dtype", choices=["fp32", "bf16"], default="fp32",
                    help="gradient reduce-scatter dtype; bf16 halves PCIe traffic")
    ap.add_argument("--attn", default="sdpa", help="attn_implementation, e.g. sdpa or flash_attention_2")
    ap.add_argument("--save-every", type=int, default=200, help="optimizer steps between checkpoints")
    ap.add_argument("--keep-last", type=int, default=2, help="checkpoints to keep on disk")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--peak-tflops", type=float, default=142.0,
                    help="per-GPU dense bf16 peak for MFU; 142 was measured on an RTX 6000D")
    ap.add_argument("--seed", type=int, default=1234)
    return ap.parse_args()


def log(msg):
    if dist.get_rank() == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class PackedData:
    """Fixed-length windows over packed token files, mixed by weight.

    Every rank walks the same deterministic global order and keeps only its own
    sequences, so the per-source counters are identical everywhere and are all
    the state needed to resume.
    """

    def __init__(self, specs, seq_len, seed):
        self.seq_len = seq_len
        self.sources, weights = [], []
        for spec in specs:
            prefix, _, w = spec.partition(":")
            tokens = np.memmap(prefix + ".bin", dtype=np.uint32, mode="r")
            n_windows = (len(tokens) - 1) // seq_len
            if n_windows <= 0:
                raise ValueError(f"{prefix}.bin is shorter than one sequence")
            perm = np.random.default_rng([seed, len(self.sources)]).permutation(n_windows)
            self.sources.append((prefix, tokens, perm))
            weights.append(float(w) if w else float(n_windows))
        self.weights = np.asarray(weights) / sum(weights)
        self.counts = [0] * len(self.sources)
        self.seed = seed
        self.micro_index = 0

    def describe(self):
        return ", ".join(f"{p} ({len(perm) * self.seq_len / 1e9:.2f}B tok, w={w:.2f})"
                         for (p, _, perm), w in zip(self.sources, self.weights))

    def next(self, rank, world, micro_batch):
        rng = np.random.default_rng([self.seed, 1 << 20, self.micro_index])
        picks = rng.choice(len(self.sources), size=world * micro_batch, p=self.weights)
        self.micro_index += 1
        rows = []
        for j, s in enumerate(picks):
            _, tokens, perm = self.sources[s]
            window = perm[self.counts[s] % len(perm)]
            self.counts[s] += 1
            if j // micro_batch == rank:
                start = int(window) * self.seq_len
                rows.append(np.asarray(tokens[start:start + self.seq_len], dtype=np.int64))
        return torch.from_numpy(np.stack(rows))

    def state(self):
        return {"counts": self.counts, "micro_index": self.micro_index}

    def load_state(self, state):
        self.counts = list(state["counts"])
        self.micro_index = state["micro_index"]


def build_model(args, device):
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.use_cache = False
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, dtype=torch.float32, attn_implementation=args.attn, trust_remote_code=True
        )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    mesh = init_device_mesh(device.type, (dist.get_world_size(),))
    shard_kwargs = {
        "mesh": mesh,
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32 if args.reduce_dtype == "fp32" else torch.bfloat16,
        ),
    }
    if args.cpu_offload:
        shard_kwargs["offload_policy"] = CPUOffloadPolicy()
    for layer in model.model.layers:
        fully_shard(layer, **shard_kwargs)
    fully_shard(model, **shard_kwargs)

    # Sharded parameters live on CPU when offloading; buffers (rotary inv_freq) must be on the GPU.
    model.to_empty(device="cpu" if args.cpu_offload else device)
    return model


def load_base_weights(model, args, device):
    """Load the HF checkpoint on rank 0 and broadcast it into the FSDP shards."""
    rank = dist.get_rank()
    full_sd, ref_buffers = {}, {}
    if rank == 0:
        ref = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True
        )
        full_sd = ref.state_dict()
        ref_buffers = dict(ref.named_buffers())
    set_model_state_dict(
        model, full_sd, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
    )

    # Non-persistent buffers are not in the state dict, so broadcast them separately.
    for name, buf in list(model.named_buffers()):
        if rank == 0:
            t = ref_buffers[name].to(device=device, dtype=buf.dtype)
        else:
            t = torch.empty(buf.shape, dtype=buf.dtype, device=device)
        dist.broadcast(t, src=0)
        mod_name, _, buf_name = name.rpartition(".")
        mod = model.get_submodule(mod_name)
        mod._buffers[buf_name] = t
        if buf_name == "inv_freq" and hasattr(mod, "original_inv_freq"):
            mod.original_inv_freq = t.clone()
    del full_sd, ref_buffers
    gc.collect()


def build_optimizer(model, args):
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )


def lr_at(step, total_steps, args):
    warmup = max(1, int(total_steps * args.warmup_ratio))
    if step < warmup:
        return args.lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def latest_checkpoint(out):
    steps = sorted(int(p.name.split("-")[1]) for p in Path(out).glob("step-*") if (p / "trainer.json").exists())
    return Path(out) / f"step-{steps[-1]:07d}" if steps else None


def save_checkpoint(model, optim, data, step, args):
    path = Path(args.out) / f"step-{step:07d}"
    msd, osd = get_state_dict(model, optim)
    dcp.save({"model": msd, "optim": osd}, checkpoint_id=str(path))
    if dist.get_rank() == 0:
        # trainer.json is written last and marks the checkpoint as complete.
        with open(path / "trainer.json", "w") as f:
            json.dump({"step": step, "data": data.state(), "args": vars(args)}, f, indent=2)
        done = sorted(p for p in Path(args.out).glob("step-*") if (p / "trainer.json").exists())
        for old in done[:-args.keep_last]:
            shutil.rmtree(old, ignore_errors=True)
    dist.barrier()
    log(f"saved checkpoint {path}")


def load_checkpoint(model, optim, data, path):
    msd, osd = get_state_dict(model, optim)
    state = {"model": msd, "optim": osd}
    dcp.load(state, checkpoint_id=str(path))
    set_state_dict(model, optim, model_state_dict=state["model"], optim_state_dict=state["optim"])
    with open(path / "trainer.json") as f:
        meta = json.load(f)
    data.load_state(meta["data"])
    return meta["step"]


def main():
    args = parse_args()
    if torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="cuda:nccl,cpu:gloo", device_id=device)
    else:  # CPU-only smoke tests
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(args.seed)

    seqs_per_micro = world * args.micro_batch
    accum = max(1, round(args.global_batch_tokens / (seqs_per_micro * args.seq_len)))
    tokens_per_step = accum * seqs_per_micro * args.seq_len
    total_steps = math.ceil(args.train_tokens / tokens_per_step)
    data = PackedData(args.data, args.seq_len, args.seed)
    log(f"world={world} micro_batch={args.micro_batch} seq_len={args.seq_len} accum={accum} "
        f"tokens/step={tokens_per_step:,} total_steps={total_steps}")
    log(f"data: {data.describe()}")

    t0 = time.time()
    model = build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters())
    optim = build_optimizer(model, args)

    start_step = 0
    resume = latest_checkpoint(args.out)
    # Buffers always come from the base model; weights too unless resuming.
    load_base_weights(model, args, device)
    if resume is not None:
        start_step = load_checkpoint(model, optim, data, resume)
        log(f"resumed from {resume} at step {start_step}")
    log(f"model ready: {n_params / 1e9:.2f}B params in {time.time() - t0:.0f}s, "
        f"cpu_offload={args.cpu_offload}")

    end_step = min(total_steps, start_step + args.max_steps) if args.max_steps else total_steps
    model.train()
    for step in range(start_step, end_step):
        t_step = time.time()
        lr = lr_at(step, total_steps, args)
        for group in optim.param_groups:
            group["lr"] = lr

        loss_sum = torch.zeros((), device=device)
        for _ in range(accum):
            input_ids = data.next(rank, world, args.micro_batch).to(device, non_blocking=True)
            loss = model(input_ids=input_ids, labels=input_ids).loss
            (loss / accum).backward()
            loss_sum += loss.detach()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()
        optim.step()
        optim.zero_grad(set_to_none=True)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
        dt = time.time() - t_step
        if (step + 1) % args.log_every == 0:
            tok_s = tokens_per_step / dt
            mfu = 6 * n_params * tok_s / (world * args.peak_tflops * 1e12)
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
            eta_h = (total_steps - step - 1) * dt / 3600
            log(f"step {step + 1}/{total_steps} loss {loss_sum.item() / accum:.4f} "
                f"gnorm {grad_norm.item():.3f} lr {lr:.2e} {dt:.1f}s/step {tok_s:,.0f} tok/s "
                f"mfu {mfu:.1%} mem {mem:.1f}GiB eta {eta_h:.1f}h")

        if (step + 1) % args.save_every == 0 or step + 1 == end_step:
            save_checkpoint(model, optim, data, step + 1, args)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
