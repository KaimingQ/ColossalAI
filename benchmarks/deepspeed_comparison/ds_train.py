#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepSpeed Pretraining & Benchmarking Script
Supports AutoModelForCausalLM / Qwen3_5 / Gemma4 / etc.
Tracks throughput (tokens/s), 6N TFLOPS, peak memory (alloc/resv), and catches OOM.
"""
import argparse
import gc
import json
import os
import sys
import time

import torch
import deepspeed
import deepspeed.comm as dist
from transformers import AutoConfig, AutoModelForCausalLM

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, default=os.path.expanduser("~/models/Qwen3.8-27B"))
    p.add_argument("--ds-config", type=str, default="ds_config_zero3_puregpu.json")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1, help="micro batch per GPU")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-file", type=str, default="", help="rank0 JSONL log file")
    p.add_argument("--save-dir", type=str, default="")
    p.add_argument("--no-grad-ckpt", action="store_true", help="disable gradient checkpointing")
    p.add_argument("--local_rank", type=int, default=-1, help="local rank from launcher")
    return p.parse_args()


class SyntheticDataset:
    def __init__(self, vocab_size, seq_len, seed, rank):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.g = torch.Generator()
        self.g.manual_seed(seed * 1000 + rank)

    def get_batch(self, batch_size, device):
        ids = torch.randint(0, self.vocab_size, (batch_size, self.seq_len),
                            generator=self.g, dtype=torch.long)
        return ids.to(device)


def main():
    args = parse_args()
    deepspeed.init_distributed()
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = dist.get_local_rank()
    device = torch.device("cuda", local_rank)

    torch.manual_seed(args.seed + local_rank)
    torch.cuda.set_device(device)

    with open(args.ds_config) as f:
        ds_cfg = json.load(f)

    ds_cfg["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_cfg["gradient_accumulation_steps"] = args.grad_accum
    ds_cfg["train_batch_size"] = args.batch_size * world * args.grad_accum
    if "optimizer" in ds_cfg and "params" in ds_cfg["optimizer"]:
        ds_cfg["optimizer"]["params"]["lr"] = args.lr
    if "scheduler" in ds_cfg and "params" in ds_cfg["scheduler"]:
        ds_cfg["scheduler"]["params"]["warmup_num_steps"] = args.warmup_steps

    # ---------- Load Model ----------
    t_load = time.time()
    try:
        cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            vocab_size = getattr(cfg.text_config, "vocab_size", 152064)
        else:
            vocab_size = getattr(cfg, "vocab_size", 152064)

        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
    except Exception:
        from transformers import Qwen3_5ForCausalLM
        model = Qwen3_5ForCausalLM.from_pretrained(
            args.model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        vocab_size = model.config.vocab_size

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters())

    try:
        engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model, model_parameters=model.parameters(), config_params=ds_cfg)
        engine.module.train()
    except torch.cuda.OutOfMemoryError as e:
        if rank == 0:
            print(f"[OOM during DeepSpeed init]: {e}", file=sys.stderr)
            if args.log_file:
                with open(args.log_file, "w") as fp:
                    fp.write(json.dumps({"status": "OOM_INIT", "error": str(e)}) + "\n")
        sys.exit(1)

    load_time = time.time() - t_load

    # ---------- Data ----------
    data = SyntheticDataset(vocab_size, args.seq_len, args.seed, rank)
    gbs_tokens = args.batch_size * world * args.grad_accum * args.seq_len
    flops_per_token = 6.0 * n_params

    if rank == 0:
        print(f"[config] model={args.model_dir} params={n_params/1e9:.2f}B "
              f"gpus={world} micro_bs={args.batch_size} accum={args.grad_accum} "
              f"seq={args.seq_len} global_batch_tokens={gbs_tokens} "
              f"steps={args.steps} lr={args.lr} grad_ckpt={not args.no_grad_ckpt} ds_config={args.ds_config}")
        print(f"[load] model+deepspeed init took {load_time:.1f}s")

    log_fp = open(args.log_file, "w") if (rank == 0 and args.log_file) else None
    total_tok = 0
    t_global = time.time()
    recent = []

    oom_occurred = False
    for step in range(args.steps):
        t_step = time.time()
        loss_sum = 0.0
        try:
            for _ in range(args.grad_accum):
                input_ids = data.get_batch(args.batch_size, device)
                out = engine(input_ids=input_ids, labels=input_ids)
                engine.backward(out.loss)
                engine.step()
                loss_sum += out.loss.detach().float()
        except torch.cuda.OutOfMemoryError as e:
            oom_occurred = True
            mem_alloc = torch.cuda.max_memory_allocated(device) / 1e9
            mem_resv = torch.cuda.max_memory_reserved(device) / 1e9
            if rank == 0:
                print(f"[OOM at step {step+1}]: {e}, peak_alloc={mem_alloc:.1f}GB, peak_resv={mem_resv:.1f}GB", file=sys.stderr)
                if log_fp:
                    log_fp.write(json.dumps({"status": "OOM", "step": step+1, "mem_alloc_gb": mem_alloc, "mem_resv_gb": mem_resv}) + "\n")
                    log_fp.flush()
            break

        loss = loss_sum / args.grad_accum
        dt = time.time() - t_step
        total_tok += gbs_tokens
        tok_ps = gbs_tokens / dt
        tflops = flops_per_token * gbs_tokens / dt / 1e12
        recent.append(tok_ps)
        mem_alloc = torch.cuda.max_memory_allocated(device) / 1e9
        mem_resv = torch.cuda.max_memory_reserved(device) / 1e9

        if rank == 0:
            m = {
                "step": step + 1, "loss": round(float(loss), 4),
                "tokens_per_s": round(tok_ps), "tflops": round(tflops, 1),
                "step_time_s": round(dt, 2),
                "mem_alloc_gb": round(mem_alloc, 1), "mem_resv_gb": round(mem_resv, 1),
                "lr": round(float(lr_scheduler.get_last_lr()[0]), 8) if lr_scheduler else args.lr,
            }
            print(f"[step {step+1:>3}/{args.steps}] loss={m['loss']} "
                  f"{m['tokens_per_s']} tok/s {m['tflops']} TFLOPS "
                  f"{m['step_time_s']}s/step mem={mem_alloc:.1f}/{mem_resv:.1f}GB")
            if log_fp:
                log_fp.write(json.dumps(m) + "\n")
                log_fp.flush()

    if not oom_occurred:
        elapsed = time.time() - t_global
        if len(recent) > max(2, args.steps // 5):
            recent = recent[-max(2, args.steps // 5):]
        stable_tok_ps = sum(recent) / len(recent)
        stable_tflops = flops_per_token * stable_tok_ps / 1e12
        if rank == 0:
            mem_alloc = torch.cuda.max_memory_allocated(device) / 1e9
            mem_resv = torch.cuda.max_memory_reserved(device) / 1e9
            summary = {
                "status": "SUCCESS",
                "framework": "DeepSpeed",
                "config": args.ds_config,
                "params": n_params, "gpus": world, "seq_len": args.seq_len,
                "micro_bs": args.batch_size, "grad_accum": args.grad_accum,
                "global_batch_tokens": gbs_tokens, "steps": args.steps,
                "total_tokens": total_tok, "elapsed_s": round(elapsed, 1),
                "avg_tokens_per_s": round(total_tok / elapsed),
                "stable_tokens_per_s": round(stable_tok_ps),
                "stable_tflops": round(stable_tflops, 1),
                "peak_mem_alloc_gb": round(mem_alloc, 1),
                "peak_mem_resv_gb": round(mem_resv, 1),
                "load_time_s": round(load_time, 1),
            }
            print("[summary] " + json.dumps(summary, ensure_ascii=False))
            if log_fp:
                log_fp.write("[summary] " + json.dumps(summary, ensure_ascii=False) + "\n")

    if log_fp:
        log_fp.close()
    if not oom_occurred:
        dist.barrier()


if __name__ == "__main__":
    main()
