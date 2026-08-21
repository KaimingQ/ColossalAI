#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PyTorch 原生训练基线脚本（DDP / FSDP），与 DeepSpeed / ColossalAI 同口径。

指标口径完全一致：
  - TFLOPS = 6 × 参数量 × 全局每步 token 数 / 步耗时
  - stable_* 剔除 warmup 步
  - 合成数据流式预训练（seed + rank 独立流）

用法：
  torchrun --nproc_per_node=8 pytorch_native_train.py \
      --mode fsdp --seq-len 4096 --batch-size 1 --grad-accum 4 --steps 25
"""
import argparse
import json
import os
import time
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from transformers import Qwen3_5ForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer


class SyntheticDataset:
    """与 DeepSpeed / ColossalAI 版完全一致的合成数据流。"""

    def __init__(self, vocab_size, seq_len, seed, rank):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.g = torch.Generator()
        self.g.manual_seed(seed * 1000 + rank)

    def get_batch(self, batch_size, device):
        ids = torch.randint(0, self.vocab_size, (batch_size, self.seq_len),
                            generator=self.g, dtype=torch.long)
        return ids.to(device)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, default=os.path.expanduser("~/models/Qwen3.8-27B"))
    p.add_argument("--mode", type=str, default="fsdp", choices=["ddp", "fsdp"])
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1, help="micro batch per GPU")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-file", type=str, default="", help="rank0 JSONL 日志（可选）")
    p.add_argument("--no-grad-ckpt", action="store_true", help="关闭梯度检查点")
    return p.parse_args()


def main():
    args = parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    # ---------- 加载模型 ----------
    t_load = time.time()
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.config.use_cache = False
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters())
    vocab_size = model.config.vocab_size

    # ---------- 优化器 / 调度器 ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    def lr_lambda(step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return float(step + 1) / args.warmup_steps
        return 1.0

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---------- 并行策略 ----------
    if args.mode == "fsdp":
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={Qwen3_5DecoderLayer},
            ),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=local_rank,
            use_orig_params=False,
        )
    else:  # ddp
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True,
                    static_graph=True)

    model.train()
    load_time = time.time() - t_load

    # ---------- 数据 ----------
    data = SyntheticDataset(vocab_size, args.seq_len, args.seed, rank)

    gbs_tokens = args.batch_size * world * args.grad_accum * args.seq_len
    flops_per_token = 6.0 * n_params
    if rank == 0:
        print(f"[config] model={args.model_dir} params={n_params/1e9:.2f}B "
              f"gpus={world} mode={args.mode} micro_bs={args.batch_size} "
              f"accum={args.grad_accum} seq={args.seq_len} "
              f"global_batch_tokens={gbs_tokens} steps={args.steps} lr={args.lr} "
              f"grad_ckpt={not args.no_grad_ckpt}")
        print(f"[load] model+parallel took {load_time:.1f}s")

    log_fp = open(args.log_file, "w") if (rank == 0 and args.log_file) else None
    total_tok = 0
    t_global = time.time()
    recent = []

    # 分段计时
    timings = {"fwd": [], "bwd": [], "opt": [], "zero": []}

    for step in range(args.steps):
        t_step = time.time()
        loss_sum = 0.0
        t_fwd = t_bwd = 0.0
        for _ in range(args.grad_accum):
            input_ids = data.get_batch(args.batch_size, device)
            torch.cuda.synchronize(device)
            t_f = time.time()
            out = model(input_ids=input_ids, labels=input_ids)
            torch.cuda.synchronize(device)
            t_fwd += time.time() - t_f
            t_b = time.time()
            out.loss.backward()
            torch.cuda.synchronize(device)
            t_bwd += time.time() - t_b
            loss_sum += out.loss.detach().float()
        torch.cuda.synchronize(device)
        t_o = time.time()
        optimizer.step()
        torch.cuda.synchronize(device)
        t_opt = time.time() - t_o
        t_z = time.time()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        t_zero = time.time() - t_z
        lr_scheduler.step()
        loss = loss_sum / args.grad_accum
        dt = time.time() - t_step
        if step >= min(2, args.steps - 1):
            timings["fwd"].append(t_fwd)
            timings["bwd"].append(t_bwd)
            timings["opt"].append(t_opt)
            timings["zero"].append(t_zero)
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
                "lr": round(float(lr_scheduler.get_last_lr()[0]), 8),
            }
            print(f"[step {step+1:>3}/{args.steps}] loss={m['loss']} "
                  f"{m['tokens_per_s']} tok/s {m['tflops']} TFLOPS "
                  f"{m['step_time_s']}s/step mem={mem_alloc:.1f}/{mem_resv:.1f}GB")
            if log_fp:
                log_fp.write(json.dumps(m) + "\n")
                log_fp.flush()

    elapsed = time.time() - t_global
    if len(recent) > max(2, args.steps // 5):
        recent = recent[-max(2, args.steps // 5):]
    stable_tok_ps = sum(recent) / len(recent)
    stable_tflops = flops_per_token * stable_tok_ps / 1e12

    avg_timing = {}
    if timings["fwd"]:
        n = len(timings["fwd"])
        avg_timing = {
            "fwd_ms": round(sum(timings["fwd"]) / n * 1e3),
            "bwd_ms": round(sum(timings["bwd"]) / n * 1e3),
            "opt_ms": round(sum(timings["opt"]) / n * 1e3),
            "zero_ms": round(sum(timings["zero"]) / n * 1e3),
        }

    if rank == 0:
        mem_alloc = torch.cuda.max_memory_allocated(device) / 1e9
        summary = {
            "params": n_params, "gpus": world, "mode": args.mode,
            "seq_len": args.seq_len, "global_batch_tokens": gbs_tokens,
            "steps": args.steps, "total_tokens": total_tok,
            "elapsed_s": round(elapsed, 1),
            "avg_tokens_per_s": round(total_tok / elapsed),
            "stable_tokens_per_s": round(stable_tok_ps),
            "stable_tflops": round(stable_tflops, 1),
            "peak_mem_alloc_gb": round(mem_alloc, 1),
            "load_time_s": round(load_time, 1),
        }
        if avg_timing:
            summary.update(avg_timing)
        print("[summary] " + json.dumps(summary, ensure_ascii=False))
        if log_fp:
            log_fp.write("[summary] " + json.dumps(summary, ensure_ascii=False) + "\n")
    if log_fp:
        log_fp.close()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
