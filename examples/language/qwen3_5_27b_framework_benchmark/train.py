#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepSpeed 预训练（框架效率基准）脚本 —— Qwen3.5 27B 纯文本版（Qwen3_5ForCausalLM）
用法:
    deepspeed --num_gpus 8 train.py --ds-config ds_config.json \
        --model-dir ~/models/Qwen3.8-27B --steps 30 --seq-len 4096 \
        --batch-size 2 --grad-accum 4 --log-file bench.jsonl
"""
import argparse
import json
import os
import time

import torch
import deepspeed
import deepspeed.comm as dist
from transformers import Qwen3_5ForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, default=os.path.expanduser("~/models/Qwen3.8-27B"))
    p.add_argument("--ds-config", type=str, default="ds_config.json")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=2, help="micro batch per GPU")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-file", type=str, default="", help="rank0 JSONL 日志（可选）")
    p.add_argument("--save-dir", type=str, default="", help="DeepSpeed checkpoint 目录（可选）")
    p.add_argument("--no-grad-ckpt", action="store_true", help="关闭梯度检查点")
    p.add_argument("--local_rank", type=int, default=-1, help="deepspeed launcher 自动传入")
    return p.parse_args()


class SyntheticDataset:
    """无限流式合成预训练数据：按 seed+rank 产生伪随机 token 序列。

    与真实文本训练的计算路径（attention/线性层/embedding 访存）完全一致，
    用于框架效率基准时无需下载语料，且完全可复现。
    """

    def __init__(self, vocab_size, seq_len, seed, rank):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.g = torch.Generator()
        self.g.manual_seed(seed * 1000 + rank)  # 每 rank 独立数据流

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

    # 训练规模参数以 CLI 为准，注入 ds config，避免三处不一致
    ds_cfg["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_cfg["gradient_accumulation_steps"] = args.grad_accum
    ds_cfg["train_batch_size"] = args.batch_size * world * args.grad_accum
    ds_cfg["optimizer"]["params"]["lr"] = args.lr
    if "scheduler" in ds_cfg:
        ds_cfg["scheduler"]["params"]["warmup_num_steps"] = args.warmup_steps

    # ---------- 加载模型 ----------
    t_load = time.time()
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.config.use_cache = False  # 训练不需要 KV cache
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters())

    engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config_params=ds_cfg)
    # 关键：DeepSpeed initialize 会将模型置为 eval 模式，导致
    # GradientCheckpointingLayer 的 checkpoint 不生效（激活全量保留、显存暴涨）。
    # 必须切回训练模式。
    engine.module.train()
    load_time = time.time() - t_load

    # ---------- 数据 ----------
    vocab_size = model.config.vocab_size
    data = SyntheticDataset(vocab_size, args.seq_len, args.seed, rank)

    gbs_tokens = args.batch_size * world * args.grad_accum * args.seq_len
    flops_per_token = 6.0 * n_params  # 前向+反向 2N + 优化 2N
    if rank == 0:
        print(f"[config] model={args.model_dir} params={n_params/1e9:.2f}B "
              f"gpus={world} micro_bs={args.batch_size} accum={args.grad_accum} "
              f"seq={args.seq_len} global_batch_tokens={gbs_tokens} "
              f"steps={args.steps} lr={args.lr} grad_ckpt={not args.no_grad_ckpt}")
        print(f"[load] model+deepspeed init took {load_time:.1f}s")

    log_fp = open(args.log_file, "w") if (rank == 0 and args.log_file) else None
    total_tok = 0
    t_global = time.time()
    recent = []  # 最近 steps 的 tokens/s，用于报告稳定吞吐

    for step in range(args.steps):
        t_step = time.time()
        loss_sum = 0.0
        for _ in range(args.grad_accum):
            input_ids = data.get_batch(args.batch_size, device)
            out = engine(input_ids=input_ids, labels=input_ids)
            engine.backward(out.loss)
            engine.step()
            loss_sum += out.loss.detach().float()
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

    elapsed = time.time() - t_global
    if len(recent) > max(2, args.steps // 5):  # 剔除最前若干步（含预热/编译）
        recent = recent[-max(2, args.steps // 5):]
    stable_tok_ps = sum(recent) / len(recent)
    stable_tflops = flops_per_token * stable_tok_ps / 1e12
    if rank == 0:
        mem_alloc = torch.cuda.max_memory_allocated(device) / 1e9
        summary = {
            "params": n_params, "gpus": world, "seq_len": args.seq_len,
            "global_batch_tokens": gbs_tokens, "steps": args.steps,
            "total_tokens": total_tok, "elapsed_s": round(elapsed, 1),
            "avg_tokens_per_s": round(total_tok / elapsed),
            "stable_tokens_per_s": round(stable_tok_ps),
            "stable_tflops": round(stable_tflops, 1),
            "peak_mem_alloc_gb": round(mem_alloc, 1),
            "load_time_s": round(load_time, 1),
        }
        print("[summary] " + json.dumps(summary, ensure_ascii=False))
        if log_fp:
            log_fp.write("[summary] " + json.dumps(summary, ensure_ascii=False) + "\n")
        if args.save_dir and rank == 0:
            with open(os.path.join(args.save_dir, "bench_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
    if log_fp:
        log_fp.close()

    # 可选：保存 DeepSpeed checkpoint（ZeRO-3 会分片保存）
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        engine.save_checkpoint(args.save_dir, client_state={"step": args.steps})

    dist.barrier()


if __name__ == "__main__":
    main()
