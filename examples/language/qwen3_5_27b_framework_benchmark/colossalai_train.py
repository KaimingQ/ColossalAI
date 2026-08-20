#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ColossalAI 预训练（框架效率基准）脚本 —— Qwen3.5 27B 纯文本版（Qwen3_5ForCausalLM）
与 DeepSpeed 版 train.py 完全对齐的指标口径（tok/s、6N TFLOPS、显存、summary）。

用法（torchrun 启动）:
    torchrun --nproc_per_node=8 --master_port=29502 colossalai_train.py \
        --model-dir ~/models/Qwen3.8-27B --plugin gemini \
        --steps 25 --seq-len 4096 --batch-size 1 --grad-accum 4 --log-file bench.jsonl

插件:
    --plugin gemini : ColossalAI 特有优化 Gemini（chunk 化异构内存管理 + 参数/优化器分片）
                     静态纯 GPU 模式 = ZeRO-3 风格，对标 DeepSpeed ZeRO-3
    --plugin fsdp   : 封装 PyTorch FSDP（FULL_SHARD, bf16），标准 ZeRO-3
"""
import argparse
import json
import os
import time
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, TorchFSDPPlugin
from colossalai.nn.optimizer import HybridAdam
from transformers import Qwen3_5ForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, default=os.path.expanduser("~/models/Qwen3.8-27B"))
    p.add_argument("--plugin", type=str, default="gemini", choices=["gemini", "fsdp"])
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1, help="micro batch per GPU")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-file", type=str, default="", help="rank0 JSONL 日志（可选）")
    p.add_argument("--save-dir", type=str, default="", help="checkpoint 目录（可选）")
    p.add_argument("--no-grad-ckpt", action="store_true", help="关闭梯度检查点")
    return p.parse_args()


class SyntheticDataset:
    """与 DeepSpeed 版完全一致的合成数据流（seed+rank 独立流）。"""

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
    colossalai.launch_from_torch(backend="nccl", seed=args.seed, verbose=False)
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
    vocab_size = model.config.vocab_size  # boost 前保存（包装后无 .config）

    # ---------- 优化器 / 调度器（与 DeepSpeed 配置对齐） ----------
    # Gemini 插件只接受 ColossalAI 自家优化器（FusedAdam/CPUAdam/HybridAdam），
    # 用 HybridAdam（adamw_mode=True 即 AdamW 语义）手动适配；
    # FSDP 插件接受普通 torch 优化器。
    if args.plugin == "gemini":
        optimizer = HybridAdam(model.parameters(), lr=args.lr,
                               betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
                               adamw_mode=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      betas=(0.9, 0.95), eps=1e-8,
                                      weight_decay=0.1)

    def lr_lambda(step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return float(step + 1) / args.warmup_steps
        return 1.0

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---------- 插件（ColossalAI 优化） ----------
    if args.plugin == "gemini":
        # Gemini：ColossalAI 特有优化。static + 纯 GPU = ZeRO-3 风格（参数/优化器分片）
        plugin = GeminiPlugin(
            placement_policy="static",
            shard_param_frac=1.0,
            offload_optim_frac=0.0,
            offload_param_frac=0.0,
            precision="bf16",
            master_weights=True,
            enable_gradient_accumulation=True,
            force_outputs_fp32=False,
        )
    else:
        plugin = TorchFSDPPlugin(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                           reduce_dtype=torch.bfloat16,
                                           buffer_dtype=torch.bfloat16),
            auto_wrap_policy=partial(transformer_auto_wrap_policy,
                                     transformer_layer_cls={Qwen3_5DecoderLayer}),
        )

    booster = Booster(plugin=plugin)
    model, optimizer, _, _, lr_scheduler = booster.boost(
        model, optimizer, lr_scheduler=lr_scheduler)
    model.train()  # 确保训练模式（梯度检查点/ dropout 生效）
    load_time = time.time() - t_load

    # ---------- 数据 ----------
    data = SyntheticDataset(vocab_size, args.seq_len, args.seed, rank)

    gbs_tokens = args.batch_size * world * args.grad_accum * args.seq_len
    flops_per_token = 6.0 * n_params
    if rank == 0:
        print(f"[config] model={args.model_dir} params={n_params/1e9:.2f}B "
              f"gpus={world} plugin={args.plugin} micro_bs={args.batch_size} "
              f"accum={args.grad_accum} seq={args.seq_len} "
              f"global_batch_tokens={gbs_tokens} steps={args.steps} lr={args.lr} "
              f"grad_ckpt={not args.no_grad_ckpt}")
        print(f"[load] model+boost took {load_time:.1f}s")

    log_fp = open(args.log_file, "w") if (rank == 0 and args.log_file) else None
    total_tok = 0
    t_global = time.time()
    recent = []

    for step in range(args.steps):
        t_step = time.time()
        loss_sum = 0.0
        for _ in range(args.grad_accum):
            input_ids = data.get_batch(args.batch_size, device)
            out = model(input_ids=input_ids, labels=input_ids)
            booster.backward(out.loss, optimizer)
            loss_sum += out.loss.detach().float()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()
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
    if rank == 0:
        mem_alloc = torch.cuda.max_memory_allocated(device) / 1e9
        summary = {
            "params": n_params, "gpus": world, "plugin": args.plugin,
            "seq_len": args.seq_len, "global_batch_tokens": gbs_tokens,
            "steps": args.steps, "total_tokens": total_tok,
            "elapsed_s": round(elapsed, 1),
            "avg_tokens_per_s": round(total_tok / elapsed),
            "stable_tokens_per_s": round(stable_tok_ps),
            "stable_tflops": round(stable_tflops, 1),
            "peak_mem_alloc_gb": round(mem_alloc, 1),
            "load_time_s": round(load_time, 1),
        }
        print("[summary] " + json.dumps(summary, ensure_ascii=False))
        if log_fp:
            log_fp.write("[summary] " + json.dumps(summary, ensure_ascii=False) + "\n")
    if log_fp:
        log_fp.close()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        booster.save_model(model, os.path.join(args.save_dir, "model"),
                           shard=True, use_safetensors=True)
        booster.save_optimizer(optimizer, os.path.join(args.save_dir, "optim"))

    dist.barrier()


if __name__ == "__main__":
    main()
