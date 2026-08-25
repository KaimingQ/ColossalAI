#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ColossalAI Pretraining & Benchmarking Script
Supports GeminiPlugin, TorchFSDPPlugin (ZeRO-3 / ZeRO-2), AutoModel / Qwen3.5 / Gemma4.
Tracks throughput (tokens/s), 6N TFLOPS, peak memory (alloc/resv), and catches OOM.
"""
import argparse
import gc
import json
import os
import sys
import time
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, TorchFSDPPlugin
from colossalai.nn.optimizer import HybridAdam
from transformers import AutoConfig, AutoModelForCausalLM

def _patch_conv1d_for_gemini():
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
        _orig = mq.causal_conv1d_fn
        def _patched(hidden_states, weight, bias=None, activation=None, **kwargs):
            weight = weight.clone()
            if bias is not None:
                bias = bias.clone()
            return _orig(hidden_states, weight, bias, activation, **kwargs)
        mq.causal_conv1d_fn = _patched
    except Exception:
        pass


def get_decoder_layer_cls():
    layer_classes = set()
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
        layer_classes.add(Qwen3_5DecoderLayer)
    except Exception:
        pass
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        layer_classes.add(Qwen3DecoderLayer)
    except Exception:
        pass
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
        layer_classes.add(Gemma4TextDecoderLayer)
    except Exception:
        pass
    return layer_classes


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, default=os.path.expanduser("~/models/Qwen3.8-27B"))
    p.add_argument("--plugin", type=str, default="fsdp", choices=["gemini", "fsdp", "fsdp_grad"])
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
    p.add_argument("--cpu-offload", action="store_true", help="enable CPU offloading")
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
    colossalai.launch_from_torch(backend="nccl", seed=args.seed, verbose=False)
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

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

    # ---------- Optimizer ----------
    if args.plugin == "gemini":
        optimizer = HybridAdam(model.parameters(), lr=args.lr,
                               betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
                               adamw_mode=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      betas=(0.9, 0.95), eps=1e-8,
                                      weight_decay=0.1)
        # Fix for PyTorch >=2.5 defaults compatibility with TorchFSDPPlugin
        if hasattr(optimizer, "defaults"):
            optimizer.defaults.pop("decoupled_weight_decay", None)

    def lr_lambda(step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return float(step + 1) / args.warmup_steps
        return 1.0

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---------- Plugin ----------
    if args.plugin == "gemini":
        _patch_conv1d_for_gemini()
        plugin = GeminiPlugin(
            placement_policy="static",
            shard_param_frac=1.0,
            offload_optim_frac=0.0,
            offload_param_frac=0.0,
            precision="bf16",
            master_weights=True,
            enable_gradient_accumulation=True,
            force_outputs_fp32=False,
            max_prefetch=2,
            enable_fused_normalization=True,
            enable_jit_fused=True,
            enable_async_reduce=True,
        )
    else:
        strategy = (ShardingStrategy.SHARD_GRAD_OP if args.plugin == "fsdp_grad"
                    else ShardingStrategy.FULL_SHARD)
        layer_classes = get_decoder_layer_cls()
        cpu_offload = CPUOffload(offload_params=True) if args.cpu_offload else None
        plugin = TorchFSDPPlugin(
            sharding_strategy=strategy,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                           reduce_dtype=torch.bfloat16,
                                           buffer_dtype=torch.bfloat16),
            auto_wrap_policy=partial(transformer_auto_wrap_policy,
                                     transformer_layer_cls=layer_classes),
            cpu_offload=cpu_offload,
        )

    booster = Booster(plugin=plugin)
    try:
        model, optimizer, _, _, lr_scheduler = booster.boost(
            model, optimizer, lr_scheduler=lr_scheduler)
        model.train()
    except torch.cuda.OutOfMemoryError as e:
        if rank == 0:
            print(f"[OOM during ColossalAI boost]: {e}", file=sys.stderr)
            if args.log_file:
                with open(args.log_file, "w") as fp:
                    fp.write(json.dumps({"status": "OOM_BOOST", "error": str(e)}) + "\n")
        sys.exit(1)

    load_time = time.time() - t_load

    # ---------- Data ----------
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

    oom_occurred = False
    for step in range(args.steps):
        t_step = time.time()
        loss_sum = 0.0
        try:
            for _ in range(args.grad_accum):
                input_ids = data.get_batch(args.batch_size, device)
                out = model(input_ids=input_ids, labels=input_ids)
                booster.backward(out.loss, optimizer)
                loss_sum += out.loss.detach().float()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
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
                "lr": round(float(lr_scheduler.get_last_lr()[0]), 8),
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
                "framework": "ColossalAI",
                "plugin": args.plugin,
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
