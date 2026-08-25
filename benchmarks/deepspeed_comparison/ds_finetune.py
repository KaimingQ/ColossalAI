#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepSpeed Fine-Tuning Benchmark Script (Full Parameter SFT & LoRA PEFT)
Supports Qwen3.5-9B, Qwen3.8-27B, Gemma4-31B on single-card & multi-card H20.
Measures: SFT loss, trainable params, peak GPU memory, step time, tokens/s, TFLOPS.
"""
import argparse
import gc
import json
import os
import sys
import time

import deepspeed
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, default=os.path.expanduser("~/models/Qwen3.5-9B"))
    p.add_argument("--ds-config", type=str, default="/home/qukaiming/deepspeed/ds_config_zero3_puregpu.json")
    p.add_argument("--mode", type=str, default="full", choices=["full", "lora"], help="Fine-tuning mode")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--prompt-ratio", type=float, default=0.5, help="Ratio of prompt tokens (masked with -100)")
    p.add_argument("--batch-size", type=int, default=1, help="micro batch per GPU")
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--warmup-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-file", type=str, default="", help="rank0 JSONL log file")
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


class SyntheticSFTDataset:
    def __init__(self, vocab_size, seq_len, prompt_ratio, seed, rank):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.prompt_len = int(seq_len * prompt_ratio)
        self.g = torch.Generator()
        self.g.manual_seed(seed * 1000 + rank)

    def get_batch(self, batch_size, device):
        input_ids = torch.randint(10, self.vocab_size, (batch_size, self.seq_len),
                                  generator=self.g, dtype=torch.long).to(device)
        labels = input_ids.clone()
        labels[:, :self.prompt_len] = -100  # mask prompt tokens
        attention_mask = torch.ones_like(input_ids, device=device)
        return input_ids, attention_mask, labels


def main():
    args = parse_args()
    deepspeed.init_distributed()
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    # ---------- Load Config & Model ----------
    t_load = time.time()
    with open(args.ds_config) as f:
        ds_cfg = json.load(f)

    ds_cfg["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_cfg["gradient_accumulation_steps"] = args.grad_accum
    ds_cfg["train_batch_size"] = args.batch_size * world * args.grad_accum

    cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    if hasattr(cfg, "text_config") and cfg.text_config is not None:
        vocab_size = getattr(cfg.text_config, "vocab_size", 152064)
    else:
        vocab_size = getattr(cfg, "vocab_size", 152064)

    is_zero3 = ds_cfg.get("zero_optimization", {}).get("stage", 0) == 3

    if is_zero3:
        with deepspeed.zero.Init(config_dict_or_path=ds_cfg):
            model = AutoModelForCausalLM.from_pretrained(
                args.model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.gradient_checkpointing_enable()
    total_params = sum(p.numel() for p in model.parameters())

    # ---------- Apply LoRA if mode == 'lora' ----------
    if args.mode == "lora":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if rank == 0:
            print(f"[LoRA Config] r={args.lora_rank}, alpha={args.lora_alpha}, "
                  f"trainable={trainable_params:,} ({trainable_params/total_params*100:.3f}%), total={total_params:,}")
    else:
        trainable_params = total_params
        if rank == 0:
            print(f"[Full SFT Config] trainable={trainable_params:,}, total={total_params:,}")

    # ---------- DeepSpeed Initialize ----------
    try:
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=[p for p in model.parameters() if p.requires_grad],
            config=ds_cfg,
        )
    except torch.cuda.OutOfMemoryError as e:
        if rank == 0:
            print(f"[OOM during DeepSpeed init]: {e}", file=sys.stderr)
            if args.log_file:
                with open(args.log_file, "w") as fp:
                    fp.write(json.dumps({"status": "OOM_INIT", "error": str(e)}) + "\n")
        sys.exit(1)

    load_time = time.time() - t_load

    # ---------- Data & SFT Loop ----------
    dataset = SyntheticSFTDataset(vocab_size, args.seq_len, args.prompt_ratio, args.seed, rank)
    gbs_tokens = args.batch_size * world * args.grad_accum * args.seq_len

    if rank == 0:
        print(f"[config] model={args.model_dir} mode={args.mode} params={total_params/1e9:.2f}B "
              f"gpus={world} micro_bs={args.batch_size} accum={args.grad_accum} "
              f"seq={args.seq_len} gbs_tokens={gbs_tokens} steps={args.steps} lr={args.lr}")
        print(f"[load] model+deepspeed init took {load_time:.1f}s")

    step_times = []
    tokens_per_s_list = []
    tflops_list = []

    for step in range(args.steps):
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.time()

        accum_loss = 0.0
        try:
            for micro_step in range(args.grad_accum):
                input_ids, attention_mask, labels = dataset.get_batch(args.batch_size, device)
                outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                model_engine.backward(loss)
                model_engine.step()
                accum_loss += loss.item()

            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as e:
            mem_alloc = torch.cuda.max_memory_allocated() / (1024**3)
            mem_resv = torch.cuda.max_memory_reserved() / (1024**3)
            if rank == 0:
                print(f"[OOM at step {step+1}]: {e}, peak_alloc={mem_alloc:.1f}GB, peak_resv={mem_resv:.1f}GB",
                      file=sys.stderr)
                if args.log_file:
                    with open(args.log_file, "w") as fp:
                        fp.write(json.dumps({
                            "status": "OOM", "step": step + 1,
                            "peak_mem_alloc_gb": round(mem_alloc, 2),
                            "peak_mem_resv_gb": round(mem_resv, 2),
                            "error": str(e)
                        }) + "\n")
            sys.exit(1)

        t_step = time.time() - t0
        mem_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        mem_resv = torch.cuda.max_memory_reserved() / (1024**3)

        tok_s = gbs_tokens / t_step
        flops_coeff = 6 if args.mode == "full" else 3
        tflops = (flops_coeff * total_params * gbs_tokens) / (t_step * 1e12)

        if step >= args.warmup_steps:
            step_times.append(t_step)
            tokens_per_s_list.append(tok_s)
            tflops_list.append(tflops)

        if rank == 0:
            print(f"[step {step+1:3d}/{args.steps}] loss={accum_loss:.4f} "
                  f"{tok_s:.0f} tok/s {tflops:.1f} TFLOPS {t_step:.2f}s/step "
                  f"mem={mem_alloc:.1f}/{mem_resv:.1f}GB")

    # ---------- Summary ----------
    if rank == 0 and len(step_times) > 0:
        avg_tok_s = int(sum(tokens_per_s_list) / len(tokens_per_s_list))
        avg_tflops = round(sum(tflops_list) / len(tflops_list), 1)
        summary = {
            "status": "SUCCESS",
            "framework": "DeepSpeed",
            "mode": args.mode,
            "config": args.ds_config,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "gpus": world,
            "seq_len": args.seq_len,
            "micro_bs": args.batch_size,
            "grad_accum": args.grad_accum,
            "global_batch_tokens": gbs_tokens,
            "steps": args.steps,
            "avg_tokens_per_s": avg_tok_s,
            "stable_tflops": avg_tflops,
            "peak_mem_alloc_gb": round(mem_alloc, 1),
            "peak_mem_resv_gb": round(mem_resv, 1),
            "load_time_s": round(load_time, 1)
        }
        print(f"[summary] {json.dumps(summary)}")
        if args.log_file:
            with open(args.log_file, "w") as fp:
                fp.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
