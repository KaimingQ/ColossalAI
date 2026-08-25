#!/usr/bin/env python3
"""
PyTorch Native Benchmark for LLM Fine-Tuning (Full Parameter SFT & LoRA PEFT).
Supports Single GPU, PyTorch DDP, and PyTorch FSDP.
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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


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
    p.add_argument("--model-dir", type=str, default=os.path.expanduser("~/models/Qwen3.5-9B"))
    p.add_argument("--mode", type=str, default="lora", choices=["full", "lora"], help="Fine-tuning mode")
    p.add_argument("--strategy", type=str, default="single", choices=["single", "ddp", "fsdp"], help="PyTorch strategy")
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
    p.add_argument("--cpu-offload", action="store_true", help="enable CPU offloading")
    return p.parse_args()


class SyntheticSFTDataset:
    """Synthetic SFT Dataset with prompt masking (labels=-100 for prompt tokens)"""
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


def compute_tflops(num_params, batch_size, seq_len, step_time_s, is_training=True):
    factor = 6 if is_training else 2
    flops = factor * num_params * batch_size * seq_len
    return (flops / step_time_s) / 1e12


def main():
    args = parse_args()

    is_dist = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_dist:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)

    # ---------- Model Loading ----------
    t_load_start = time.time()
    cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    if hasattr(cfg, "text_config") and cfg.text_config is not None:
        vocab_size = getattr(cfg.text_config, "vocab_size", 152064)
    else:
        vocab_size = getattr(cfg, "vocab_size", 152064)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.gradient_checkpointing_enable()
    total_params = sum(p.numel() for p in model.parameters())

    # ---------- Apply LoRA if mode == 'lora' ----------
    if args.mode == "lora":
        if "gemma" in args.model_dir.lower():
            target_modules = r".*language_model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        else:
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
            print(f"[PyTorch LoRA Config] r={args.lora_rank}, alpha={args.lora_alpha}, "
                  f"trainable={trainable_params:,} ({trainable_params/total_params*100:.3f}%), total={total_params:,}")
    else:
        trainable_params = total_params
        if rank == 0:
            print(f"[PyTorch Full SFT Config] trainable={trainable_params:,}, total={total_params:,}")

    # ---------- Optimizer ----------
    trainable_p = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_p, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)

    # ---------- Wrapper Strategy ----------
    if args.strategy == "single" or not is_dist:
        model = model.to(device)
    elif args.strategy == "ddp":
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    elif args.strategy == "fsdp":
        layer_classes = get_decoder_layer_cls()
        cpu_offload = CPUOffload(offload_params=True) if args.cpu_offload else None
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD if world > 1 else ShardingStrategy.NO_SHARD,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                           reduce_dtype=torch.bfloat16,
                                           buffer_dtype=torch.bfloat16),
            auto_wrap_policy=partial(transformer_auto_wrap_policy,
                                     transformer_layer_cls=layer_classes),
            cpu_offload=cpu_offload,
            device_id=device,
        )

    model.train()
    load_time_s = time.time() - t_load_start

    dataset = SyntheticSFTDataset(vocab_size, args.seq_len, args.prompt_ratio, args.seed, rank)
    gbs_tokens = args.batch_size * world * args.grad_accum * args.seq_len

    if rank == 0:
        print(f"[config] PyTorch Native model={args.model_dir} mode={args.mode} params={total_params/1e9:.2f}B "
              f"gpus={world} micro_bs={args.batch_size} accum={args.grad_accum} seq={args.seq_len} "
              f"gbs_tokens={gbs_tokens} steps={args.steps} lr={args.lr}")
        print(f"[load] model init took {load_time_s:.1f}s")

    step_times = []
    tokens_per_s_list = []
    tflops_list = []

    for step in range(1, args.steps + 1):
        torch.cuda.reset_peak_memory_stats(device)
        t_start = time.time()
        optimizer.zero_grad()
        total_loss = 0.0

        for acc in range(args.grad_accum):
            input_ids, attention_mask, labels = dataset.get_batch(args.batch_size, device)
            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / args.grad_accum
                loss.backward()
                total_loss += loss.item()
            except torch.cuda.OutOfMemoryError as e:
                peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                peak_resv = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
                if rank == 0:
                    print(f"[OOM at step {step}]: {e}, peak_alloc={peak_alloc:.1f}GB, peak_resv={peak_resv:.1f}GB", file=sys.stderr)
                    if args.log_file:
                        with open(args.log_file, "a") as f:
                            f.write(json.dumps({
                                "status": "OOM",
                                "framework": "PyTorch Native",
                                "mode": args.mode,
                                "strategy": args.strategy,
                                "total_params": total_params,
                                "trainable_params": trainable_params,
                                "gpus": world,
                                "seq_len": args.seq_len,
                                "micro_bs": args.batch_size,
                                "grad_accum": args.grad_accum,
                                "step": step,
                                "peak_mem_alloc_gb": round(peak_alloc, 1),
                                "peak_mem_resv_gb": round(peak_resv, 1),
                                "error": str(e),
                            }) + "\n")
                sys.exit(1)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        torch.cuda.synchronize(device)
        step_time = time.time() - t_start
        step_tokens = args.batch_size * world * args.grad_accum * args.seq_len
        tok_s = step_tokens / step_time
        eval_params = total_params if args.mode == "full" else (total_params + trainable_params)
        tflops = compute_tflops(eval_params, args.batch_size * world * args.grad_accum, args.seq_len, step_time)

        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_resv = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

        if rank == 0:
            print(f"[step {step:3d}/{args.steps}] loss={total_loss:.4f} {tok_s:4.0f} tok/s "
                  f"{tflops:4.1f} TFLOPS {step_time:.2f}s/step mem={peak_alloc:.1f}/{peak_resv:.1f}GB")

        if step > args.warmup_steps:
            step_times.append(step_time)
            tokens_per_s_list.append(tok_s)
            tflops_list.append(tflops)

    if rank == 0:
        avg_tok_s = sum(tokens_per_s_list) / len(tokens_per_s_list) if tokens_per_s_list else tok_s
        avg_tflops = sum(tflops_list) / len(tflops_list) if tflops_list else tflops
        summary = {
            "status": "SUCCESS",
            "framework": "PyTorch Native",
            "mode": args.mode,
            "strategy": args.strategy,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "gpus": world,
            "seq_len": args.seq_len,
            "micro_bs": args.batch_size,
            "grad_accum": args.grad_accum,
            "global_batch_tokens": gbs_tokens,
            "steps": args.steps,
            "avg_tokens_per_s": int(round(avg_tok_s)),
            "stable_tflops": round(avg_tflops, 1),
            "peak_mem_alloc_gb": round(peak_alloc, 1),
            "peak_mem_resv_gb": round(peak_resv, 1),
            "load_time_s": round(load_time_s, 1),
        }
        print(f"[summary] {json.dumps(summary)}")
        if args.log_file:
            with open(args.log_file, "a") as f:
                f.write(json.dumps(summary) + "\n")

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
