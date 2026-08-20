# Qwen3.5 27B 预训练 · DeepSpeed 框架效率基准（复现文档）

> 目标：使用 DeepSpeed（ZeRO-3）在 8×NVIDIA H20 上对 Qwen3.8-27B 进行预训练
> 训练（合成数据），测试训练框架的吞吐 / 算力利用率 / 显存占用，并沉淀可复现的
> 环境、脚本、命令与排障记录。
>
> 机器日期：2026-08-20 · 复现人：qukaiming

---

## 1. 环境

### 1.1 硬件

| 项目 | 配置 |
|---|---|
| GPU | 8 × NVIDIA H20（97,871 MiB / 卡，sm_90，NVLink SHARP 不可用需禁用） |
| 驱动 / CUDA | 595.71.05 / CUDA 13.2 |
| CPU / 内存 | 208 核 / 2 TB RAM |
| 磁盘 | `/` 剩余 ~298G；`/home`（NFS）剩余 ~2.7T |

### 1.2 软件环境（`/home/qukaiming/deepspeed/.venv`）

| 包 | 版本 |
|---|---|
| Python | 3.12.13（uv 管理） |
| torch | 2.13.0+cu130 |
| transformers | 5.15.1（支持 `qwen3_5` 架构，需 ≥5.8.0） |
| deepspeed | 0.19.5 |
| 其他 | ninja（fused_adam JIT 编译必需）、safetensors、sentencepiece |

```bash
# 创建虚拟环境并安装（外网受限时先设置代理）
export https_proxy=http://vpn.luchentech.com:31171
export http_proxy=http://vpn.luchentech.com:31171
export no_proxy=127.0.0.1,localhost

cd ~/deepspeed
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch transformers deepspeed \
    accelerate safetensors sentencepiece ninja
```

> 注意：`ninja` 装好后其可执行文件位于 `.venv/bin/`，运行任何 deepspeed 命令前请
> 把 venv 的 bin 加入 PATH（否则 fused_adam 的 JIT 编译报 "Ninja is required"）：
> `export PATH=$PWD/.venv/bin:$PATH`

### 1.3 模型

`~/models/Qwen3.8-27B/`（54 GB，18 个 safetensors 分片）。

- `config.json`：`model_type=qwen3_5`，架构 `Qwen3_5ForConditionalGeneration`
  （多模态）。纯文本训练用 `Qwen3_5ForCausalLM` 直接加载，权重全部匹配
  （851/851，26.90B 参数）。
- 结构要点：64 层，hidden=5120，intermediate=17408；48 层 GatedDeltaNet
  线性注意力（`linear_attention`，每 4 层插入 1 层全注意力），
  `mamba_ssm_dtype=float32`，vocab=248,320。

---

## 2. 坑与修复（重要排障记录）

按出现顺序记录，每条都曾真实发生：

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | NCCL init 报 `CUDA error 401: Failed to bind NVLink SHARP (NVLS) Multicast memory` | 该环境 NVSwitch/Fabric Manager 不支持 NVLS 多播 | 运行前加 `NCCL_NVLS_ENABLE=0` |
| 2 | `AttributeError: 'NoneType' object has no attribute 'get_rank'` | deepspeed 0.19.x 中 `deepspeed.dist` 不再可用 | 改用 `import deepspeed.comm as dist`，用 `dist.get_rank()` 等 |
| 3 | `RuntimeError: Unable to JIT load the fused_adam op due to ninja not being installed` | fused_adam 首次使用需 JIT 编译，需要 ninja | `uv pip install ninja`，并把 `.venv/bin` 加入 PATH |
| 4 | **bs=2/seq=4096 首个前向即 OOM（已分配 93.4 GiB / 95.08 GiB）** | `deepspeed.initialize` 会把模型置为 **eval 模式**；transformers 5.x 的 `GradientCheckpointingLayer.__call__` 要求 `self.training=True` 才启用梯度检查点 → checkpoint 形同虚设，激活全量保留（bs=1/seq=512 时激活即达 ~30 GB） | `deepspeed.initialize` 后立即 `engine.module.train()`；实测同配置显存 57.9 GB → 29.0 GB |
| 5 | bs=2/seq=4096 第 2 步 OOM（峰值需求 ~95 GB，卡在 96 GB 上限） | 即使 checkpoint 生效，前向+反向+Adam(fp32) 更新的峰值仍超单卡 96 GB | 降为 micro-batch=1（激活减半）后稳定运行；另加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解碎片 |
| 6 | 后台启动命令被 bash 工具判定超时并连带杀进程 | 后台进程持有终端管道 | 用 `setsid nohup env ... > log 2>&1 < /dev/null &` 完全脱离会话 |
| 7 | 模型加载后首个前向耗时长 | fused_adam JIT 编译 + GatedDeltaNet 首次 CUDA 图/autotune 预热（仅首次，缓存后消失） | 属正常现象；benchmark 统计时剔除前若干步 |

> **排障过程中的关键测量**（`diag_mem.py` 分阶段显存轨迹，rank0）：
>
> ```
> after_ds_init                    alloc=27.90GB   # ZeRO-3 静态（参数分片+Adam 状态）
> fwd bs1_seq512 (ckpt ON, eval)   alloc=57.87GB   # eval 模式 → checkpoint 失效
> fwd bs1_seq512 (ckpt ON, train)  alloc=29.05GB   # engine.module.train() 后 → checkpoint 生效
> fwd bs1_seq512 (ckpt OFF, train) alloc=60.57GB   # 对照：关闭 checkpoint 的基线
> ```

---

## 3. 脚本

### 3.1 `train.py`（预训练 / 基准主脚本）

```python
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
```

> 说明：
> - 数据为**合成随机 token**（按 seed+rank 流式生成），计算路径（embedding /
>   attention / 线性层 / 激活）与真实语料完全一致，用于效率基准无需下载数据；
>   若需真实预训练，替换 `SyntheticDataset` 为真实语料 DataLoader 即可，其余不变。
> - 效率指标口径：`TFLOPS = 6 × 参数量 × 全局每步 token 数 / 步耗时`；
>   `stable_*` 为剔除前若干步（预热/编译）后的均值。

### 3.2 `ds_config.json`（DeepSpeed ZeRO-3 配置）

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "zero_optimization": {
    "stage": 3,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "overlap_comm": true,
    "allgather_partitions": true,
    "reduce_scatter": true,
    "contiguous_gradients": true,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "bf16": {
    "enabled": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 3e-4,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.1
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 3e-4,
      "warmup_num_steps": 2
    }
  },
  "gradient_clipping": 1.0,
  "steps_per_print": 1,
  "wall_clock_breakdown": false
}
```

> `train_batch_size` 等三项由 `train.py` 根据 CLI 参数运行时注入，JSON 里写
> `"auto"` 占位即可。

---

## 4. 启动命令

统一前置（每个新 shell 都需要）：

```bash
cd ~/deepspeed
export PATH=$PWD/.venv/bin:$PATH          # ninja / deepspeed 可执行文件
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0                 # 本机 NVLS 不可用，必须禁用
```

### 4.1 冒烟验证（3 步，seq=512）

```bash
deepspeed --num_gpus 8 train.py \
    --ds-config ds_config.json \
    --steps 3 --seq-len 512 --batch-size 1 --grad-accum 2 \
    --log-file smoke.log.jsonl
```

### 4.2 正式基准（25 步，seq=4096，micro-batch=1）

> micro-batch=2 时峰值显存需求 ~95 GB 超过单卡 96 GB 上限（见 §2 坑 #5），
> 因此正式基准使用 micro-batch=1。全局 batch = 1 × 8 × 4 = 32 序列 × 4096 =
> 131,072 token/步。

```bash
# 前台运行（约 25 分钟）
deepspeed --num_gpus 8 train.py \
    --ds-config ds_config.json \
    --steps 25 --seq-len 4096 --batch-size 1 --grad-accum 4 \
    --warmup-steps 3 --log-file bench.log.jsonl

# 或后台运行（脱离会话，防中断）：
setsid nohup env NCCL_DEBUG=WARN NCCL_NVLS_ENABLE=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TRITON_CACHE_DIR=$PWD/.triton-cache \
    deepspeed --num_gpus 8 train.py \
    --ds-config ds_config.json \
    --steps 25 --seq-len 4096 --batch-size 1 --grad-accum 4 \
    --warmup-steps 3 --log-file bench.log.jsonl \
    > bench.out.log 2>&1 < /dev/null &
```

---

## 5. 基准结果（2026-08-20 实测）

### 5.1 正式基准汇总（8×H20，seq=4096，micro-bs=1，accum=4，25 步）

```
[config] model=~/models/Qwen3.8-27B params=26.90B gpus=8 micro_bs=1 accum=4
         seq=4096 global_batch_tokens=131072 steps=25 lr=0.0003 grad_ckpt=True
[load]   model+deepspeed init took 30.0s
[summary] {"params": 26895998464, "gpus": 8, "seq_len": 4096,
           "global_batch_tokens": 131072, "steps": 25, "total_tokens": 3276800,
           "elapsed_s": 1389.8, "avg_tokens_per_s": 2358,
           "stable_tokens_per_s": 2372, "stable_tflops": 382.7,
           "peak_mem_alloc_gb": 74.6, "load_time_s": 30.0}
```

| 指标 | 数值 |
|---|---|
| 模型参数量 | 26.90 B（bf16，54 GB 权重） |
| 全局 batch | 32 序列 × 4096 = 131,072 token/步 |
| 稳定吞吐 | **2,372 token/s（8 卡合计）** |
| 稳定算力 | **382.7 TFLOPS**（`6N×tokens/s` 口径） |
| MFU（按 H20 bf16 峰值 148 TFLOPS/卡） | 382.7 / (8×148) = **32.3%** |
| 单步耗时 | ~55 s（第 1 步 57 s，之后稳定 55 s 左右） |
| 峰值显存（alloc/reserved） | 74.6 / 85.2 GB 每卡（共 95.6 GiB 可用） |
| 全程 | 25 步 / 3,276,800 token / 1,389.8 s |
| 模型加载+DS 初始化 | 30.0 s |

逐步数据（节选，loss 为合成随机数据的交叉熵，基准仅关注效率指标）：

```
[step   1/25] loss=13.0465 2296 tok/s 370.5 TFLOPS 57.09s/step mem=58.3/73.0GB
[step   5/25] loss=16.8813 2371 tok/s 382.7 TFLOPS 55.27s/step mem=74.6/85.2GB
[step  10/25] loss=14.7502 2346 tok/s 378.6 TFLOPS 55.87s/step mem=74.6/85.2GB
[step  15/25] loss=12.7160 2347 tok/s 378.7 TFLOPS 55.86s/step mem=74.6/85.2GB
[step  20/25] loss=12.5568 2366 tok/s 381.8 TFLOPS 55.39s/step mem=74.6/85.2GB
[step  25/25] loss=12.5313 2350 tok/s 379.2 TFLOPS 55.79s/step mem=74.6/85.2GB
```

### 5.2 冒烟验证结果（3 步，seq=512，修复 engine.module.train() 之后）

```
[summary] {"params": 26895998464, "gpus": 8, "seq_len": 512,
           "global_batch_tokens": 8192, "steps": 3, "total_tokens": 24576,
           "elapsed_s": 22.7, "avg_tokens_per_s": 1084,
           "stable_tokens_per_s": 1216, "stable_tflops": 196.2,
           "peak_mem_alloc_gb": 61.2, "load_time_s": 26.8}
```

修复前同配置峰值显存 89.7 GB → 修复后 61.2 GB（checkpoint 生效）。

### 5.3 效率解读（供框架效率评估参考）

- **MFU 32.3%**：H20 为显存带宽取向卡（bf16 峰值 148 TFLOPS/卡），27B 规模下
  该数值在预期范围内；瓶颈主要在 GatedDeltaNet 的 torch chunked 实现
  （`torch_chunk_gated_delta_rule` 为 Python 循环，非融合 kernel）。若安装
  `fla`（flash-linear-attention）等融合 kernel，吞吐可进一步提升。
- 显存大头（74.6 GB/卡，95.6 GiB 可用）：参数分片 ~6.7 GB + Adam fp32 状态
  ~27 GB + 梯度 ~6.7 GB + 激活（checkpoint 下仍约 10+ GB） + 参数 gather 缓存 +
  logits（bs×4096×248,320×2B）。
- 通信：本机 NVLS 不可用（已禁用），走常规 ring allreduce，对 8 卡规模影响不大。

---

## 6. 快速复现步骤（Checklist）

1. 按 §1.2 创建环境并安装依赖（含 ninja、代理设置）。
2. 确认模型目录 `~/models/Qwen3.8-27B/` 存在。
3. 确认 8 卡空闲：`nvidia-smi`（每卡 used 接近 0）。
4. 按 §4 设置环境变量（PATH / NCCL_NVLS_ENABLE=0）。
5. 跑 §4.1 冒烟：预期 3 步全绿，`[summary]` 输出峰值显存 ~61 GB。
6. 跑 §4.2 正式基准：25 步约 25 分钟，逐步看 `tok/s` 与 `TFLOPS`；
   结束后 `[summary]` 即汇总指标；`bench.log.jsonl` 为逐步 JSONL 明细。
7. 换参对比（例如 batch/seq/accum），直接改 CLI 参数即可，无需改脚本。

---

## 7. 附录：文件清单

| 文件 | 用途 |
|---|---|
| `train.py` | 预训练/基准主脚本（§3.1 全文） |
| `ds_config.json` | ZeRO-3 配置（§3.2 全文） |
| `diag_mem.py` | 显存分阶段诊断脚本（§2 排障 #4 所用，定位 checkpoint 失效） |
| `train_qwen3_5_27b.md` | 本文档 |
| `smoke.log.jsonl` / `bench.log.jsonl` | 逐步指标明细（JSONL） |
| `bench*.out.log` | 各次运行完整 stdout/stderr |

