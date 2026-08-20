# Qwen3.5 27B Framework Benchmark — ColossalAI vs DeepSpeed

Pretraining-efficiency benchmark of **Qwen3.8-27B** (26.9B params, `qwen3_5` architecture,
GatedDeltaNet linear attention) on **8× NVIDIA H20** (96 GB), comparing **ColossalAI 0.5.0**
(TorchFSDPPlugin, ZeRO-3) against **DeepSpeed 0.19.5** (ZeRO-3) under the *exact same*
configuration: seq=4096, micro-bs=1, grad-accum=4, 25 steps, synthetic data, bf16,
gradient checkpointing.

> This directory is the DEMO material: runnable scripts + measured result records.
> Model checkpoint: `~/models/Qwen3.8-27B` (BF16, 54 GB) and `~/models/Qwen3.8-27B-FP8`.

## Result summary (2026-08-20, same config, 8×H20)

Three ColossalAI configurations vs DeepSpeed, all seq=4096 / micro-bs=1 / accum=4 / 25 steps:

| Metric | DeepSpeed ZeRO-3 | ColossalAI FSDP | ColossalAI Gemini (opt) |
|---|---|---|---|
| Stable throughput | **2,372 tok/s** | 2,039 tok/s | 1,832 tok/s |
| Stable compute | **382.7 TFLOPS** | 329.0 TFLOPS | 295.7 TFLOPS |
| Peak GPU mem (alloc) | 74.6 GB | **65.6 GB** | 82.1 GB |
| Load + parallel init | 30.0 s | **15.8 s** | 33.4 s |

Takeaways:
- **DeepSpeed wins throughput** (+16% over FSDP, +30% over Gemini).
- **ColossalAI FSDP wins memory** (-12%) and **init speed** (2× faster).
- **Gemini plugin** successfully adapted to GatedDeltaNet (conv1d chunk fix),
  optimized with `max_prefetch=2` + fused_norm + jit_fused + async_reduce → **+6.6%**.
  Gemini's chunk management overhead (fetch→cast→compute→cast→reduce→return) limits throughput.

## Files

| File | Purpose |
|---|---|
| `colossalai_train.py` | ColossalAI benchmark script (`--plugin gemini\|fsdp`, torchrun launch) |
| `train.py` | DeepSpeed benchmark script (same metrics, `deepspeed` launch) |
| `ds_config.json` | DeepSpeed ZeRO-3 config |
| `diag_mem.py` | staged GPU-memory diagnostic (used to locate the eval-mode gradient-checkpointing bug) |
| `colossalai_vs_deepspeed.md` | full comparison doc: env setup, adaptation log, scripts, results, analysis |
| `train_qwen3_5_27b.md` | DeepSpeed reproduction doc: env, scripts, commands, results, troubleshooting |

## Quick start (ColossalAI side)

```bash
cd ~/deepspeed && uv venv --python 3.12 venv_colossalai
uv pip install --python venv_colossalai/bin/python torch==2.5.1 \
    numpy transformers==5.15.1 safetensors sentencepiece ninja einops packaging tqdm
export PYTHONPATH=$HOME/ColossalAI   # ColossalAI 0.5.0 source
export PATH=$PWD/venv_colossalai/bin:$PATH
export NCCL_NVLS_ENABLE=0            # required on this box (NVLS unsupported)

# smoke test
torchrun --nproc_per_node=8 --master_port=29502 colossalai_train.py \
    --plugin fsdp --steps 3 --seq-len 512 --batch-size 1 --grad-accum 2

# official benchmark (25 steps, ~27 min)
torchrun --nproc_per_node=8 --master_port=29502 colossalai_train.py \
    --plugin fsdp --steps 25 --seq-len 4096 --batch-size 1 --grad-accum 4 \
    --warmup-steps 3 --log-file cola_bench.log.jsonl
```

DeepSpeed side: `deepspeed --num_gpus 8 train.py --ds-config ds_config.json
--steps 25 --seq-len 4096 --batch-size 1 --grad-accum 4 --warmup-steps 3` (same config).

## Adaptation notes (ColossalAI × this architecture)

- **Gemini plugin (ColossalAI-specific optimization) is NOT compatible with the
  GatedDeltaNet `causal_conv1d`**: Gemini's chunked virtual storage conflicts with the
  conv1d weight reshape/setStorage (`storage of size 0` at first forward). Workaround:
  use `TorchFSDPPlugin` (native PyTorch FSDP, same ZeRO-3 semantics). To enable Gemini,
  the conv1d layers must be excluded from chunk management or re-implemented.
- Gemini only accepts ColossalAI optimizers → use `HybridAdam(adamw_mode=True)`.
- `booster.boost` wraps the model (no `.config`) → save `vocab_size` before boost.
- torch 2.5.1 `transformer_auto_wrap_policy` → pass `partial(transformer_auto_wrap_policy,
  transformer_layer_cls={Qwen3_5DecoderLayer})`.


