"""显存分阶段诊断：定位 90GB+ 显存占用来源（静态/前向/反向）。"""
import json
import torch
import deepspeed
import deepspeed.comm as dist
from transformers import Qwen3_5ForCausalLM

MD = "/home/qukaiming/models/Qwen3.8-27B"


def mem(label, rank):
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    print(f"[{rank}][{label}] alloc={a:.2f}GB resv={r:.2f}GB", flush=True)


def main():
    deepspeed.init_distributed()
    rank = dist.get_rank()
    local = dist.get_local_rank()
    torch.cuda.set_device(local)

    mem("start", rank)
    model = Qwen3_5ForCausalLM.from_pretrained(MD, torch_dtype=torch.bfloat16,
                                               low_cpu_mem_usage=True)
    model.config.use_cache = False
    mem("after_load(cpu)", rank)
    model.gradient_checkpointing_enable()
    print(f"[{rank}] grad_ckpt_flag={model.is_gradient_checkpointing}", flush=True)

    with open("ds_config.json") as f:
        cfg = json.load(f)
    world = dist.get_world_size()
    cfg["train_micro_batch_size_per_gpu"] = 1
    cfg["gradient_accumulation_steps"] = 1
    cfg["train_batch_size"] = 1 * world * 1
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(),
                                           config_params=cfg)
    mem("after_ds_init", rank)
    print(f"[{rank}] model.training={engine.module.training}", flush=True)
    print(f"[{rank}] layer0.gradient_checkpointing={engine.module.model.layers[0].gradient_checkpointing}",
          flush=True)
    print(f"[{rank}] layer0.training={engine.module.model.layers[0].training}", flush=True)

    # 若处于 eval 模式，先切回训练模式再测
    if not engine.module.training:
        engine.module.train()
        print(f"[{rank}] -> set train() done, training={engine.module.training}", flush=True)
        torch.cuda.empty_cache()
        mem("after_train()", rank)

    vocab = model.config.vocab_size
    ids = torch.randint(0, vocab, (1, 512), dtype=torch.long).to("cuda")

    # --- checkpoint 开启 ---
    out = engine(input_ids=ids, labels=ids)
    mem("fwd_ckpt_ON bs1_seq512", rank)
    out.loss.backward()
    mem("bwd_ckpt_ON", rank)
    engine.zero_grad()
    torch.cuda.empty_cache()

    # --- checkpoint 关闭 ---
    engine.module.gradient_checkpointing_disable()
    print(f"[{rank}] ckpt_after_disable={engine.module.is_gradient_checkpointing}", flush=True)
    out = engine(input_ids=ids, labels=ids)
    mem("fwd_ckpt_OFF bs1_seq512", rank)
    out.loss.backward()
    mem("bwd_ckpt_OFF", rank)
    engine.zero_grad()
    torch.cuda.empty_cache()

    # --- 大输入（仅 ckpt ON） ---
    ids = torch.randint(0, vocab, (2, 4096), dtype=torch.long).to("cuda")
    try:
        out = engine(input_ids=ids, labels=ids)
        mem("fwd_ckpt_ON bs2_seq4096", rank)
        out.loss.backward()
        mem("bwd_ckpt_ON bs2_seq4096", rank)
    except RuntimeError as e:
        print(f"[{rank}] OOM at bs2_seq4096: {str(e)[:120]}", flush=True)

    dist.barrier()


if __name__ == "__main__":
    main()
