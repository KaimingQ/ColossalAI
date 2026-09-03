# Adaptation of Shardformer MoE expert parallelism for DeepSeek-V4 (transformers >= 5.x)
#
# Differences vs DeepseekV3:
# - transformers 5.x stores experts as fused 3D tensors (gate_up_proj / down_proj)
#   in `DeepseekV4Experts`, instead of a ModuleList of expert MLPs.
# - The MoE block forward takes an extra `input_ids` argument (hash router layers).
#
# This module wraps a native DeepseekV4SparseMoeBlock, slices the expert tensors
# along the expert dimension across the EP group, and implements all-to-all based
# token dispatch/combine (same scheme as EpDeepseekV3MoE).

from typing import Optional

import gc

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup
from torch.nn import Module

from colossalai.lazy import LazyInitContext
from colossalai.moe._operation import (
    DPGradScalerIn,
    DPGradScalerOut,
    EPGradScalerIn,
    EPGradScalerOut,
    all_to_all_uneven,
)
from colossalai.shardformer.layer.linear import ParallelModule
from colossalai.tensor.moe_tensor.api import set_moe_tensor_ep_group


class V4Expert(nn.Module):
    """单个专家 (gate/up/down), 参数形状与融合张量的单专家切片一致。"""

    def __init__(self, hidden_dim: int, intermediate_dim: int, act_fn, limit: float, dtype=None):
        super().__init__()
        # 注意: 在 LazyInitContext 内创建时, torch.empty 被拦截为懒初始化;
        # dtype 必须与融合权重一致 (否则 F.linear dtype mismatch)
        self.gate_up_proj = nn.Parameter(torch.empty(2 * intermediate_dim, hidden_dim, dtype=dtype))
        self.down_proj = nn.Parameter(torch.empty(hidden_dim, intermediate_dim, dtype=dtype))
        self.act_fn = act_fn
        self.limit = limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = F.linear(x, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return F.linear(self.act_fn(gate) * up, self.down_proj)


class V4ExpertsList(nn.Module):
    """逐专家 ModuleList 形式的专家集合 (替代融合 3D 张量), 便于专家并行切分与按需加载。"""

    def __init__(self, fused):
        super().__init__()
        self.num_experts = fused.num_experts
        self.hidden_dim = fused.hidden_dim
        self.intermediate_dim = fused.intermediate_dim
        self.act_fn = fused.act_fn
        self.limit = fused.limit
        self.experts = nn.ModuleList(
            [
                V4Expert(self.hidden_dim, self.intermediate_dim, self.act_fn, self.limit,
                         dtype=fused.gate_up_proj.dtype)
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        # 无专家并行时的朴素实现 (逐专家循环), 训练路径使用 EpDeepseekV4MoE
        final = torch.zeros_like(hidden_states)
        for e, expert in enumerate(self.experts):
            mask = top_k_index == e
            if not mask.any():
                continue
            token_idx, pos = mask.nonzero(as_tuple=True)
            out = expert(hidden_states[token_idx]) * top_k_weights[token_idx, pos, None]
            final.index_add_(0, token_idx, out.to(final.dtype))
        return final


def convert_fused_experts_to_modulelist(model: nn.Module, copy_weights: bool = True) -> None:
    """将 DeepseekV4Experts (融合 3D 参数) 原地替换为 V4ExpertsList。

    训练路径: 在 LazyInitContext 内调用, 新参数经 torch.empty 懒初始化,
    权重由逐专家键的 checkpoint 加载, 使专家并行只加载/保存本 rank 的 1/ep 专家。
    copy_weights: 融合参数已物化时是否拷贝权重到逐专家参数 (懒张量时自动跳过)。
    """
    from colossalai.lazy.lazy_init import LazyTensor

    for layer in model.model.layers:
        mlp = layer.mlp
        if mlp.experts.__class__.__name__ == "DeepseekV4Experts":
            fused = mlp.experts
            new_experts = V4ExpertsList(fused)
            if copy_weights and not isinstance(fused.gate_up_proj, LazyTensor):
                with torch.no_grad():
                    for e in range(fused.num_experts):
                        new_experts.experts[e].gate_up_proj.copy_(fused.gate_up_proj[e])
                        new_experts.experts[e].down_proj.copy_(fused.down_proj[e])
            mlp.experts = new_experts


class EpDeepseekV4MoE(ParallelModule):
    """Expert-parallel DeepSeek-V4 Sparse MoE block."""

    def __init__(self, config):
        raise RuntimeError(f"Please use `from_native_module` to create an instance of {self.__class__.__name__}")

    def setup_process_groups(self, moe_dp_group: ProcessGroup, ep_group: ProcessGroup):
        assert moe_dp_group is not None
        assert ep_group is not None

        self.ep_size = dist.get_world_size(ep_group)
        self.ep_rank = dist.get_rank(ep_group)
        self.num_experts = self.experts.num_experts
        assert self.num_experts % self.ep_size == 0, "num_experts must be divisible by ep_size"

        self.ep_group = ep_group
        self.num_experts_per_ep = self.num_experts // self.ep_size
        self.experts_per_rank = self.num_experts_per_ep
        self.expert_start_idx = self.ep_rank * self.num_experts_per_ep

        # 逐专家 ModuleList: 仅保留本 rank 持有的专家, 其余参数置 None (自包含实现,
        # 不使用 set_tensors_to_none, 避免其递归影响 gate 等其他参数),
        # 使物化与权重加载只覆盖 1/ep 的专家, 实现真正的专家显存切分
        # 注意: 保持 ModuleList 的全局索引布局 (梯度规约按全局专家号), 仅置空非持有专家参数
        held = set(range(self.expert_start_idx, self.expert_start_idx + self.num_experts_per_ep))
        for i, expert in enumerate(self.experts.experts):
            if i in held:
                continue
            for pname in list(expert._parameters.keys()):
                expert._parameters[pname] = None

        # setup moe_dp group
        self.moe_dp_group = moe_dp_group
        self.moe_dp_size = dist.get_world_size(moe_dp_group)

        for p in self.experts.parameters():
            set_moe_tensor_ep_group(p, ep_group)

    def _held_expert_params(self):
        """本 rank 持有的逐专家参数列表 (fuse 前置检查用)。"""
        if getattr(self, "fused_gate_up", None) is not None:
            return []
        if self.ep_size <= 1 or self.moe_dp_size != 1 or not hasattr(torch, "_grouped_mm"):
            return []
        params = []
        for i in range(self.expert_start_idx, self.expert_start_idx + self.num_experts_per_ep):
            e = self.experts.experts[i]
            if e.gate_up_proj is not None:
                params.append(e.gate_up_proj)
            if e.down_proj is not None:
                params.append(e.down_proj)
        return params

    def fuse_local_experts(self):
        """权重加载完成后, 将本 rank 持有的专家融合为 3D 冻结权重 (grouped GEMM 布局,
        参照 Megatron TEGroupedMLP): 训练 forward 用 torch._grouped_mm 一次算完全部本地
        专家, 替代逐专家小 GEMM 循环 (消除 kernel launch 风暴)。
        仅适用于 EP 且 moe_dp_size==1 (DPGradScaler 在 dp=1 时为 no-op, grouped 路径无法逐专家缩放);
        buffer persistent=False, 不进 state_dict (仅适配只保存 adapter 的训练)。
        调用前需先由 fuse_v4_local_experts 清理 optimizer/ZeRO 对原参数的引用。
        """
        if getattr(self, "fused_gate_up", None) is not None:
            return False
        if self.ep_size <= 1 or self.moe_dp_size != 1:
            return False
        if not hasattr(torch, "_grouped_mm"):
            return False
        held = [
            self.experts.experts[i]
            for i in range(self.expert_start_idx, self.expert_start_idx + self.num_experts_per_ep)
        ]
        params = [(e.gate_up_proj, e.down_proj) for e in held]
        gate_up = torch.stack([p.data for p, _ in params])  # [E_local, 2I, H]
        down = torch.stack([d.data for _, d in params])  # [E_local, H, I]
        for e in held:  # 释放逐专家原参数 (外部引用已由 fuse_v4_local_experts 预先清理)
            for pname in ("gate_up_proj", "down_proj"):
                p = e._parameters[pname]
                # LazyTensor 物化时绑定 tolist=MethodType(_data_tolist, tensor) 形成自引用环,
                # 仅减引用计数不会释放, 需打断环 (残余环由后续 gc.collect 兜底)
                try:
                    del p.tolist
                except AttributeError:
                    pass
                e._parameters[pname] = None
        del params
        # 释放的小块碎片无法被 caching allocator 合并为下一层 stack 所需大块,
        # 逐层归还未用缓存避免碎片累积 OOM (43 层一次性开销 ~数秒)
        torch.cuda.empty_cache()
        self.register_buffer("fused_gate_up", gate_up, persistent=False)
        self.register_buffer("fused_down", down, persistent=False)
        return True

    def unfuse_local_experts(self) -> bool:
        """保存前将 fused buffer 还原为逐专家参数布局 (state_dict 兼容,
        DPO/ORPO 等保存完整 EP 分片的训练路径在 save 前调用)。逐层 clone+删 buffer, 峰值 +1 层。"""
        if getattr(self, "fused_gate_up", None) is None:
            return False
        held = self.experts.experts[self.expert_start_idx : self.expert_start_idx + self.num_experts_per_ep]
        for e, gu, dn in zip(held, self.fused_gate_up, self.fused_down):
            e._parameters["gate_up_proj"] = nn.Parameter(gu.clone(), requires_grad=False)
            e._parameters["down_proj"] = nn.Parameter(dn.clone(), requires_grad=False)
        del self._buffers["fused_gate_up"]
        del self._buffers["fused_down"]
        torch.cuda.empty_cache()
        return True

    def _grouped_local_forward(self, gathered: torch.Tensor, local_counts_list) -> torch.Tensor:
        """grouped GEMM 计算全部本地专家。_grouped_mm 要求每组行数为 8 的倍数
        (bf16 16B 对齐), 故将每组 pad 到 8 倍数 (平均 ~11% 冗余计算, 远小于
        逐专家小 GEMM 的效率损失与 launch 开销); pad/unpad 全部 GPU 向量化, 无额外同步。"""
        device = gathered.device
        num_rows = gathered.shape[0]
        counts = torch.tensor(local_counts_list, dtype=torch.int64, device=device)
        counts_pad = (counts + 7) // 8 * 8
        total_pad = sum((n + 7) // 8 * 8 for n in local_counts_list)  # CPU 侧已知, 免同步
        offs_pad = torch.cumsum(counts_pad, 0).to(torch.int32)

        # 原序列行 -> pad 序列目标位置: dst = pad_start[seg] + row_in_seg
        seg_id = torch.repeat_interleave(
            torch.arange(self.experts_per_rank, device=device), counts
        )
        orig_start = torch.cumsum(counts, 0) - counts
        pad_start = offs_pad.long() - counts_pad
        dst = pad_start[seg_id] + (torch.arange(num_rows, device=device) - orig_start[seg_id])

        pad = gathered.new_zeros(total_pad, gathered.shape[-1])
        pad[dst] = gathered
        limit = self.experts.limit
        gu = torch._grouped_mm(pad, self.fused_gate_up.transpose(1, 2), offs=offs_pad)
        gate, up = gu.chunk(2, dim=-1)
        act = self.experts.act_fn(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
        out = torch._grouped_mm(act, self.fused_down.transpose(1, 2), offs=offs_pad)
        return out[dst]

    @staticmethod
    def from_native_module(module: Module, moe_dp_group: ProcessGroup, ep_group: ProcessGroup, *args, **kwargs):
        # 鸭子类型: 需要 gate / experts(V4ExpertsList) / shared_experts 结构
        assert hasattr(module, "gate") and hasattr(module, "experts") and hasattr(module, "shared_experts"), (
            f"EpDeepseekV4MoE expects a SparseMoE-block-like module, got {module.__class__.__name__}"
        )
        assert module.experts.__class__.__name__ == "V4ExpertsList", (
            "请先调用 convert_fused_experts_to_modulelist(model) 将融合专家转为逐专家布局"
        )
        if module.__class__ is not EpDeepseekV4MoE:
            module.__class__ = EpDeepseekV4MoE
            module.setup_process_groups(moe_dp_group, ep_group)
        LazyInitContext.materialize(module)
        return module

    def forward(self, hidden_states: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states
        flat = hidden_states.view(-1, hidden_dim)
        if self.is_hash:
            # hash router needs token-level input_ids aligned with flattened hidden states;
            # 打包/不对齐时回退为 topk 打分 (保证训练不崩, 该 3 层路由退化为可学习形式)
            ids = input_ids.reshape(-1) if input_ids is not None else None
            if ids is not None and ids.numel() == flat.shape[0]:
                _, weights, indices = self.gate(hidden_states, ids)
            else:
                logits = torch.nn.functional.linear(flat, self.gate.weight)
                scores = self.gate.score_fn(logits)
                indices = torch.topk(scores, self.gate.top_k, dim=-1, sorted=False).indices
                weights = scores.gather(1, indices)
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
                weights = weights * self.gate.routed_scaling_factor
        else:
            _, weights, indices = self.gate(hidden_states)
        routed = self.ep_experts_forward(flat, indices, weights).view(batch, seq_len, hidden_dim)
        return routed + self.shared_experts(residual)

    def _expert_forward(self, x: torch.Tensor, e: int) -> torch.Tensor:
        """单个专家计算: e 为局部专家下标, 存储位置为全局专家号, x [n, H] -> [n, H]"""
        return self.experts.experts[e + self.expert_start_idx](x)

    def ep_experts_forward(
        self, x: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """All-to-all expert-parallel MoE computation. x: [T, H] flattened tokens."""
        num_tokens = x.shape[0]
        k = top_k_index.shape[-1]
        device = x.device

        # token counts per global expert
        counts = torch.bincount(top_k_index.reshape(-1), minlength=self.num_experts)

        if self.ep_size > 1:
            counts_grouped = counts.view(self.ep_size, self.experts_per_rank).sum(dim=1)
            recv_counts = torch.empty_like(counts_grouped)
            dist.all_to_all_single(recv_counts, counts_grouped, group=self.ep_group)
            send_splits = counts_grouped.tolist()
            recv_splits = recv_counts.tolist()
        else:
            recv_counts = counts
            send_splits = recv_splits = [int(num_tokens)]

        # sort tokens by (global expert id)
        flat_idx = top_k_index.reshape(-1)
        sort_order = flat_idx.argsort(stable=True)
        token_order = sort_order // k
        sorted_tokens = x[token_order]
        sorted_expert = flat_idx[sort_order]

        if self.ep_size > 1:
            # dispatch tokens to owning EP ranks
            gathered, _ = all_to_all_uneven(sorted_tokens, send_splits, recv_splits, self.ep_group)
            expert_gathered, _ = all_to_all_uneven(sorted_expert.unsqueeze(-1), send_splits, recv_splits, self.ep_group)
            expert_gathered = expert_gathered.squeeze(-1) - self.expert_start_idx  # local expert id

            # group received tokens by local expert
            local_order = expert_gathered.argsort(stable=True)
            gathered = gathered[local_order]
            local_counts = torch.bincount(expert_gathered, minlength=self.experts_per_rank)

            # moe-dp: 本 rank 局部专家的激活情况 (跨 DP 组规约), 供梯度缩放
            activate_experts = (local_counts > 0).int()
            dist.all_reduce(activate_experts, group=self.moe_dp_group)

            # 一次性取回 CPU (2 次同步), 消除专家循环内逐专家 int()/item() 的
            # GPU->CPU 同步风暴 (每层 ~100 次 -> 4 次); 同步会打断流水线并放大
            # 后续集合通信的 rank 间等待
            local_counts_list = local_counts.tolist()
            act_list = activate_experts.tolist()

            gathered = EPGradScalerIn.apply(gathered, self.ep_size)
        else:
            local_counts_list = counts.tolist()
            act_list = None
            gathered = sorted_tokens

        # per-local-expert grouped computation
        if self.ep_size > 1 and getattr(self, "fused_gate_up", None) is not None:
            # grouped GEMM 快路径 (见 fuse_local_experts)
            local_out = self._grouped_local_forward(gathered, local_counts_list)
            outputs = None
        else:
            outputs = []
            start = 0
            for e in range(self.experts_per_rank):
                n = local_counts_list[e]
                if n == 0:
                    continue
                toks = gathered[start : start + n]
                if self.ep_size > 1:
                    toks = DPGradScalerIn.apply(toks, self.moe_dp_size, act_list[e])
                out_e = self._expert_forward(toks, e)  # [n, H]
                if self.ep_size > 1:
                    out_e = DPGradScalerOut.apply(out_e, self.moe_dp_size, act_list[e])
                outputs.append(out_e)
                start += n

        if outputs is not None:
            if outputs:
                local_out = torch.cat(outputs, dim=0)
            else:
                local_out = gathered[:0]

        if self.ep_size > 1:
            local_out = EPGradScalerOut.apply(local_out, self.ep_size)
            # local_order 将接收序列按本地专家分组计算, 回传前必须做逆置换恢复接收顺序;
            # 否则源 rank 收到错位的专家输出 (token 拿到别的专家的结果), 训练 loss 会异常偏高。
            # 参照 EpDeepseekV3MoE 的 new_x[gatherd_idxs] = outs 还原逻辑。
            unsorted = torch.empty_like(local_out)
            unsorted[local_order] = local_out
            # combine: scatter results back to source ranks (与 dispatch 相反的拆分)
            combined, _ = all_to_all_uneven(unsorted, recv_splits, send_splits, self.ep_group)
            # combined 已恢复为 sorted-token 顺序, 按路由权重加权
            w = top_k_weights.reshape(-1)[sort_order]
            final_sorted = combined * w[:, None]
        else:
            w = top_k_weights.reshape(-1)[sort_order]
            final_sorted = local_out * w[:, None]

        # 恢复原始 token 顺序 (同一 token 的 k 个专家输出累加)
        final = torch.zeros(num_tokens, x.shape[-1], dtype=final_sorted.dtype, device=device)
        final.index_add_(0, token_order, final_sorted)
        return final


def _strip_freed_params(container, freed_ids, seen) -> None:
    """递归清理 optimizer / ZeRO wrapper 内指向已释放专家参数的引用:
    param_groups (HybridAdam 在 boost 前收集了全部 base 参数) 与
    LowLevelZeroOptimizer.pg_to_param_list (按 DP 组存储的全量参数列表)。"""
    if container is None or id(container) in seen:
        return
    seen.add(id(container))
    groups = getattr(container, "param_groups", None)
    if isinstance(groups, list):
        for g in groups:
            if isinstance(g, dict) and isinstance(g.get("params"), list):
                g["params"] = [p for p in g["params"] if id(p) not in freed_ids]
    pg_map = getattr(container, "pg_to_param_list", None)
    if isinstance(pg_map, dict):
        for k, v in pg_map.items():
            if isinstance(v, list):
                pg_map[k] = [p for p in v if id(p) not in freed_ids]
    # 通用清理: param_to_pg / *_to_*_param 等以 param 对象为键或值的映射
    cvars = vars(container) if hasattr(container, "__dict__") else {}
    for name, attr in list(cvars.items()):
        if isinstance(attr, dict) and attr:
            stale_keys = [k for k, v in attr.items() if id(k) in freed_ids or id(v) in freed_ids]
            for k in stale_keys:
                del attr[k]
    for inner_name in ("optim", "optimizer"):
        inner = getattr(container, inner_name, None)
        if inner is not None and inner is not container:
            _strip_freed_params(inner, freed_ids, seen)


def fuse_v4_local_experts(model: nn.Module, optimizer=None) -> int:
    """对模型内所有 EpDeepseekV4MoE 块融合本地专家权重 (权重加载完成后调用)。
    optimizer: boost 后的 optimizer (ZeRO wrapper)。两阶段执行:
    先从 optimizer/ZeRO 容器清理对逐专家参数的引用 (HybridAdam 在 boost 前收集了
    全部 base 参数, 不清理则释放无效导致 OOM), 再逐层 stack+释放+归还缓存。"""
    blocks = [m for m in model.modules() if isinstance(m, EpDeepseekV4MoE)]
    held = []
    for m in blocks:
        held.extend(m._held_expert_params())
    if not held:
        return 0
    if optimizer is not None:
        _strip_freed_params(optimizer, {id(p) for p in held}, set())
    del held

    n = 0
    for m in blocks:
        if m.fuse_local_experts():
            n += 1
    gc.collect()  # 回收残余的自引用环 (tolist 绑定方法等)
    torch.cuda.empty_cache()
    return n


def unfuse_v4_local_experts(model: nn.Module) -> int:
    """保存前还原逐专家参数布局 (与 fuse_v4_local_experts 配对, 供完整 EP 分片保存)。"""
    n = 0
    for m in model.modules():
        if isinstance(m, EpDeepseekV4MoE) and m.unfuse_local_experts():
            n += 1
    return n


def v4_fast_eager_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """DeepSeek-V4 eager attention 的提速等价实现 (head_dim=512 无法走 SDPA/FA):
    1. MQA 广播: key/value 为单 KV 头 [B,1,Skv,D], matmul 自动广播,
       省去 HF repeat_kv 的 num_heads 倍物理拷贝 (每层 2×67MB @ seq512);
    2. sinks 融合: HF 将 per-head sink logit cat 到 [.., S+1] 后整体 softmax 再丢列;
       这里等价地并入 max/分母 (denom = Σexp + exp(sink)), 省掉 cat 拷贝与第二次遍历。
    数值与 HF eager_attention_forward 一致 (bf16 舍入噪声级)。
    """
    sinks = kwargs.get("s_aux", None)
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    if sinks is not None:
        sk = sinks.reshape(1, -1, 1, 1)
        m = torch.maximum(attn_weights.amax(dim=-1, keepdim=True), sk)
        p = torch.exp(attn_weights - m)
        denom = p.sum(dim=-1, keepdim=True) + torch.exp(sk - m)
        attn_weights = p / denom
    else:
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=attn_weights.dtype)
    if dropout > 0.0 and module.training:
        attn_weights = F.dropout(attn_weights, p=dropout, training=True)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def install_v4_fast_attention() -> None:
    """monkeypatch transformers 的 deepseek_v4 eager attention 为提速等价实现。
    Attention.forward 的 get_interface(_attn_implementation, eager_attention_forward)
    在运行时解析模块全局名, 故 patch 模块属性即生效 (仅限 eager 路径)。"""
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as mv4

    if getattr(mv4, "_v4_fast_attn_installed", False):
        return
    mv4.eager_attention_forward = v4_fast_eager_attention
    mv4._v4_fast_attn_installed = True
