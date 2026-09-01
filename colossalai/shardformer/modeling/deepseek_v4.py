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
        sorted_tokens = x[sort_order // k]
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

            gathered = EPGradScalerIn.apply(gathered, self.ep_size)
        else:
            local_counts = counts
            activate_experts = None
            gathered = sorted_tokens

        # per-local-expert grouped computation
        outputs = []
        start = 0
        for e in range(self.experts_per_rank):
            n = int(local_counts[e])
            if n == 0:
                continue
            toks = gathered[start : start + n]
            if self.ep_size > 1:
                toks = DPGradScalerIn.apply(toks, self.moe_dp_size, activate_experts[e])
            out_e = self._expert_forward(toks, e)  # [n, H]
            if self.ep_size > 1:
                out_e = DPGradScalerOut.apply(out_e, self.moe_dp_size, activate_experts[e])
            outputs.append(out_e)
            start += n

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
        final.index_add_(0, sort_order // k, final_sorted)
        return final
