# Shardformer policy for DeepSeek-V4 (transformers >= 5.x native implementation).
# Modeled after policies/deepseek_v3.py; pipeline parallelism is not supported yet
# (transformers 5.x V4 forward signature differs), EP + ZeRO/CPU-offload is the target.

from typing import Dict, List, Union

import torch.nn as nn

from colossalai.shardformer.layer import FusedRMSNorm
from colossalai.shardformer.modeling.deepseek_v4 import EpDeepseekV4MoE
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["DeepseekV4Policy", "DeepseekV4ModelPolicy", "DeepseekV4ForCausalLMPolicy"]


class DeepseekV4Policy(Policy):
    def config_sanity_check(self):
        assert not self.shard_config.enable_tensor_parallelism, "DeepSeekV4 does not support tensor parallelism"
        assert not self.shard_config.enable_sequence_parallelism, "DeepSeekV4 does not support sequence parallelism"
        assert self.shard_config.pipeline_stage_manager is None, (
            "DeepSeekV4 does not support pipeline parallelism yet"
        )

    def preprocess(self):
        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        policy = {}

        if self.shard_config.expert_parallel_size > 1:
            # expert parallel: replace the Sparse MoE block (suffix "mlp" of decoder layer)
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="mlp",
                        target_module=EpDeepseekV4MoE,
                        kwargs={
                            "ep_group": self.shard_config.ep_group,
                            "moe_dp_group": self.shard_config.moe_dp_group,
                        },
                    )
                ],
                policy=policy,
                target_key="DeepseekV4DecoderLayer",
            )

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="input_layernorm",
                        target_module=FusedRMSNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="post_attention_layernorm",
                        target_module=FusedRMSNorm,
                    ),
                ],
                policy=policy,
                target_key="DeepseekV4DecoderLayer",
            )

            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="norm",
                    target_module=FusedRMSNorm,
                ),
                policy=policy,
                target_key="DeepseekV4Model",
            )

        return policy

    def postprocess(self):
        return self.model

    def get_held_layers(self) -> List[nn.Module]:
        raise NotImplementedError("DeepSeekV4 does not support pipeline parallelism yet")


class DeepseekV4ModelPolicy(DeepseekV4Policy):
    def module_policy(self):
        return super().module_policy()


class DeepseekV4ForCausalLMPolicy(DeepseekV4Policy):
    def module_policy(self):
        return super().module_policy()
