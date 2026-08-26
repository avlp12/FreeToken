from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    make_moe_layer,
    silu_and_mul,
)

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class _SharedExpert(BaseOP):
    """Always-on shared SwiGLU expert. Deliberately bf16-only: the Qwen3.8-FP8 release
    whitelists the shared expert out of quantization (no weight_scale_inv in the
    checkpoint), so unlike qwen3_5 this must NOT follow config.expert_quant."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size, [intermediate_size, intermediate_size], has_bias=False
        )
        self.down_proj = LinearRowParallel(intermediate_size, hidden_size, has_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class Qwen4ExpMoE(BaseOP):
    """512 routed experts (fp8_block banks, offload cache) top-10 + sigmoid-gated shared
    expert. Router: softmax over ALL experts -> top-k -> renormalize (norm_topk_prob=True
    per the transformers Qwen4ExpTextConfig default; HF evaluates shared expert first)."""

    def __init__(self, config: ModelConfig, layer_id: int | None = None):
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            weight_format="fp8_block",
        )
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _SharedExpert(
            config.hidden_size, config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Router + shared expert BEFORE the routed experts: the fused MoE kernel may
        # write into hidden_states in place (same ordering note as qwen3_5).
        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        shared = shared * torch.sigmoid(self.shared_expert_gate.forward(hidden_states))
        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        return (routed + shared).view(num_tokens, hidden_dim)


__all__ = ["Qwen4ExpMoE"]
