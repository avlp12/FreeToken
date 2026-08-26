from __future__ import annotations

import torch
from freetoken.layers import BaseOP
from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet

# Qwen4-Exp's GatedDeltaNet is byte-for-byte the qwen3_5 recurrence (same in_proj split,
# same conv+silu, same beta/g params, l2norm in kernel; verified against transformers
# modeling_qwen4_exp.Qwen4ExpTextGatedDeltaNet) with EXACTLY ONE delta: the gated output
# norm's activation is config.output_gate_type ("sigmoid" for Qwen3.8-Flash-Next) instead
# of qwen3_5's hardcoded silu. We subclass and swap the norm; weight keys are unchanged
# (linear_attn.norm.weight, raw multiply -- NOT Gemma +1).


class _SigmoidGatedRMSNorm(BaseOP):
    """norm(x) * sigmoid(z), fp32 math like the reference RMSNormGated (naive P0 port;
    the fused fla rms_norm_gated kernel only ships a silu path)."""

    def __init__(self, dim: int, eps: float):
        self.weight = torch.empty(dim)
        self.eps = eps

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        out = self.weight * xf.to(dtype)
        return (out * torch.sigmoid(z.float()).to(dtype)).to(dtype)


class Qwen4GatedDeltaNet(Qwen3_5GatedDeltaNet):
    def __init__(self, *args, output_gate_type: str = "sigmoid", **kwargs):
        super().__init__(*args, **kwargs)
        if output_gate_type == "sigmoid":
            eps = self.norm.eps
            self.norm = _SigmoidGatedRMSNorm(self.head_v_dim, eps=eps)
        elif output_gate_type not in ("silu", "swish"):
            raise NotImplementedError(f"output_gate_type={output_gate_type!r}")


__all__ = ["Qwen4GatedDeltaNet"]
