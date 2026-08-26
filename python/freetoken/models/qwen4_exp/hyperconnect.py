from __future__ import annotations

import torch
from freetoken.layers import BaseOP

# Reference: transformers modeling_qwen4_exp.Qwen4ExpTextGatedResidual (the official
# implementation, read 2026-08-26). Every formula below mirrors it 1:1; this is the
# naive-correct P0 port (no fusion -- the DSV4 mHC fused kernel has a different weight
# layout, full-rank mix vs low-rank down/up here, and is ported in a later phase).
#
# The model runs an hc_count-wide residual stream: hidden state is [N, hc_count*hidden]
# end to end. Each sublayer (attention / MoE) gets a gated MIX of the streams as its
# 1x-hidden input and INJECTs its output back into every stream with a learned weight.


def grouped_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, group_size: int, eps: float
) -> torch.Tensor:
    """RMSNorm over each ``group_size`` slice of the last dim (fp32 math, like the
    reference). ``weight`` is the checkpoint weight with the Gemma-style +1 already
    baked in by the loader (reference computes ``normed * (1 + w)``)."""
    shape = x.shape
    xf = x.float().reshape(*shape[:-1], -1, group_size)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    out = xf.reshape(shape) * weight.float()
    return out.to(x.dtype)


class Qwen4GatedResidual(BaseOP):
    """Low-rank gated hyper-connection (hc_count=4, hc_lowrank=320).

        normed = grouped_rmsnorm(x4)                                # per-stream norm
        mix    = sigmoid(up(silu(down(normed) / hc_count)))         # [N, hc*H]
        mixed  = mean over streams of (mix * normed)                # [N, H] -> sublayer in
        inject = 2 * sigmoid(block_inject(normed) / hc_count)       # [N, hc]
        x4'    = x4 + sublayer_out[:, None, :] * inject[:, :, None] # flattened

    ``use_combine=False`` (the model-final ``hyper_connection_mixer``) has no
    block_inject_weight and returns only the mixed 1x-hidden tensor.
    """

    def __init__(self, hidden_size: int, hc_count: int, hc_lowrank: int, eps: float,
                 use_combine: bool = True):
        self.hidden_size = hidden_size
        self.hc_count = hc_count
        hc_hidden = hc_count * hidden_size
        self.eps = eps
        # Plain parameter tensors (bf16); shapes match the checkpoint exactly.
        self.hc_norm = _NormHolder(hc_hidden)
        self.input_mix_weight_down = _LinearHolder(hc_lowrank, hc_hidden)
        self.input_mix_weight_up = _LinearHolder(hc_hidden, hc_lowrank)
        self.block_inject_weight = _LinearHolder(hc_count, hc_hidden) if use_combine else None

    def _normed(self, x4: torch.Tensor) -> torch.Tensor:
        return grouped_rms_norm(x4, self.hc_norm.weight, self.hidden_size, self.eps)

    def mix(self, x4: torch.Tensor) -> torch.Tensor:
        """The ``use_combine=False`` path (final mixer): [N, hc*H] -> [N, H]."""
        normed = self._normed(x4)
        mixed, _ = self._mix_from_normed(normed)
        return mixed

    def _mix_from_normed(self, normed: torch.Tensor):
        w = torch.nn.functional.silu(
            torch.nn.functional.linear(normed, self.input_mix_weight_down.weight)
            / self.hc_count
        )
        w = torch.sigmoid(torch.nn.functional.linear(w, self.input_mix_weight_up.weight))
        w = w.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed = (w * normed.unflatten(-1, (self.hc_count, self.hidden_size))).mean(dim=-2)
        return mixed, normed

    def forward(self, x4: torch.Tensor):
        """Returns (mixed [N,H], inject_weights [N,hc]). Residual base is the RAW x4."""
        assert self.block_inject_weight is not None
        normed = self._normed(x4)
        mixed, _ = self._mix_from_normed(normed)
        inject = 2.0 * torch.sigmoid(
            torch.nn.functional.linear(normed, self.block_inject_weight.weight)
            / self.hc_count
        )
        return mixed, inject

    def combine(self, x4: torch.Tensor, sublayer_out: torch.Tensor,
                inject: torch.Tensor) -> torch.Tensor:
        """x4 + sublayer_out broadcast into each stream by its injection weight."""
        injected = sublayer_out.unsqueeze(-2) * inject.unsqueeze(-1)  # [N, hc, H]
        return x4 + injected.flatten(-2).to(x4.dtype)


class _NormHolder(BaseOP):
    def __init__(self, dim: int):
        self.weight = torch.empty(dim)


class _LinearHolder(BaseOP):
    """Bare weight holder yielding the checkpoint key ``<name>.weight`` ([out, in])."""

    def __init__(self, out_features: int, in_features: int):
        self.weight = torch.empty(out_features, in_features)


__all__ = ["Qwen4GatedResidual", "grouped_rms_norm"]
