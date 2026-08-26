from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, GemmaRMSNorm, LinearColParallelMerged, LinearReplicated
from freetoken.layers.rotary import get_rope
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class _DormantIndexer(BaseOP):
    """QSA indexer weights, LOADED but UNUSED in P0.

    For sequences <= indexer_budget (2048) the reference indexer selects every visible
    token (all complete blocks fit the budget, the incomplete tail is always included),
    so dense attention is bit-exact there. The P1 sparse backend activates these:
    index_qk_proj [(n_heads + kv_heads) * head_dim, hidden], q/k_layernorm [head_dim]
    (Gemma-style, +1 baked by the loader)."""

    def __init__(self, config: ModelConfig):
        args = config.qwen4_args
        out = (args.indexer_n_heads + args.indexer_kv_heads) * args.indexer_head_dim
        self.index_qk_proj = LinearReplicated(config.hidden_size, out, has_bias=False)
        self.q_layernorm = GemmaRMSNorm(args.indexer_head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = GemmaRMSNorm(args.indexer_head_dim, eps=config.rms_norm_eps)


class Qwen4ExpAttention(BaseOP):
    """Gated full attention (same scheme as qwen3_5): per-head output gate folded into a
    2x-wide q projection, q/k Gemma RMSNorm, partial NeoX rope (rotary_dim 64 of 256).
    Dense bf16 linears -- the FP8 release keeps all attention weights bf16 (only routed
    experts are fp8). QSA indexer weights ride along dormant (see _DormantIndexer)."""

    def __init__(self, config: ModelConfig, layer_id: int):
        head_dim = config.head_dim
        self.layer_id = layer_id
        self.num_q = config.num_qo_heads
        self.num_kv = config.num_kv_heads
        self.head_dim = head_dim
        self.qo_attn_dim = self.num_q * head_dim
        self.kv_attn_dim = self.num_kv * head_dim

        # Fused q|k|v projection; the q half is 2x for the output gate (per-head layout
        # [q(head_dim) | gate(head_dim)] -- matches HF's view+chunk).
        self._qkv_split = [self.num_q * head_dim * 2, self.kv_attn_dim, self.kv_attn_dim]
        self.qkv_proj = LinearColParallelMerged(
            config.hidden_size, self._qkv_split, has_bias=False
        )
        self.q_norm = GemmaRMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(head_dim, eps=config.rms_norm_eps)
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=config.rotary_config.rotary_dim,
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
            rope_scaling=(
                tuple(config.rotary_config.scaling.items())
                if config.rotary_config.scaling
                else None
            ),
        )
        self.o_proj = LinearReplicated(self.qo_attn_dim, config.hidden_size, has_bias=False)
        self.indexer = _DormantIndexer(config)

    def _project(self, x: torch.Tensor):
        positions = get_global_ctx().batch.positions
        qkv = self.qkv_proj.forward(x)
        qg, k, v = torch.split(qkv, self._qkv_split, dim=-1)
        qg = qg.view(-1, self.num_q, self.head_dim * 2)
        q = qg[..., : self.head_dim].contiguous()
        gate = qg[..., self.head_dim :].reshape(-1, self.qo_attn_dim)
        k = k.view(-1, self.num_kv, self.head_dim).contiguous()
        v = v.contiguous()
        q = self.q_norm.forward(q).reshape(-1, self.qo_attn_dim)
        k = self.k_norm.forward(k).reshape(-1, self.kv_attn_dim)
        q, k = self.rotary.forward(positions, q, k)
        return q.view(-1, self.num_q, self.head_dim), k, v, gate

    def _combine(self, attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gated = attn_out.reshape(-1, self.qo_attn_dim) * torch.sigmoid(gate)
        return self.o_proj.forward(gated)

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        q, k, v, gate = self._project(x)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        return self._combine(o, gate)


__all__ = ["Qwen4ExpAttention"]
