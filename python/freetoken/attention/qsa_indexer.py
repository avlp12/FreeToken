"""Qwen QSA indexer: 4-token block pooling, ReLU-sum scoring, token selection.

Numerical source of truth for the Qwen3.8-Flash-Next / qwen4_exp QSA indexer,
ported from HuggingFace ``Qwen4ExpTextQSAIndexer`` (transformers
``modeling_qwen4_exp.py``). The first version is PyTorch-only; Triton
kernelization lives under ``kernel/triton/qsa/`` as TODOs.

Official geometry (defaults)
----------------------------
* ``indexer_budget=2048``, ``compress_ratio=4``, ``n_heads=4``, ``kv_heads=1``,
  ``head_dim=128`` → ``block_topk = budget // ratio = 512``.
* ``index_qk_proj`` is fused q(4×128)+k(1×128). q is RMSNorm + RoPE in the
  **model**. k is the **raw** token key: stored as-is, then at score time
  4-token mean → ``k_layernorm`` (Qwen ``(1+w)`` RMSNorm) → RoPE at the
  **block-start position**.
* Score (fp32): ``sum_h relu(q_h · k_block) / sqrt(128)``.
* Selected set: tokens of the top-512 complete blocks, plus **every** token
  of the incomplete tail block. ``max_selected = 2048 + 4 - 1 = 2051``.
* Sequence length ``<= 2048+3``: every visible token is selected (dense).

Hole-free causal serving (this module's vectorized path) is identical to the
official "group the visible-index list by 4" rule: visible tokens are
``0..p`` so blocks are ``[0,1,2,3], [4,5,6,7], ...``. The naive reference
keeps the official per-query loop so a padded / holed mask can still be
checked against it.

Cache layout (linear, not paged)
--------------------------------
Raw keys: ``[n_slots, Di]`` addressed by the same physical rows as paged GQA
(``out_loc`` / page-table columns). At 256K tokens: ``256K * 128 * 2 B = 64 MiB``
per layer.

Pooled keys (post-ln, post-rope) are written at the block-start physical row
when a block completes. Decode updates that slot with a masked, fixed-shape
write so the path is CUDA-graph safe (see ``QSAIndexerKeyCache.update_pooled_decode``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class QSAIndexerConfig:
    """QSA indexer geometry. Official Qwen4-exp defaults."""

    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    rms_norm_eps: float = 1e-6

    @property
    def block_topk(self) -> int:
        return self.indexer_budget // self.indexer_compress_ratio

    @property
    def max_selected(self) -> int:
        # top-k complete blocks + longest incomplete tail (ratio-1 tokens)
        return self.indexer_budget + self.indexer_compress_ratio - 1

    @property
    def dense_token_limit(self) -> int:
        """Visible length at which the selected set is every token."""
        return self.max_selected

    @property
    def score_scale(self) -> float:
        return self.indexer_head_dim**-0.5

    @classmethod
    def from_model_config(cls, config: Any) -> "QSAIndexerConfig":
        args = getattr(config, "qwen4_exp_args", None)
        src = args if args is not None else config
        eps = getattr(src, "rms_norm_eps", None)
        if eps is None:
            eps = getattr(config, "rms_norm_eps", 1e-6)
        return cls(
            indexer_budget=int(getattr(src, "indexer_budget", 2048)),
            indexer_compress_ratio=int(getattr(src, "indexer_compress_ratio", 4)),
            indexer_n_heads=int(getattr(src, "indexer_n_heads", 4)),
            indexer_kv_heads=int(getattr(src, "indexer_kv_heads", 1)),
            indexer_head_dim=int(getattr(src, "indexer_head_dim", 128)),
            rms_norm_eps=float(eps),
        )


def qwen_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen4ExpTextRMSNorm: ``(x / rms) * (1 + w)`` in fp32, result in ``x.dtype``.

    The checkpoint stores ``w`` as ``scale - 1`` (zeros at init).
    """
    out = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    out = out * (1.0 + weight.float())
    return out.type_as(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_neox(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int | None = None,
) -> torch.Tensor:
    """HF ``apply_rotary_pos_emb`` (NeoX half-rotation) for one tensor.

    ``cos``/``sin`` last dim is the rotary width (full ``head_dim`` when there
    is no nope tail). Interleaved MRoPE is assumed already baked into the
    tables the model passes.
    """
    if unsqueeze_dim is not None:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    x_rope, x_nope = x[..., :rotary_dim], x[..., rotary_dim:]
    x_rope = (x_rope * cos) + (rotate_half(x_rope) * sin)
    if x_nope.numel() == 0:
        return x_rope
    return torch.cat([x_rope, x_nope], dim=-1)


def qsa_score_blocks(index_q: torch.Tensor, block_k: torch.Tensor) -> torch.Tensor:
    """fp32 ReLU-sum scores: ``index_q`` [..., H, D], ``block_k`` [..., B, D] → [..., B]."""
    # Official: matmul(q[H,D], k[B,D].T) → [H,B]; relu; sum heads; /sqrt(D).
    # Use math.sqrt (not ** -0.5) so the scale matches Qwen4ExpTextQSAIndexer.
    logits = torch.matmul(index_q.float(), block_k.float().transpose(-1, -2))
    return torch.relu(logits).sum(dim=-2) / math.sqrt(index_q.shape[-1])


def pool_complete_blocks(
    raw_k: torch.Tensor,
    k_ln_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    ratio: int,
    eps: float,
) -> torch.Tensor:
    """Mean-pool aligned complete blocks, then k_layernorm + block-start RoPE.

    ``raw_k`` is sequence-order ``[L, D]``. Only ``L // ratio`` complete blocks
    are produced. RoPE uses ``rope_cos/sin[block_start_pos]``.
    """
    n_blocks = raw_k.shape[0] // ratio
    if n_blocks == 0:
        return raw_k.new_empty(0, raw_k.shape[-1])
    dim = raw_k.shape[-1]
    groups = raw_k[: n_blocks * ratio].view(n_blocks, ratio, dim)
    pooled = groups.float().mean(dim=1).to(raw_k.dtype)
    pooled = qwen_rmsnorm(pooled, k_ln_weight, eps)
    starts = torch.arange(n_blocks, device=raw_k.device, dtype=torch.long) * ratio
    return apply_rotary_neox(pooled, rope_cos[starts], rope_sin[starts])


def qsa_select_tokens_naive(
    index_q: torch.Tensor,
    raw_k: torch.Tensor,
    visible: torch.Tensor,
    k_ln_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    cfg: QSAIndexerConfig,
) -> torch.Tensor:
    """Official O(B·Q) loop. ``index_q`` [B,Q,H,D], ``raw_k`` [B,K,D],
    ``visible`` [B,Q,K] bool, ``rope_cos/sin`` [B,K,D] or [K,D].

    Returns int32 ``[B, Q, max_selected]`` with ``-1`` padding. This is the
    parity-test reference; do not use it in the serving path.
    """
    if rope_cos.dim() == 2:
        rope_cos = rope_cos.unsqueeze(0).expand(index_q.shape[0], -1, -1)
        rope_sin = rope_sin.unsqueeze(0).expand(index_q.shape[0], -1, -1)
    batch, n_q, _, dim = index_q.shape
    ratio = cfg.indexer_compress_ratio
    max_sel = cfg.max_selected
    device = index_q.device
    selected = torch.full((batch, n_q, max_sel), -1, dtype=torch.int32, device=device)
    for b in range(batch):
        for qi in range(n_q):
            vis = torch.nonzero(visible[b, qi], as_tuple=False).flatten()
            n_complete = vis.numel() // ratio
            if n_complete > 0:
                block_tok = vis[: n_complete * ratio].view(n_complete, ratio)
                key_groups = raw_k[b].index_select(0, block_tok.flatten())
                key_groups = key_groups.view(n_complete, ratio, dim)
                pooled = key_groups.float().mean(dim=1).to(raw_k.dtype)
                pooled = qwen_rmsnorm(pooled, k_ln_weight, eps=cfg.rms_norm_eps)
                group_starts = block_tok[:, 0]
                block_k = apply_rotary_neox(
                    pooled.unsqueeze(1),
                    rope_cos[b].index_select(0, group_starts),
                    rope_sin[b].index_select(0, group_starts),
                    unsqueeze_dim=1,
                ).squeeze(1)
                scores = torch.matmul(
                    index_q[b, qi].float(), block_k.float().transpose(-1, -2)
                ).transpose(-1, -2)
                scores = torch.relu(scores).sum(dim=-1) / math.sqrt(dim)
                n_pick = min(cfg.block_topk, n_complete)
                pick = scores.topk(n_pick, dim=0).indices
                sel_tok = block_tok.index_select(0, pick).flatten()
            else:
                sel_tok = vis.new_empty(0, dtype=torch.long)
            tail = vis[n_complete * ratio :]
            sel_tok = torch.cat([sel_tok, tail]).to(torch.int32)
            selected[b, qi, : sel_tok.numel()] = sel_tok
    return selected


def qsa_select_tokens(
    index_q: torch.Tensor,
    raw_k: torch.Tensor,
    q_positions: torch.Tensor,
    k_ln_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    cfg: QSAIndexerConfig,
) -> torch.Tensor:
    """Vectorized hole-free causal selection. Math-equal to the official loop.

    ``index_q`` [Q, H, D] (already q_layernorm + RoPE), ``raw_k`` [K, D] raw
    sequence-order keys, ``q_positions`` [Q] absolute positions (query at ``p``
    sees tokens ``0..p``), ``rope_cos/sin`` [max_pos, D].

    Returns int32 ``[Q, max_selected]`` (``-1`` padded). Block tokens are in
    top-k rank order (official), tail tokens in position order.
    """
    n_q = index_q.shape[0]
    ratio = cfg.indexer_compress_ratio
    device = index_q.device
    block_k = pool_complete_blocks(
        raw_k, k_ln_weight, rope_cos, rope_sin, ratio, cfg.rms_norm_eps
    )
    n_blocks = block_k.shape[0]
    vis_len = q_positions.to(torch.long) + 1
    n_complete_q = vis_len // ratio
    max_sel = cfg.max_selected
    if n_blocks == 0:
        tail = _tail_tokens(n_complete_q, vis_len, ratio, n_q, device)
        pad = max_sel - tail.shape[-1]
        if pad > 0:
            tail = torch.cat(
                [tail, tail.new_full((n_q, pad), -1)], dim=-1
            )
        return tail

    scores = qsa_score_blocks(index_q, block_k)  # [Q, n_blocks]
    dead = torch.arange(n_blocks, device=device).unsqueeze(0) >= n_complete_q.unsqueeze(1)
    scores = scores.masked_fill(dead, float("-inf"))
    n_pick = min(cfg.block_topk, n_blocks)
    vals, idx = scores.topk(n_pick, dim=-1)
    valid = torch.isfinite(vals)
    offs = torch.arange(ratio, device=device)
    tok = idx.unsqueeze(-1) * ratio + offs
    tok = torch.where(valid.unsqueeze(-1), tok, tok.new_full((), -1)).reshape(n_q, n_pick * ratio)
    tail = _tail_tokens(n_complete_q, vis_len, ratio, n_q, device)
    out = torch.cat([tok.to(torch.int32), tail], dim=-1)
    if out.shape[-1] < max_sel:
        out = torch.cat(
            [out, out.new_full((n_q, max_sel - out.shape[-1]), -1)], dim=-1
        )
    return out[:, :max_sel]


def _tail_tokens(
    n_complete_q: torch.Tensor,
    vis_len: torch.Tensor,
    ratio: int,
    n_q: int,
    device: torch.device,
) -> torch.Tensor:
    """Incomplete tail: positions ``[n_complete*ratio, vis_len)``, padded to ``ratio-1``."""
    tail_w = ratio - 1
    if tail_w <= 0:
        return torch.empty(n_q, 0, dtype=torch.int32, device=device)
    base = n_complete_q * ratio
    offs = torch.arange(tail_w, device=device)
    pos = base.unsqueeze(1) + offs
    valid = pos < vis_len.unsqueeze(1)
    return torch.where(valid, pos, pos.new_full((), -1)).to(torch.int32)


def selected_bool_mask(selected: torch.Tensor, kv_len: int) -> torch.Tensor:
    """Official scatter: ``-1`` is absorbed into a dummy column and dropped.

    ``selected`` [..., S] int → bool mask [..., kv_len].
    """
    dummy = selected.new_zeros(*selected.shape[:-1], kv_len + 1)
    scatter = torch.where(selected >= 0, selected.long(), selected.new_full((), kv_len))
    dummy.scatter_(-1, scatter, 1)
    return dummy[..., :kv_len].bool()


def selection_sets(selected: torch.Tensor) -> list[set[int]]:
    """Per-row sets of valid (>=0) indices. ``selected`` is ``[..., S]``."""
    flat = selected.reshape(-1, selected.shape[-1])
    return [set(int(x) for x in row.tolist() if int(x) >= 0) for row in flat]


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Official ``repeat_kv``: [B, Hkv, S, D] → [B, Hq, S, D]."""
    if n_rep == 1:
        return hidden_states
    b, h, s, d = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(b, h, n_rep, s, d)
        .reshape(b, h * n_rep, s, d)
    )


def gqa_attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    *,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Eager GQA matching HF ``eager_attention_forward`` (fp32 softmax).

    ``q`` [B, Hq, Q, D], ``k``/``v`` [B, Hkv, K, D]. ``attn_mask`` is an
    additive float mask broadcastable to [B, 1|Hq, Q, K] (0 keep, ``-inf`` drop).
    Returns ``[B, Q, Hq, D]`` (official transpose of the head axis).
    """
    n_rep = q.shape[1] // k.shape[1]
    k_s = repeat_kv(k, n_rep)
    v_s = repeat_kv(v, n_rep)
    attn = torch.matmul(q, k_s.transpose(-2, -1)) * sm_scale
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn, v_s)
    return out.transpose(1, 2).contiguous()


def causal_additive_mask(
    q_positions: torch.Tensor,
    kv_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """[1, 1, Q, K] additive causal mask: keep ``k <= q_pos``."""
    k_pos = torch.arange(kv_len, device=device)
    keep = k_pos.unsqueeze(0) <= q_positions.to(device).unsqueeze(1)
    # -inf (not finfo.min): two masks may be added, and min+min overflows to
    # a different bit pattern than a single min.
    return torch.where(
        keep,
        torch.zeros((), dtype=dtype, device=device),
        torch.tensor(float("-inf"), dtype=dtype, device=device),
    ).view(1, 1, -1, kv_len)


class QSAIndexerKeyCache:
    """Linear raw-token + incremental pooled-block key store.

    ``raw_k[layer, row, Di]`` and ``pooled_k[layer, row, Di]`` share the paged
    GQA physical-row address space (``out_loc``). Pooled values are published
    at the **block-start** row, post-ln and post-rope, so decode scoring is a
    gather + ReLU-dot with no RoPE on the hot path.

    An extra dummy row (index ``n_slots``) absorbs CUDA-graph masked writes
    that must not land on a live row.
    """

    def __init__(
        self,
        num_layers: int,
        n_slots: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_layers = num_layers
        self.n_slots = n_slots
        self.dim = dim
        self.dtype = dtype
        self.raw_k = torch.zeros(num_layers, n_slots, dim, device=device, dtype=dtype)
        # +1 dummy row for graph-safe masked publishes
        self.pooled_k = torch.zeros(num_layers, n_slots + 1, dim, device=device, dtype=dtype)

    def store_raw(self, layer_id: int, keys: torch.Tensor, out_loc: torch.Tensor) -> None:
        self.raw_k[layer_id][out_loc.long()] = keys

    def publish_complete_blocks(
        self,
        layer_id: int,
        rows: torch.Tensor,
        kv_len: int,
        k_ln_weight: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        cfg: QSAIndexerConfig,
    ) -> None:
        """Eager (prefill) rebuild of every complete block for one request.

        ``rows`` is the request's position-ordered physical rows ``[L]``.
        """
        ratio = cfg.indexer_compress_ratio
        n_blocks = kv_len // ratio
        if n_blocks == 0:
            return
        block_rows = rows[: n_blocks * ratio].long().view(n_blocks, ratio)
        raw = self.raw_k[layer_id].index_select(0, block_rows.flatten()).view(
            n_blocks, ratio, self.dim
        )
        pooled = raw.float().mean(dim=1).to(self.dtype)
        pooled = qwen_rmsnorm(pooled, k_ln_weight, cfg.rms_norm_eps)
        starts = torch.arange(n_blocks, device=rows.device, dtype=torch.long) * ratio
        pooled = apply_rotary_neox(pooled, rope_cos[starts], rope_sin[starts])
        self.pooled_k[layer_id].index_copy_(0, block_rows[:, 0], pooled)

    def update_pooled_decode(
        self,
        layer_id: int,
        rows: torch.Tensor,
        pos: torch.Tensor,
        k_ln_weight: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        cfg: QSAIndexerConfig,
    ) -> None:
        """Fixed-shape decode publish of the just-completed block (or a dummy).

        Ordering / visibility: ``store_raw`` on this stream happens-before the
        gather below, so the current token's raw key is visible. A block ``b``
        is published only when ``(pos+1) % ratio == 0`` (i.e. ``kv_len`` just
        became a multiple of 4). Scoring reads pooled keys for blocks with
        ``(b+1)*ratio <= kv_len``, which is exactly the published set.
        Incomplete tail tokens are never read from ``pooled_k`` -- they are
        appended from ``kv_len`` arithmetic. Incomplete / padded requests
        write the dummy row (``n_slots``), so row 0 is never clobbered.
        All shapes are static (``[bs, ratio, D]``) → CUDA-graph safe.
        """
        ratio = cfg.indexer_compress_ratio
        start = (pos.long() // ratio) * ratio
        offs = torch.arange(ratio, device=pos.device)
        tok_pos = start.unsqueeze(1) + offs
        width = rows.shape[1]
        in_range = (tok_pos >= 0) & (tok_pos < width)
        gather_pos = tok_pos.clamp(0, width - 1)
        phys = rows.gather(1, gather_pos)
        phys = torch.where(in_range, phys, phys.new_full((), -1))
        keys = self.raw_k[layer_id][phys.clamp_min(0).long()]
        keys = keys * (phys >= 0).unsqueeze(-1).to(keys.dtype)
        pooled = keys.float().mean(dim=1)
        pooled = qwen_rmsnorm(pooled, k_ln_weight, cfg.rms_norm_eps)
        start_clamped = start.clamp(0, rope_cos.shape[0] - 1)
        pooled = apply_rotary_neox(pooled, rope_cos[start_clamped], rope_sin[start_clamped])
        complete = ((pos.long() + 1) % ratio) == 0
        dest = rows.gather(1, start.clamp(0, width - 1).unsqueeze(1)).squeeze(1)
        dest_idx = torch.where(
            complete & (dest >= 0) & (start < width),
            dest.long(),
            dest.new_full((), self.n_slots),
        )
        self.pooled_k[layer_id].index_copy_(0, dest_idx, pooled.to(self.dtype))

    def gather_pooled_blocks(
        self, layer_id: int, block_rows: torch.Tensor
    ) -> torch.Tensor:
        """``block_rows`` [bs, n_blocks] physical rows of block starts → [bs, n_blocks, D]."""
        safe = block_rows.clamp_min(0).long()
        gathered = self.pooled_k[layer_id][safe]
        return gathered * (block_rows >= 0).unsqueeze(-1).to(gathered.dtype)


__all__ = [
    "QSAIndexerConfig",
    "QSAIndexerKeyCache",
    "apply_rotary_neox",
    "causal_additive_mask",
    "gqa_attend",
    "pool_complete_blocks",
    "qsa_score_blocks",
    "qsa_select_tokens",
    "qsa_select_tokens_naive",
    "qwen_rmsnorm",
    "repeat_kv",
    "rotate_half",
    "selected_bool_mask",
    "selection_sets",
]
