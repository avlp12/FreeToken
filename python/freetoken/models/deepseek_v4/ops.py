"""Small math ops for DeepSeek-V4 (pure torch, faithful to the reference).

These are cheap, low-arithmetic ops kept in torch (the heavy GEMM/attention paths
are Triton). The reference's FP8/FP4 *activation* quant simulations and the
Hadamard rotation are applied on the compressor/attention paths via the dsv4
Triton kernels (``fp8_linear.act_quant_fp8_inplace`` / ``fp4_act_quant_inplace``
and ``hadamard.hadamard_transform``; see ``compress.py`` / ``attention.py``), so
precision matches the reference.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch


def precompute_freqs_cis(
    dim: int, seqlen: int, original_seq_len: int, base: float,
    factor: float, beta_fast: int, beta_slow: int,
) -> torch.Tensor:
    """YaRN rotary frequencies (verbatim from the reference)."""

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(mn, mx, dim):
        if mn == mx:
            mx += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - mn) / (mx - mn)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


@lru_cache(2)  # exactly the model's two rope regimes; a stale 1M entry pins ~256 MB of VRAM
def get_freqs_cis(
    dim: int, seqlen: int, original_seq_len: int, base: float,
    factor: float, beta_fast: int, beta_slow: int, device: torch.device,
) -> torch.Tensor:
    """Device-resident ``freqs_cis``, shared across layers (the DSV4 analogue of
    ``layers.rotary.get_rope``'s instance cache).

    All 43 layers use one of two rope regimes (windowed vs compressed), and at 1M
    positions each table is ~256 MB -- per-layer ``.to(device)`` copies would cost
    ~11 GB after the KV pool has already claimed its memory budget."""
    return precompute_freqs_cis(
        dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow
    ).to(device)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """In-place interleaved rotary embedding (verbatim from the reference)."""
    y = x
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if xc.ndim == 3:
        freqs_cis = freqs_cis.view(1, xc.size(1), xc.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
    xr = torch.view_as_real(xc * freqs_cis).flatten(-2)
    y.copy_(xr)
    return y


def apply_rotary_emb_decode(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    inverse: bool = False,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """In-place interleaved rotary for batched DECODE: PER-ROW freqs.

    ``x`` is ``[B, 1, ..., rd]`` (one decode token per row), ``freqs_cis`` is ``[B, rd//2]``
    complex (each row's own position). Broadcasts the row's freqs over the inner (head) dim.
    At B=1 this is identical to ``apply_rotary_emb`` (the shared bs-broadcast view coincides).

    With ``positions`` (int64 ``[B]``), ``freqs_cis`` is instead the FULL table and the
    kernel gathers each row's frequencies itself, deleting the caller's
    ``index_select`` launch. Same entries, same arithmetic -- see
    ``kernel/triton/dsv4/rope.py``.

    Fused into one Triton kernel (borrowed from sglang's deepseek_v4_rope): the prior torch path
    (``view_as_complex`` -> complex mul -> ``view_as_real`` -> ``copy_``) was ~3-5 small at::native
    kernels per call, called for q/kv/o in every layer. Same interleaved fp32 complex math, stored
    back to ``x``'s dtype -> bit-identical (parity-gated)."""
    from freetoken.kernel.triton.dsv4.norm_rope import unfused_norm_rope
    from freetoken.kernel.triton.dsv4.rope import rope_decode_inplace

    if positions is not None and unfused_norm_rope():
        freqs_cis = freqs_cis.index_select(0, positions)
        positions = None
    return rope_decode_inplace(x, freqs_cis, inverse, positions)


def rms_norm_rope_decode(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    *,
    heads: int = 1,
    inverse: bool = False,
) -> torch.Tensor:
    """``rms_norm`` then decode RoPE over the last ``rope_dim``, in one launch.

    ``freqs_cis`` is the full table; the kernel gathers it by ``positions``.
    ``heads`` is the number of rows per batch element (``x`` is ``[B, 1, heads, D]``
    or ``[B, 1, D]``) -- how a flattened row finds the token position to rotate by.

    ``FREETOKEN_UNFUSED_NORM_ROPE=1`` takes the three-launch composition below,
    which is also the reference the bit-identity test compares against."""
    from freetoken.kernel.triton.dsv4.norm import rms_norm
    from freetoken.kernel.triton.dsv4.norm_rope import (
        rms_norm_rope_decode as _fused,
        unfused_norm_rope,
    )

    if unfused_norm_rope():
        y = rms_norm(x, weight, eps)
        apply_rotary_emb_decode(
            y[..., -rope_dim:], freqs_cis.index_select(0, positions), inverse
        )
        return y
    return _fused(
        x, weight, eps, freqs_cis, positions, rope_dim, heads=heads, inverse=inverse
    )


def hc_split_sinkhorn(
    mixes: torch.Tensor,      # [n, (2+hc)*hc] fp32
    hc_scale: torch.Tensor,   # [3]
    hc_base: torch.Tensor,    # [(2+hc)*hc]
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
):
    """Split the hyper-connection mix into (pre, post, comb), Sinkhorn-normalizing
    ``comb`` into a doubly-stochastic mixing matrix. Torch port of the tilelang kernel.

    Returns ``pre[n,hc]``, ``post[n,hc]``, ``comb[n,hc,hc]`` (all fp32).
    """
    hc = hc_mult
    mixes = mixes.float()
    sc = hc_scale.float()
    base = hc_base.float()
    pre = torch.sigmoid(mixes[:, :hc] * sc[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[:, hc:2 * hc] * sc[1] + base[hc:2 * hc])
    comb = mixes[:, 2 * hc:] * sc[2] + base[2 * hc:]
    comb = comb.view(-1, hc, hc)
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


__all__ = ["precompute_freqs_cis", "apply_rotary_emb", "hc_split_sinkhorn"]
