"""Fake-quantization (round-trip) simulation for ds_fp4 expert banks.

This is a QUALITY probe, not a speed/format change: weights stay in the exact same
ds_fp4 storage (e2m1 nibble pairs packed 2/byte + one ``float8_e8m0fnu`` group-of-32
scale) -- only the VALUES get degraded, as if a real sub-4-bit kernel had produced
them, so we can measure the damage before investing in an actual packed k-bit format.

Enabled by the env var ``FREETOKEN_FAKEQUANT_BITS`` (a float, e.g. ``2``, ``2.5``,
``3``; unset or ``0`` is off). When on, :func:`apply_fakequant_to_banks` runs, once,
right after the ds_fp4 expert banks are fully loaded (see
``freetoken.moe.expert_banks._dsfp4_banks``): for every expert row of every bank
(gate_up and down, every layer) it

  1. dequantizes the existing e2m1 + e8m0 values to fp32 (:func:`dequant_e2m1_e8m0`,
     mirroring ``freetoken.kernel.triton.dsv4.fused_moe``'s decode: ``W = LUT[code] *
     2**(e8m0_code - 127)``),
  2. simulates a k-bit round-trip on those fp32 values (:func:`simulate_kbit`), and
  3. re-encodes the result back to e2m1 + e8m0 (:func:`encode_e2m1_e8m0`), overwriting
     the SAME pinned host bank buffers in place -- shapes, dtypes, and the storage
     format itself are untouched; only the bit patterns inside the existing bytes
     change.

Simulated k-bit format
-----------------------
Symmetric absmax uniform ("mid-rise", no zero level) quantizer, grouped by 32 along
the SAME axis the e8m0 scale already groups by 32 on. For a group of real values with
``amax = max(|x_i|)``, normalize ``u_i = x_i / amax`` and pick the nearest of ``L``
levels evenly spaced across ``[-1, 1]``:

    L = round(2**k)              (>= 2)
    j = clamp(round((u + 1) * (L - 1) / 2), 0, L - 1)
    q = -1 + j * 2 / (L - 1)
    x_hat = q * amax

For integer k this is exactly ``2**k`` signed levels with no zero (e.g. k=2 -> L=4 ->
levels {-1, -1/3, +1/3, +1} * amax, matching the "symmetric int levels" scheme this
was speced from). Fractional k (e.g. k=2.5) is realized as ``L = round(2**k)`` levels
directly (2.5 -> 6 levels) rather than alternating 2-bit/3-bit groups -- simpler,
single code path, and an equally defensible reading of a non-power-of-two bit budget.

OPTIMISTIC BIAS: the k-bit group scale (``amax``) is kept at full fp32 precision for
this simulation. A real k-bit kernel would have to spend a few extra bits per group to
store that scale too (fp16, e4m3, ...), which this does NOT charge for -- so the
measured quality here is a slight upper bound (best case) on what a real packed k-bit
format would achieve at the same nominal bit budget.

e2m1/e8m0 codec
----------------
:func:`encode_e2m1_e8m0` picks its own e8m0 group scale (the smallest power of two
``2**p`` with ``amax <= 6 * 2**p``, 6 being the largest E2M1 magnitude) using
``torch.frexp`` -- an EXACT power-of-two decomposition -- rather than
``ceil(log2(amax/6))``: the latter is fine almost everywhere, but at exactly two of
the eight E2M1 magnitudes (1.5 and 3.0) ``amax/6`` lands exactly on a power of two, and
a 1-ULP log2 rounding error there would silently pick the wrong scale and corrupt the
"k=0 is a no-op" / "decode(encode(decode(x))) == decode(x)" identities this module is
built on (empirically verified in ``/root/probe_numerics.py`` against a brute-force
reference across magnitudes x scale range -20..20, and via full grid self-consistency).
"""

from __future__ import annotations

import os
import time

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

# The 8 positive E2M1 magnitudes (index 0-7 of the 4-bit code) and the negative mirror
# (index 8-15) -- identical table to freetoken.kernel.triton.dsv4.fused_moe._E2M1_VALUES
# / freetoken.kernel.triton.nvfp4_dequant._E2M1_VALUES, mirrored here in plain torch (no
# triton/CUDA dependency) so this module also runs CPU-only for unit tests.
_E2M1_MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Nearest-neighbor decision boundaries between consecutive magnitudes (7 of them for 8
# levels); a value's absolute magnitude bucketizes into 0..7, exactly the low 3 bits of
# the e2m1 code (the sign becomes the 4th bit separately).
_E2M1_MID = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)

GROUP = 32  # ds_fp4's e8m0 block size (one scale byte per 32 packed values)

_ENV_VAR = "FREETOKEN_FAKEQUANT_BITS"
_MIN_BITS = 1.5   # below this the e2m1 grid (already ~3-4 effective bits) isn't the bottleneck
_MAX_BITS = 4.0   # at/above this there's nothing left to simulate -- ds_fp4 already IS ~4 bits


def fakequant_bits_from_env() -> float:
    """Parse ``FREETOKEN_FAKEQUANT_BITS``; ``0.0`` means off (unset, ``0``, unparseable,
    or outside the sane ``[1.5, 4.0)`` range -- each of the latter two warns once)."""
    raw = os.environ.get(_ENV_VAR, "").strip()
    if not raw:
        return 0.0
    try:
        k = float(raw)
    except ValueError:
        logger.warning_rank0(f"{_ENV_VAR}={raw!r} is not a number; fakequant disabled")
        return 0.0
    if k == 0.0:
        return 0.0
    if k < _MIN_BITS or k >= _MAX_BITS:
        logger.warning_rank0(
            f"{_ENV_VAR}={k} is outside the supported [{_MIN_BITS}, {_MAX_BITS}) range; "
            "fakequant disabled"
        )
        return 0.0
    return k


def fakequant_requested() -> bool:
    """True iff a valid, in-range ``FREETOKEN_FAKEQUANT_BITS`` is set."""
    return fakequant_bits_from_env() > 0.0


def _e2m1_lut(device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    vals = list(_E2M1_MAG) + [-v for v in _E2M1_MAG]
    return torch.tensor(vals, dtype=dtype, device=device)


def _as_u8(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def dequant_e2m1_e8m0(packed: torch.Tensor, scale) -> torch.Tensor:
    """``[..., K//2]`` uint8 (e2m1 nibbles) + ``[..., K//32]`` e8m0 codes -> ``[..., K]``
    fp32. Bit-for-bit the same numerics as the decode kernels: ``W = LUT[code] *
    2**(e8m0_code - 127)``, low nibble -> even output element."""
    assert packed.dtype == torch.uint8, packed.dtype
    scale_u8 = _as_u8(scale)
    lut = _e2m1_lut(packed.device)
    lo = (packed & 0x0F).long()
    hi = (packed >> 4).long()
    val_lo = lut[lo]
    val_hi = lut[hi]
    out = torch.stack((val_lo, val_hi), dim=-1).flatten(-2)  # [..., K]
    s = torch.exp2(scale_u8.to(torch.float32) - 127.0)
    s = s.repeat_interleave(GROUP, dim=-1)
    return out * s


def _scale_exponent_for_group(amax: torch.Tensor) -> torch.Tensor:
    """Smallest integer ``p`` with ``amax <= 6 * 2**p`` (``0`` when ``amax == 0``).

    Derived via ``torch.frexp`` (exact ``amax = mant * 2**exp``, ``0.5 <= mant < 1``)
    instead of ``ceil(log2(amax / 6))`` to sidestep float rounding at the two exact
    power-of-two boundaries (see module docstring). For ``mant <= 0.75``: ``mant * 8 <=
    6`` iff ``mant <= 0.75``, and no larger power works (``mant * 16 >= 8 > 6``), so
    ``p = exp - 3``. Otherwise ``mant * 4 < 4 <= 6`` always holds and ``mant * 8 > 6``
    always fails, so ``p = exp - 2``.
    """
    mant, exp = torch.frexp(amax)
    q = torch.where(mant <= 0.75, torch.full_like(exp, 3), torch.full_like(exp, 2))
    p = exp - q
    return torch.where(amax > 0, p, torch.zeros_like(p))


def encode_e2m1_e8m0(x: torch.Tensor, group: int = GROUP) -> tuple[torch.Tensor, torch.Tensor]:
    """``[..., K]`` float -> (``[..., K//2]`` uint8 packed e2m1, ``[..., K//group]``
    uint8 e8m0 codes). Exact inverse of :func:`dequant_e2m1_e8m0` for values already on
    the e2m1 * e8m0 grid -- see module docstring for why this round-trips exactly."""
    *lead, K = x.shape
    assert K % group == 0, (K, group)
    G = K // group
    xg = x.reshape(*lead, G, group).to(torch.float32)
    amax = xg.abs().amax(dim=-1, keepdim=True)
    p = _scale_exponent_for_group(amax)
    p = torch.clamp(p, -127, 127)  # e8m0 code range [0, 254]; 255 is reserved (NaN)
    code = (p + 127).to(torch.uint8)
    scale_pow2 = torch.exp2(p.to(torch.float32))
    scale_safe = torch.where(amax > 0, scale_pow2, torch.ones_like(scale_pow2))
    r = xg / scale_safe
    mid = torch.tensor(_E2M1_MID, dtype=torch.float32, device=x.device)
    idx = torch.bucketize(r.abs(), mid)  # nearest of the 8 magnitudes, in [0, 7]
    sign = (r < 0).to(torch.uint8) * 8
    nibble = (idx.to(torch.uint8) + sign).reshape(*lead, K)
    lo = nibble[..., 0::2]
    hi = nibble[..., 1::2]
    packed = lo | (hi << 4)
    scale_u8 = code.reshape(*lead, G)
    return packed, scale_u8


def _levels_for_bits(k: float) -> int:
    return max(2, int(round(2.0 ** k)))


def simulate_kbit(x: torch.Tensor, k: float, group: int = GROUP) -> torch.Tensor:
    """Round ``x`` (``[..., K]`` fp32) through the simulated k-bit format described in
    the module docstring, grouped by ``group`` along the last axis. Pure fp32 torch, no
    packing -- the caller re-encodes the result to real e2m1 + e8m0 afterwards."""
    L = _levels_for_bits(k)
    *lead, K = x.shape
    assert K % group == 0, (K, group)
    G = K // group
    xg = x.reshape(*lead, G, group)
    amax = xg.abs().amax(dim=-1, keepdim=True)
    amax_safe = torch.where(amax > 0, amax, torch.ones_like(amax))
    u = xg / amax_safe
    j = torch.round((u + 1.0) * (L - 1) / 2.0)
    j = torch.clamp(j, 0.0, float(L - 1))
    q = -1.0 + j * (2.0 / (L - 1))
    x_hat = torch.where(amax > 0, q * amax_safe, torch.zeros_like(xg))
    return x_hat.reshape(*lead, K)


_DEFAULT_CHUNK_EXPERTS = 8
_MAX_CHUNK_EXPERTS = 16
_MIN_CHUNK_EXPERTS = 1
_LIVE_TENSORS_PER_CHUNK = 5  # dequant x, simulate intermediates, encode intermediates (rough)
_VRAM_HEADROOM = 512 << 20   # leave this much free for whatever else is resident (e.g. a live server)


def _pick_chunk_experts(bytes_per_expert_fp32: int, device: torch.device) -> int:
    """Row-block size (in experts) for one dequant/simulate/encode round, sized to the
    ACTUAL free VRAM at call time (a live server may be resident, leaving only a few GiB
    free) rather than a fixed constant. CPU device: fixed default (no VRAM to query)."""
    if device.type != "cuda" or bytes_per_expert_fp32 <= 0:
        return _DEFAULT_CHUNK_EXPERTS
    free, _total = torch.cuda.mem_get_info(device)
    budget = max(free - _VRAM_HEADROOM, 0) // _LIVE_TENSORS_PER_CHUNK
    n = int(budget // bytes_per_expert_fp32)
    return max(_MIN_CHUNK_EXPERTS, min(_MAX_CHUNK_EXPERTS, n))


def _transform_bank_inplace(packed: torch.Tensor, scale: torch.Tensor, k: float, device: torch.device) -> None:
    """One (packed, scale) bank pair for one layer, chunked over the expert axis."""
    E, N, K_half = packed.shape
    K = K_half * 2
    bytes_per_expert_fp32 = N * K * 4
    chunk = _pick_chunk_experts(bytes_per_expert_fp32, device)
    for lo in range(0, E, chunk):
        hi = min(lo + chunk, E)
        p_chunk = packed[lo:hi].to(device, non_blocking=True)
        s_chunk = _as_u8(scale[lo:hi]).to(device, non_blocking=True)
        x = dequant_e2m1_e8m0(p_chunk, s_chunk)
        x = simulate_kbit(x, k)
        new_packed, new_scale = encode_e2m1_e8m0(x)
        packed[lo:hi].copy_(new_packed.to("cpu"))
        scale[lo:hi].copy_(new_scale.to("cpu").view(scale.dtype))
        del p_chunk, s_chunk, x, new_packed, new_scale
        if device.type == "cuda":
            torch.cuda.empty_cache()


_BANK_PAIRS = (("gate_up_packed", "gate_up_scale"), ("down_packed", "down_scale"))


def apply_fakequant_to_banks(banks: dict[str, list[torch.Tensor]], *, device: torch.device | None = None) -> bool:
    """Apply the fake-quant round-trip in place to every layer of a ds_fp4
    ``{bank_name: [per_layer_tensor]}`` dict (the shape ``load_dsfp4_expert_sources`` /
    ``load_dsfp4_expert_sources_parallel`` return). No-op, buffers untouched, unless
    ``FREETOKEN_FAKEQUANT_BITS`` is set to a valid in-range value. Returns whether it ran.
    """
    k = fakequant_bits_from_env()
    if k <= 0.0:
        return False
    for name in ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"):
        if name not in banks:
            raise ValueError(f"apply_fakequant_to_banks: missing bank {name!r}")
    n_layers = len(banks["gate_up_packed"])
    if device is None:
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    t0 = time.time()
    for layer in range(n_layers):
        for packed_name, scale_name in _BANK_PAIRS:
            _transform_bank_inplace(banks[packed_name][layer], banks[scale_name][layer], k, device)
    dt = time.time() - t0
    logger.info_rank0(f"fakequant: k={k} bits applied to {n_layers} layers x 4 banks in {dt:.1f} s")
    return True


__all__ = [
    "GROUP",
    "apply_fakequant_to_banks",
    "dequant_e2m1_e8m0",
    "encode_e2m1_e8m0",
    "fakequant_bits_from_env",
    "fakequant_requested",
    "simulate_kbit",
]
