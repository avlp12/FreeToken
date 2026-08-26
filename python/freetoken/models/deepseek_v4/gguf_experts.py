"""Native GGUF IQ-quant routed-expert banks for DeepSeek-V4 (unsloth UD-Q2_K_XL).

The ``ds_fp4`` sibling of this loader (:mod:`freetoken.models.deepseek_v4.weight`)
reads the authors' FP4 safetensors; this one reads a llama.cpp / unsloth GGUF whose
routed experts are stored in *several* ggml types at once -- IQ2_XS on most
``ffn_gate_exps``/``ffn_up_exps``, IQ3_XXS on a few, IQ3_XXS on most
``ffn_down_exps`` and MXFP4 on the rest.

One bank cannot mix row strides, so the ``q2_k_ud`` schema
(:data:`freetoken.moe.offload_cache._BANK_SCHEMAS`) stores BOTH banks at a uniform
pitch wide enough for the widest type that bank holds -- ``row_bytes(H, IQ3_XXS)``
for gate_up, ``row_bytes(I, MXFP4)`` for down -- and every narrower native row is
copied into the row prefix with the tail left zero. ``ggml_moe_a8_vec``'s
``row_pitch_bytes`` then addresses those padded rows while decoding each row at its
own (per-layer) ggml type, so a per-layer quant-type side table travels with the
banks (see :func:`load_q2k_ud_expert_sources`'s return value).

Nothing here changes the numbers any more: every row is a byte copy of the GGUF's
own bytes. Until the CUDA kernels learned MXFP4 (ggml type 39) the two down layers
unsloth stored at 4.25 bpw did not fit the 784 B IQ3_XXS pitch, and they were
dequantized and re-encoded to Q2_K RTN (672 B, ~0.28 relative RMS) to squeeze in --
discarding the precision on precisely the layers it had been spent on. The down
pitch is now 1088 B and those rows pass through untouched. :func:`quantize_q2_k`
survives because the round-trip is still worth testing, not because anything calls
it on the load path.

The wider down pitch costs storage on every layer, not just the two: a q2_k_ud slot
is 10,878,976 B instead of 9,633,792 (+12.9%). It costs far less than that in decode
PCIe -- see :func:`q2k_ud_layer_copy_bytes`.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from freetoken.models.gguf.dequant import (
    GGML_IQ2_XS,
    GGML_IQ3_XXS,
    GGML_MXFP4,
    GGML_NAME,
    row_bytes,
)
from freetoken.utils import init_logger

from .args import DeepseekV4Args
from .parallel import tp_size

logger = init_logger(__name__)

# ggml_type id of Q2_K. Not in models.gguf.dequant's table (nothing there dequantizes
# it -- the CUDA kernels do), but the re-encoded down rows must report it to the GEMV.
GGML_Q2_K = 10
_QK_K = 256  # k-quant super-block
_Q2_K_BYTES = 84  # 16 packed scale/min nibbles + 64 qs bytes + fp16 d + fp16 dmin

# Bump whenever the BYTES this loader writes into a bank change for an unchanged GGUF:
# a re-encoder numerics change (the ALS fit, the scale/min quantization, the qs bit
# layout), a pitch change, a type-table change. checkpoint/convert.py folds this into an
# FTW checkpoint's source fingerprint, which is what distinguishes "same GGUF, nothing
# changed" from "same GGUF, this loader now writes different bytes" -- otherwise the two
# are byte-identical-looking on disk.
#
# It does NOT make an older FTW refuse to load, and it should not: an FTW carries its own
# bank shapes and its own per-layer ggml types, so a v1 checkpoint still decodes exactly
# as it was written. It is simply the OLD quality. The signal for that is a warning at
# load (see moe.expert_banks._warn_stale_q2k_ud_banks), not a refusal to serve a
# checkpoint that is internally consistent.
#
# 2: the down bank went from a 784 B IQ3_XXS pitch with layers 26/42 dequantized and
#    re-encoded to Q2_K (0.279 relative L2 against the reference weights), to a 1088 B
#    MXFP4 pitch with those layers carried natively (0.000).
Q2K_REENCODE_VERSION = 2

# One expert's worth of down rows per MXFP4 -> Q2_K chunk (H=4096 rows -> ~33 MB fp32).
_REENCODE_EXPERTS_PER_RMS_SAMPLE = 64

# GGUF tensor suffix -> (bank name, "which half of gate_up")
_GATE = "ffn_gate_exps.weight"
_UP = "ffn_up_exps.weight"
_DOWN = "ffn_down_exps.weight"


# --------------------------------------------------------------------------------------
# Q2_K re-encode (RTN). gguf-py ships a Q2_K *dequantizer* but no quantizer
# (``gguf.quants.quantize(..., Q2_K)`` raises NotImplementedError), so this is the
# minimal round-trip-validated encoder that was written for the MXFP4 down rows.
#
# NOTHING ON THE LOAD PATH CALLS IT ANY MORE. The kernels gained native MXFP4 and the
# down bank widened to fit it, so those rows are a byte copy like every other row. It
# stays because it is the only Q2_K encoder in the tree and its round trip is worth
# keeping tested -- and because it is the measurement of what the old path cost
# (~0.28 relative RMS on layers 26 and 42) that a regression here would silently undo.
# --------------------------------------------------------------------------------------


_ALS_ITERS = 3  # alternating least-squares refinements of the per-sub-block (step, offset)


def _fit_step_offset(xb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-16-element sub-block ``(step, offset)`` for the model ``x ~= step * q - offset``.

    Seeded with the plain min/max fit (``step = (max - min) / 3``, ``offset = -min``), then
    refined by alternating least squares: freeze the integer codes, re-solve the 2x2 normal
    equations for ``(step, offset)``, re-quantize, repeat. This is the cheap stand-in for
    llama.cpp's ``make_qkx2_quants`` scale sweep and buys ~17% lower reconstruction RMS than
    the seed alone -- worth it because the seed's min/max anchoring wastes both outer levels
    on the two extreme samples of each sub-block.
    """
    hi = xb.max(axis=2)
    lo = np.minimum(xb.min(axis=2), 0.0)  # offset = -lo must stay >= 0
    step = (hi - lo) / 3.0
    off = -lo
    for _ in range(_ALS_ITERS):
        q = _codes(xb, step, off)
        sq = q.sum(axis=2)
        sq2 = (q * q).sum(axis=2)
        sx = xb.sum(axis=2)
        sqx = (q * xb).sum(axis=2)
        det = 16.0 * sq2 - sq * sq  # singular iff every code in the sub-block is equal
        ok = det > 1e-12
        safe = np.where(ok, det, 1.0)
        step = np.maximum(np.where(ok, (16.0 * sqx - sq * sx) / safe, step), 0.0)
        off = np.maximum(np.where(ok, -(sq2 * sx - sq * sqx) / safe, off), 0.0)
    return step, off


def _codes(xb: np.ndarray, step: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Nearest 2-bit codes for ``x ~= step * q - offset`` (zero where the step collapsed)."""
    inv = np.where(step > 0, 1.0 / np.where(step > 0, step, 1.0), 0.0)
    return np.clip(np.rint((xb + off[:, :, None]) * inv[:, :, None]), 0, 3)


def quantize_q2_k(x: np.ndarray) -> np.ndarray:
    """Encode ``[n, 256]`` float32 to ``[n, 84]`` uint8 ``block_q2_K`` records.

    ggml's ``block_q2_K`` reconstructs element ``e`` of sub-block ``j = e // 16`` as
    ``d * (scales[j] & 0xF) * q - dmin * (scales[j] >> 4)`` with ``q`` a 2-bit code. So each
    16-element sub-block owns a non-negative step and offset (:func:`_fit_step_offset`),
    both quantized to 4 bits against the super-block's fp16 ``d`` / ``dmin``; the codes are
    then recomputed against those *quantized* step/offset so the 4-bit rounding is corrected
    for rather than compounded.

    The 64 ``qs`` bytes are NOT sub-block-major: byte ``g * 32 + b`` holds elements
    ``g * 128 + s * 32 + b`` for ``s in 0..3`` at bit offset ``2 * s`` (mirrors
    gguf-py's ``Q2_K.dequantize_blocks``).
    """
    assert x.ndim == 2 and x.shape[1] == _QK_K, x.shape
    n = x.shape[0]
    xb = x.reshape(n, 16, 16).astype(np.float32)
    step, off = _fit_step_offset(xb)

    d = (step.max(axis=1, keepdims=True) / 15.0).astype(np.float16)
    dmin = (off.max(axis=1, keepdims=True) / 15.0).astype(np.float16)
    df = d.astype(np.float32)
    dminf = dmin.astype(np.float32)

    sc = np.where(df > 0, np.rint(step / np.where(df > 0, df, 1.0)), 0.0)
    mq = np.where(dminf > 0, np.rint(off / np.where(dminf > 0, dminf, 1.0)), 0.0)
    sc = np.clip(sc, 0, 15).astype(np.uint8)
    mq = np.clip(mq, 0, 15).astype(np.uint8)
    scales = (sc | (mq << 4)).astype(np.uint8)  # [n, 16]

    q = _codes(xb, df * sc.astype(np.float32), dminf * mq.astype(np.float32))
    q = q.astype(np.uint8).reshape(n, 2, 4, 32)

    qs = np.zeros((n, 2, 32), dtype=np.uint8)
    for s in range(4):
        qs |= q[:, :, s, :] << np.uint8(2 * s)

    return np.concatenate(
        [scales, qs.reshape(n, 64), d.view(np.uint8), dmin.view(np.uint8)], axis=1
    )


def _dequantize(blocks: np.ndarray, ggml_type: int) -> np.ndarray:
    """gguf-py dequant of packed rows ``[n, row_bytes]`` -> ``[n, numel]`` float32."""
    import gguf.quants
    from gguf import GGMLQuantizationType

    return gguf.quants.dequantize(
        np.ascontiguousarray(blocks), GGMLQuantizationType(ggml_type)
    ).astype(np.float32)


def _mxfp4_rows_to_q2_k(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``[n, 1088]`` MXFP4 rows -> ``([n, 672]`` Q2_K rows, the dequantized source)."""
    ref = _dequantize(packed, GGML_MXFP4)  # [n, I]
    n, numel = ref.shape
    q = quantize_q2_k(ref.reshape(-1, _QK_K))
    return q.reshape(n, numel // _QK_K * _Q2_K_BYTES), ref


def _rel_rms(ref: np.ndarray, got: np.ndarray) -> float:
    denom = float(np.sqrt((ref.astype(np.float64) ** 2).mean()))
    if denom == 0.0:
        return 0.0
    return float(np.sqrt(((got.astype(np.float64) - ref) ** 2).mean()) / denom)


# --------------------------------------------------------------------------------------
# Bank geometry + metadata validation
# --------------------------------------------------------------------------------------


def q2k_ud_expert_specs(args: DeepseekV4Args) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """The two ``q2_k_ud`` bank specs.

    Each bank's row pitch is the widest native row any of its layers holds: IQ3_XXS
    for gate_up (1568 B at H=4096), MXFP4 for down (1088 B at I=2048). Narrower layers
    occupy a prefix of the row; see the module docstring.
    """
    E, H, I = args.n_routed_experts, args.dim, args.moe_inter_dim
    return {
        "gate_up": ((E, 2 * I, row_bytes(H, GGML_IQ3_XXS)), torch.uint8),
        "down": ((E, H, row_bytes(I, GGML_MXFP4)), torch.uint8),
    }


# Which banks are worth copying at their native width on a DECODE miss (see
# OffloadMoeCache.set_layer_copy_bytes). Not a correctness knob -- both settings deliver
# the same payload -- but on that path fewer bytes is not automatically less time. A
# decode miss is a zero-copy pull KERNEL reading pinned host memory, and skipping the
# padding tail breaks each expert's read into one run per weight row; whether the shorter
# runs pay for the bytes they save depends on the geometry. Measured on this box (PCIe 5.0
# under WSL2 GPU-PV, 128 experts x 1024 rows, reproducible across trials;
# tests/kernels/test_pitched_index_copy.py asserts both):
#
#   gate_up  1568 B pitch -> 1184 B IQ2_XS payload:   76% of the bytes,  78% of the time
#   down     1088 B pitch ->  784 B IQ3_XXS payload:  72% of the bytes, 149% of the time
#
# So gate_up narrows on decode and down does not. The PREFILL layer fill narrows BOTH --
# it is a DMA transfer, and the copy engine's 2D mode takes the same skip at ~93% of its
# linear bandwidth, so there the bytes translate straight into time. The cache keeps the
# two tables apart; this constant only names the decode set.
#
# The follow-up that would win the down bank on decode too is to store a narrow layer's
# rows COMPACTED inside its expert slice and give the GEMV an expert stride separate from
# its row pitch -- the copy is contiguous on both sides then and the efficiency question
# disappears -- but that also moves the prefill GEMM's slicing, so it is not folded in
# here.
DECODE_NARROW_BANKS = ("gate_up",)


def q2k_ud_layer_copy_bytes(
    bank_sources: dict[str, list[torch.Tensor]],
    quant_types: dict[str, list[int]],
) -> dict[str, list[int]] | None:
    """Per-(bank, layer) native row width for :meth:`OffloadMoeCache.set_layer_copy_bytes`.

    Derived from the bank SHAPES rather than from ``DeepseekV4Args``, so it works
    identically for a cold GGUF load and for an FTW checkpoint replay -- both hand the
    cache the same ``[E, rows, pitch]`` tensors plus the per-layer type table, and
    ``ncols`` is recoverable from the shape (the down bank has one row per hidden unit,
    gate_up has two per intermediate unit).

    Reports the honest native width for EVERY bank; which of them a given copy path
    actually narrows is the cache's business (``decode_narrow``, from
    :data:`DECODE_NARROW_BANKS`), because the answer differs between the decode pull
    kernel and the prefill DMA.

    Returns ``None`` if any layer's type is not one the kernels can size, which puts
    every bank back on full-pitch copies -- the pre-existing behaviour.
    """
    from freetoken.kernel.gguf import ggml_type_row_bytes

    ncols = {
        "gate_up": bank_sources["down"][0].shape[1],       # H
        "down": bank_sources["gate_up"][0].shape[1] // 2,  # I
    }
    out: dict[str, list[int]] = {}
    for name, types in quant_types.items():
        pitch = int(bank_sources[name][0].shape[-1])
        widths = []
        for layer, qtype in enumerate(types):
            try:
                native = int(ggml_type_row_bytes(int(qtype), int(ncols[name])))
            except (ValueError, RuntimeError, ImportError):
                native = 0
            if not 0 < native <= pitch or native % 16 != 0:
                # Unknown type, a row wider than the bank (impossible -- the loader
                # rejects it), or a width the 16 B copy unit cannot express. Any of
                # those and the whole declaration is dropped rather than silently
                # applied to some layers: a partial table is harder to reason about
                # than none.
                logger.warning_rank0(
                    f"q2_k_ud copy widths disabled: bank {name!r} layer {layer} has "
                    f"ggml type {qtype} with native row {native} B against pitch {pitch} B"
                )
                return None
            widths.append(native)
        out[name] = widths
    return out


def _metadata_source(gguf_path: str) -> str:
    """A single ``.gguf`` the KV reader can parse (a split set's first shard)."""
    from freetoken.models.gguf.reader import _split_shard_paths

    shards = _split_shard_paths(gguf_path)
    return shards[0] if shards else gguf_path


def _validate_metadata(gguf_path: str, args: DeepseekV4Args) -> None:
    """Fail fast when the GGUF's ``deepseek4.*`` geometry disagrees with ``args``."""
    from freetoken.models.gguf.reader import load_gguf_metadata

    md = load_gguf_metadata(_metadata_source(gguf_path))
    arch = md.get("general.architecture")
    if arch != "deepseek4":
        raise ValueError(
            f"--expert-gguf {gguf_path}: general.architecture is {arch!r}, expected "
            "'deepseek4' (the routed-expert banks are DeepSeek-V4 specific)"
        )
    expect = {
        "deepseek4.embedding_length": (args.dim, "dim"),
        "deepseek4.expert_feed_forward_length": (args.moe_inter_dim, "moe_inter_dim"),
        "deepseek4.block_count": (args.n_layers, "n_layers"),
        "deepseek4.expert_count": (args.n_routed_experts, "n_routed_experts"),
        "deepseek4.expert_used_count": (args.n_activated_experts, "n_activated_experts"),
    }
    bad = [
        f"{key}={md.get(key)!r} but args.{attr}={want}"
        for key, (want, attr) in expect.items()
        if md.get(key) is None or int(md[key]) != int(want)
    ]
    if bad:
        raise ValueError(
            f"--expert-gguf {gguf_path} does not match this checkpoint's DeepSeek-V4 "
            "config: " + "; ".join(bad)
        )


# --------------------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------------------


def _expert_layer(name: str) -> tuple[int, str] | None:
    """``(layer, suffix)`` for a routed-expert GGUF tensor, else ``None``."""
    if not name.startswith("blk."):
        return None
    for suffix in (_GATE, _UP, _DOWN):
        if name.endswith(suffix):
            return int(name.split(".")[1]), suffix
    return None


def load_q2k_ud_expert_sources(
    gguf_path: str,
    args: DeepseekV4Args,
    *,
    dummy: bool = False,
    layer_sink=None,
    _layers: set[int] | None = None,
) -> tuple[dict[str, list[torch.Tensor]], dict[str, list[int]]]:
    """Per-layer host banks of the GGUF's routed experts, plus their quant-type table.

    Returns ``(banks, quant_types)``:

    * ``banks`` matches :func:`~freetoken.models.deepseek_v4.weight.load_dsfp4_expert_sources`'s
      contract -- ``{bank name: one [E, rows, pitch] uint8 tensor per layer}``, allocated by
      :func:`~freetoken.moe.host_banks.alloc_layer_banks`, pinned per completed layer by an
      internally owned :class:`~freetoken.moe.host_banks.PinPipeline` when ``layer_sink`` is
      ``None`` (serving) or handed to ``layer_sink`` instead (converter).
    * ``quant_types`` is ``{"gate_up": [ggml type per layer], "down": [...]}`` -- what the
      GEMV must decode each layer's rows as. Layers whose down rows were re-encoded report
      :data:`GGML_Q2_K`; everything else reports the type the GGUF stored.

    ``gate_up`` is gate-major: rows ``[0, I)`` are ``ffn_gate_exps``, rows ``[I, 2I)`` are
    ``ffn_up_exps`` -- the same order the ds_fp4 loader uses for ``w1``/``w3``, which is what
    ``fused_swiglu`` splits on.

    Row tails past the native packed width are left untouched: ``HostBank`` guarantees
    unwritten bytes read as zero (lazy anonymous mmap, or an explicitly zeroed cudaHostAlloc).

    ``_layers`` restricts the load to a subset of layer ids -- a test hook. Every layer's
    bank is still allocated (a lazy mmap commits no pages), but unselected layers are left
    zero and never complete, so they are neither pinned nor forwarded to ``layer_sink``.
    """
    if dummy:
        return dummy_q2k_ud_expert_sources(args)

    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    if tp_size() > 1:
        raise NotImplementedError(
            "q2_k_ud expert banks are TP=1 only: the packed GGUF rows span the full "
            "hidden/intermediate dim and cannot be column-sliced per rank"
        )
    _validate_metadata(gguf_path, args)
    if args.n_draft_layers:
        raise ValueError(
            "--expert-gguf carries no mtp.* drafter experts; --speculative-dspark cannot "
            "be combined with GGUF expert banks"
        )

    L, E = args.n_moe_layers, args.n_routed_experts
    H, I = args.dim, args.moe_inter_dim
    specs = q2k_ud_expert_specs(args)
    gu_pitch = specs["gate_up"][0][2]
    dn_pitch = specs["down"][0][2]
    hb = alloc_layer_banks(specs, L)
    banks = {name: [b.tensor for b in hb[name]] for name in specs}

    want = set(range(L)) if _layers is None else {int(x) for x in _layers}
    # Recorded as the GGUF's own type first; the MXFP4 down layers are rewritten to Q2_K.
    qtypes: dict[str, list[int]] = {"gate_up": [0] * L, "down": [0] * L}
    seen: dict[str, set[int]] = {_GATE: set(), _UP: set(), _DOWN: set()}

    def _place_gate_up(layer: int, t, lo: int) -> None:
        if t.row_bytes > gu_pitch:
            raise ValueError(
                f"blk.{layer}: {GGML_NAME.get(t.ggml_type, t.ggml_type)} gate/up row is "
                f"{t.row_bytes} B, wider than the q2_k_ud gate_up pitch {gu_pitch} B"
            )
        prev = qtypes["gate_up"][layer]
        if prev and prev != t.ggml_type:
            raise ValueError(
                f"blk.{layer}: ffn_gate_exps and ffn_up_exps disagree on ggml type "
                f"({GGML_NAME.get(prev, prev)} vs {GGML_NAME.get(t.ggml_type, t.ggml_type)}); "
                "one gate_up bank decodes with a single type per layer"
            )
        qtypes["gate_up"][layer] = t.ggml_type
        src = t.packed().reshape(E, I, t.row_bytes)
        banks["gate_up"][layer][:, lo:lo + I, :t.row_bytes] = src

    def _place_down(layer: int, t) -> None:
        if t.row_bytes > dn_pitch:
            raise ValueError(
                f"blk.{layer}: {GGML_NAME.get(t.ggml_type, t.ggml_type)} down row is "
                f"{t.row_bytes} B, wider than the q2_k_ud down pitch {dn_pitch} B"
            )
        qtypes["down"][layer] = t.ggml_type
        banks["down"][layer][:, :, :t.row_bytes] = t.packed().reshape(E, H, t.row_bytes)

    def _load(sink) -> None:
        # gate + up + down = 3 writes per layer.
        tracker = LayerCompletionTracker(3, hb, sink) if sink is not None else None
        pbar = tqdm(total=len(want) * 3, desc="Loading DSV4 GGUF experts")
        for t in iter_gguf_tensors(gguf_path):
            key = _expert_layer(t.name)
            if key is None:
                continue
            layer, suffix = key
            if layer not in want:
                continue
            if layer >= L:
                raise ValueError(f"{t.name}: layer {layer} beyond n_moe_layers {L}")
            if suffix == _GATE:
                _place_gate_up(layer, t, 0)
            elif suffix == _UP:
                _place_gate_up(layer, t, I)
            else:
                _place_down(layer, t)
            seen[suffix].add(layer)
            pbar.update(1)
            if tracker is not None:
                tracker.note(layer)
        pbar.close()

    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)  # CUDA-less: mmap banks stay pageable, never pinned

    missing = {k: sorted(want - v) for k, v in seen.items() if want - v}
    if missing:
        raise ValueError(f"--expert-gguf {gguf_path}: missing routed-expert tensors {missing}")
    for name in ("gate_up", "down"):
        tally: dict[int, list[int]] = {}
        for layer in sorted(want):
            tally.setdefault(qtypes[name][layer], []).append(layer)
        parts = ", ".join(
            f"{GGML_NAME.get(t, t)} x{len(ls)}"
            + (f" (layers {ls})" if len(ls) <= 4 else "")
            for t, ls in sorted(tally.items())
        )
        logger.info_rank0(f"q2_k_ud {name} bank: {parts}")
    return banks, qtypes


def dummy_q2k_ud_expert_sources(
    args: DeepseekV4Args,
) -> tuple[dict[str, list[torch.Tensor]], dict[str, list[int]]]:
    """Random q2_k_ud banks for ``--dummy-weight`` (no GGUF on disk).

    Types report the checkpoint's majority pair (IQ2_XS gate_up / IQ3_XXS down); random
    bytes are a valid bit pattern for both, so the GEMV path is exercised unchanged.
    """
    from freetoken.moe.host_banks import alloc_layer_banks, pin_banks

    L = args.n_moe_layers
    hb = alloc_layer_banks(q2k_ud_expert_specs(args), L)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    for t in banks["gate_up"] + banks["down"]:
        t.random_(0, 256)
    if torch.cuda.is_available():
        pin_banks(hb)
    qtypes = {"gate_up": [GGML_IQ2_XS] * L, "down": [GGML_IQ3_XXS] * L}
    return banks, qtypes


__all__ = [
    "GGML_Q2_K",
    "Q2K_REENCODE_VERSION",
    "quantize_q2_k",
    "q2k_ud_expert_specs",
    "q2k_ud_layer_copy_bytes",
    "DECODE_NARROW_BANKS",
    "load_q2k_ud_expert_sources",
    "dummy_q2k_ud_expert_sources",
]
