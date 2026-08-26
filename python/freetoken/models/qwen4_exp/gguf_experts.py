"""Native GGUF K-quant routed-expert banks for Qwen4-Exp (Qwen3.8-Flash-Next),
unsloth UD-Q4_K_XL.

The ``q4_k_ud`` sibling of :mod:`freetoken.models.deepseek_v4.gguf_experts`'s
``q2_k_ud``, generalized for this model's expert geometry (H=2560, I=640, 512
experts, 48 layers -- every decoder layer carries routed experts). This GGUF
mixes ggml types PER LAYER on both banks:

* ``blk.N.ffn_{gate,up}_exps.weight``: Q4_K (12) on 47 layers, Q5_K (13) on
  layer 2. Native row = 1440 B (Q4_K) / 1760 B (Q5_K) at H=2560.
* ``blk.N.ffn_down_exps.weight``: Q5_1 (7) on 43 layers, Q8_0 (8) on layers
  2, 4, 30, 46, 47. Native row = 480 B (Q5_1) / 680 B (Q8_0) at I=640.

One bank cannot mix row strides (see the ``q2_k_ud`` module docstring for why),
so each bank here pads every row to its WIDEST layer's native width, further
rounded up to a 16 B multiple (:func:`_align16` -- required for
``moe_vec_resolve_pitch``'s row-alignment guard; Q8_0's native 680 B is NOT a
16 B multiple, unlike every q2_k_ud pitch) -- 1760 B for gate_up (Q5_K, already
aligned), 688 B for down (Q8_0, padded from 680) -- and ``ggml_moe_a8_vec``'s
``row_pitch_bytes`` addresses those padded rows while decoding each row at its
own (per-layer) ggml type, via the same per-layer quant-type side table
``q2_k_ud`` uses (see :func:`load_q4k_ud_expert_sources`'s return value).

Unlike ``q2_k_ud``, NOTHING here is dequantized and re-encoded: Q4_K, Q5_K,
Q5_1 and Q8_0 all already have native ``ggml_moe_a8_vec`` MMVQ kernels and
``ggml_dequantize`` support (see ``kernel/csrc/gguf/gguf_kernel.cu``'s
``ggml_moe_a8_vec`` switch and ``dequantize.cuh``'s ``ggml_get_to_cuda``), so
every row -- at both pitches -- is a byte copy of the GGUF's own bytes.

Padded bank footprint: ``(2*640*1760 + 2560*688) * 512 experts * 48 layers``
= 98,650,030,080 B ~= 91.9 GiB host, vs ~112.5 GiB for the fp8 safetensors
bank it replaces. The +28.6% padding overhead (over the narrowest per-layer
packing) is the same unavoidable trade DSV4's ``q2_k_ud`` accepted for the
same reason -- see :func:`freetoken.models.deepseek_v4.gguf_experts.q2k_ud_layer_copy_bytes`'s
docstring for the PCIe-decode-cost side of that trade (this format only
narrows the ``gate_up`` bank on decode; see :data:`DECODE_NARROW_BANKS`).
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from freetoken.models.gguf.dequant import GGML_NAME, GGML_Q4_K, GGML_Q5_1, GGML_Q5_K, GGML_Q8_0, row_bytes
from freetoken.utils import init_logger

logger = init_logger(__name__)

# GGUF tensor suffix -> which half of gate_up (mirrors deepseek_v4.gguf_experts).
_GATE = "ffn_gate_exps.weight"
_UP = "ffn_up_exps.weight"
_DOWN = "ffn_down_exps.weight"


def _align16(n: int) -> int:
    """Round a byte width up to the next 16 B multiple.

    ``moe_vec_resolve_pitch`` (kernel/csrc/gguf/moe_vec.cuh) hard-requires
    ``row_pitch_bytes % 16 == 0`` so every expert row starts 16 B-aligned --
    q2_k_ud's two pitches (1568, 1088) happen to already be 16 B multiples, but
    that is a property of THOSE block sizes, not a general guarantee. Q8_0's
    native row at I=640 (20 blocks x 34 B = 680 B) is NOT: 680 % 16 == 8. Round
    up rather than assert, same as any other pitch-vs-native slack in this
    format -- the extra bytes are dead padding like any other, and the row
    stays >= every native width it must hold (688 >= 680).
    """
    return (n + 15) // 16 * 16


# --------------------------------------------------------------------------------------
# Bank geometry + metadata validation
# --------------------------------------------------------------------------------------


def q4k_ud_expert_specs(model_config) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """The two ``q4_k_ud`` bank specs.

    Each bank's row pitch is the widest native row any of its layers holds,
    rounded up to a 16 B multiple (see :func:`_align16`): Q5_K for gate_up
    (1760 B at H=2560, already 16 B-aligned), Q8_0 for down (680 B native at
    I=640, padded to 688 B). Narrower layers (Q4_K gate_up, Q5_1 down) occupy a
    prefix of the row; see the module docstring.
    """
    E = model_config.num_experts
    H = model_config.hidden_size
    I = model_config.moe_intermediate_size
    return {
        "gate_up": ((E, 2 * I, _align16(row_bytes(H, GGML_Q5_K))), torch.uint8),
        "down": ((E, H, _align16(row_bytes(I, GGML_Q8_0))), torch.uint8),
    }


# Which banks are worth copying at their native width on a DECODE miss (see
# OffloadMoeCache.set_layer_copy_bytes). Start conservative: only gate_up is
# declared narrow. DSV4's q2_k_ud measured that narrowing its ANALOGOUS down
# bank (the one whose widest layers are the rarer, wider type -- there MXFP4,
# here Q8_0) cost 149% of the time despite moving 72% of the bytes -- a
# geometry-dependent result, not a rule, and this format's down bank has NOT
# been measured. gate_up narrows (47 of 48 layers are the narrower Q4_K, same
# shape of win q2_k_ud measured for its own gate_up bank); down stays at full
# pitch until it is actually benchmarked on this box.
DECODE_NARROW_BANKS = ("gate_up",)


def q4k_ud_layer_copy_bytes(
    bank_sources: dict[str, list[torch.Tensor]],
    quant_types: dict[str, list[int]],
) -> dict[str, list[int]] | None:
    """Per-(bank, layer) native row width for :meth:`OffloadMoeCache.set_layer_copy_bytes`.

    Identical logic to :func:`freetoken.models.deepseek_v4.gguf_experts.q2k_ud_layer_copy_bytes`
    -- derived from the bank SHAPES so it works identically for a cold GGUF load
    and an FTW checkpoint replay. Returns ``None`` if any layer's type is not one
    the kernels can size (native width 0, wider than the pitch, or not a 16 B
    multiple -- the copy unit's own alignment requirement, same guard the
    q2_k_ud version applies), which puts every bank back on full-pitch copies.

    For THIS checkpoint's GGUF that guard always trips: Q8_0's native row at
    I=640 is 680 B, and 680 % 16 == 8, so every layer 2/4/30/46/47 (the down
    bank's Q8_0 layers) fails it and the whole declaration -- BOTH banks, not
    just down -- returns ``None``. gate_up's own native widths (Q4_K 1440 B,
    Q5_K 1760 B) are themselves 16 B-aligned, so :data:`DECODE_NARROW_BANKS`
    would narrow it if this returned a real table; on this GGUF it never gets
    the chance. Harmless -- the fallback (full-pitch decode copies) is exactly
    the correctness-preserving default -- but it means the ~24% PCIe saving
    q2_k_ud's gate_up narrowing measures does NOT apply here without also
    widening the down bank's native-width slice to a 16 B multiple (not
    attempted: unmeasured, and the down bank is not even narrowed by
    :data:`DECODE_NARROW_BANKS`, so it costs nothing to leave alone).
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
                logger.warning_rank0(
                    f"q4_k_ud copy widths disabled: bank {name!r} layer {layer} has "
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


def _validate_metadata(gguf_path: str, model_config) -> None:
    """Fail fast when the GGUF's ``qwen4exp.*`` geometry disagrees with ``model_config``.

    Same idea as deepseek_v4.gguf_experts._validate_metadata: this is what keeps a
    DSV4 GGUF from silently loading against a qwen4_exp checkpoint (or vice versa)
    -- ``general.architecture`` and the block-count/expert-count/dim KVs must match.
    """
    from freetoken.models.gguf.reader import load_gguf_metadata

    md = load_gguf_metadata(_metadata_source(gguf_path))
    arch = md.get("general.architecture")
    if arch != "qwen4exp":
        raise ValueError(
            f"--expert-gguf {gguf_path}: general.architecture is {arch!r}, expected "
            "'qwen4exp' (the routed-expert banks are Qwen4-Exp / Qwen3.8-Flash-Next specific)"
        )
    expect = {
        "qwen4exp.embedding_length": (model_config.hidden_size, "hidden_size"),
        "qwen4exp.expert_feed_forward_length": (model_config.moe_intermediate_size, "moe_intermediate_size"),
        "qwen4exp.block_count": (model_config.num_layers, "num_layers"),
        "qwen4exp.expert_count": (model_config.num_experts, "num_experts"),
        "qwen4exp.expert_used_count": (model_config.num_experts_per_tok, "num_experts_per_tok"),
    }
    bad = [
        f"{key}={md.get(key)!r} but model_config.{attr}={want}"
        for key, (want, attr) in expect.items()
        if md.get(key) is None or int(md[key]) != int(want)
    ]
    if bad:
        raise ValueError(
            f"--expert-gguf {gguf_path} does not match this checkpoint's Qwen4-Exp "
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


def load_q4k_ud_expert_sources(
    gguf_path: str,
    model_config,
    *,
    dummy: bool = False,
    layer_sink=None,
    _layers: set[int] | None = None,
) -> tuple[dict[str, list[torch.Tensor]], dict[str, list[int]]]:
    """Per-layer host banks of the GGUF's routed experts, plus their quant-type table.

    Returns ``(banks, quant_types)``:

    * ``banks`` matches the contract every offload-cache provider uses --
      ``{bank name: one [E, rows, pitch] uint8 tensor per layer}``, allocated by
      :func:`~freetoken.moe.host_banks.alloc_layer_banks`, pinned per completed
      layer by an internally owned :class:`~freetoken.moe.host_banks.PinPipeline`
      when ``layer_sink`` is ``None`` (serving) or handed to ``layer_sink``
      instead (converter).
    * ``quant_types`` is ``{"gate_up": [ggml type per layer], "down": [...]}`` --
      what the GEMV must decode each layer's rows as.

    ``gate_up`` is gate-major: rows ``[0, I)`` are ``ffn_gate_exps``, rows
    ``[I, 2I)`` are ``ffn_up_exps`` -- what ``silu_and_mul`` splits on.

    Row tails past the native packed width are left untouched: ``HostBank``
    guarantees unwritten bytes read as zero.

    ``_layers`` restricts the load to a subset of layer ids -- a test hook.
    """
    if dummy:
        return dummy_q4k_ud_expert_sources(model_config)

    from freetoken.distributed import get_tp_info
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    if get_tp_info().size > 1:
        raise NotImplementedError(
            "q4_k_ud expert banks are TP=1 only: the packed GGUF rows span the full "
            "hidden/intermediate dim and cannot be column-sliced per rank"
        )
    _validate_metadata(gguf_path, model_config)

    L = model_config.num_moe_layers  # every qwen4_exp decoder layer carries routed experts
    E = model_config.num_experts
    I = model_config.moe_intermediate_size
    specs = q4k_ud_expert_specs(model_config)
    gu_pitch = specs["gate_up"][0][2]
    dn_pitch = specs["down"][0][2]
    hb = alloc_layer_banks(specs, L)
    banks = {name: [b.tensor for b in hb[name]] for name in specs}

    want = set(range(L)) if _layers is None else {int(x) for x in _layers}
    qtypes: dict[str, list[int]] = {"gate_up": [0] * L, "down": [0] * L}
    seen: dict[str, set[int]] = {_GATE: set(), _UP: set(), _DOWN: set()}

    def _place_gate_up(layer: int, t, lo: int) -> None:
        if t.row_bytes > gu_pitch:
            raise ValueError(
                f"blk.{layer}: {GGML_NAME.get(t.ggml_type, t.ggml_type)} gate/up row is "
                f"{t.row_bytes} B, wider than the q4_k_ud gate_up pitch {gu_pitch} B"
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
                f"{t.row_bytes} B, wider than the q4_k_ud down pitch {dn_pitch} B"
            )
        qtypes["down"][layer] = t.ggml_type
        banks["down"][layer][:, :, :t.row_bytes] = t.packed().reshape(
            E, banks["down"][layer].shape[1], t.row_bytes
        )

    def _load(sink) -> None:
        # gate + up + down = 3 writes per layer.
        tracker = LayerCompletionTracker(3, hb, sink) if sink is not None else None
        pbar = tqdm(total=len(want) * 3, desc="Loading Qwen4-Exp GGUF experts")
        for t in iter_gguf_tensors(gguf_path):
            key = _expert_layer(t.name)
            if key is None:
                continue
            layer, suffix = key
            if layer not in want:
                continue
            if layer >= L:
                raise ValueError(f"{t.name}: layer {layer} beyond num_moe_layers {L}")
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
            + (f" (layers {ls})" if len(ls) <= 6 else "")
            for t, ls in sorted(tally.items())
        )
        logger.info_rank0(f"q4_k_ud {name} bank: {parts}")
    return banks, qtypes


def dummy_q4k_ud_expert_sources(
    model_config,
) -> tuple[dict[str, list[torch.Tensor]], dict[str, list[int]]]:
    """Random q4_k_ud banks for ``--dummy-weight`` (no GGUF on disk).

    Types report the checkpoint's majority pair (Q4_K gate_up / Q5_1 down);
    random bytes are a valid bit pattern for both, so the GEMV path is
    exercised unchanged.
    """
    from freetoken.moe.host_banks import alloc_layer_banks, pin_banks

    L = model_config.num_moe_layers
    hb = alloc_layer_banks(q4k_ud_expert_specs(model_config), L)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    for t in banks["gate_up"] + banks["down"]:
        t.random_(0, 256)
    if torch.cuda.is_available():
        pin_banks(hb)
    qtypes = {"gate_up": [GGML_Q4_K] * L, "down": [GGML_Q5_1] * L}
    return banks, qtypes


__all__ = [
    "q4k_ud_expert_specs",
    "q4k_ud_layer_copy_bytes",
    "DECODE_NARROW_BANKS",
    "load_q4k_ud_expert_sources",
    "dummy_q4k_ud_expert_sources",
]
