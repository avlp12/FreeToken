"""The q2_k_ud expert-bank geometry, and what carrying MXFP4 natively bought.

unsloth's UD-Q2_K_XL stores the DeepSeek-V4-Flash ``ffn_down_exps`` of layers 26 and
42 in MXFP4 -- 4.25 bpw against IQ3_XXS's 3.06 -- which is a deliberate statement
about which layers are sensitive. Those rows are 1088 B wide and did not fit the
784 B IQ3_XXS pitch the down bank used, so the loader dequantized them and
re-encoded them to Q2_K RTN (672 B) to squeeze them in: the precision was spent on
those layers and then thrown away at load.

The bank now pitches to 1088 and the rows are a byte copy. These tests pin the three
things that has to mean:

* the pitch really is the MXFP4 row width, and the slot arithmetic that follows from
  it is what the VRAM budget was computed against;
* the Q2_K encoder still round-trips (it is no longer on the load path, so nothing
  else would notice it rotting) -- and, on the same data, native MXFP4 beats it by
  the margin that justified this whole change;
* the per-layer copy widths handed to the offload cache are the native row bytes,
  and only for the bank where narrowing pays.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from freetoken.models.deepseek_v4.gguf_experts import (
    GGML_Q2_K,
    Q2K_REENCODE_VERSION,
    _mxfp4_rows_to_q2_k,
    _rel_rms,
    q2k_ud_layer_copy_bytes,
    quantize_q2_k,
)
from freetoken.models.gguf.dequant import GGML_IQ2_XS, GGML_IQ3_XXS, GGML_MXFP4, row_bytes

H, I = 4096, 2048  # DeepSeek-V4-Flash
GU_PITCH = 1568   # row_bytes(H, IQ3_XXS)
DN_PITCH = 1088   # row_bytes(I, MXFP4)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #


def test_bank_pitches_are_the_widest_native_row():
    assert row_bytes(H, GGML_IQ3_XXS) == GU_PITCH
    assert row_bytes(H, GGML_IQ2_XS) == 1184
    assert row_bytes(I, GGML_MXFP4) == DN_PITCH
    assert row_bytes(I, GGML_IQ3_XXS) == 784
    # Both pitches must be 16 B multiples or neither the GEMV's pitch guard nor the
    # copy kernel's 16 B unit can address them.
    assert GU_PITCH % 16 == 0 and DN_PITCH % 16 == 0


def test_slot_bytes_match_the_vram_budget():
    from freetoken.moe.offload_cache import _BANK_BYTES_PER_EXPERT

    per_expert = _BANK_BYTES_PER_EXPERT["q2_k_ud"](H, I)
    assert per_expert == 2 * I * GU_PITCH + H * DN_PITCH == 10_878_976
    # The pre-MXFP4 number, for the record: the wider down rows cost 12.9% of storage.
    assert per_expert / (2 * I * GU_PITCH + H * 784) == pytest.approx(1.1292, abs=1e-4)


def test_reencode_version_was_bumped_past_the_narrow_down_bank():
    """The converter's source fingerprint has to change, or `ft checkpoint` cannot tell
    "same GGUF, nothing changed" from "same GGUF, this loader now writes different
    bytes" -- the two are byte-identical-looking on disk."""
    assert Q2K_REENCODE_VERSION >= 2


def test_a_pre_mxfp4_ftw_is_flagged_but_not_rejected(monkeypatch):
    """A v1 FTW carries its own shapes and types, so it decodes exactly as written and
    refusing it would be wrong. It is just the old quality, and that deserves saying."""
    from types import SimpleNamespace

    from freetoken.moe import expert_banks

    said: list[str] = []
    # freetoken's loggers set propagate=False, so caplog never sees them.
    monkeypatch.setattr(
        expert_banks.logger, "warning_rank0", lambda msg, *a, **k: said.append(str(msg))
    )

    layers = 43
    down = [GGML_IQ3_XXS] * layers
    fresh = SimpleNamespace(
        quant_format="q2_k_ud",
        quant_types={"gate_up": [GGML_IQ2_XS] * layers, "down": list(down)},
    )
    expert_banks._warn_stale_q2k_ud_banks(fresh, "/models/v3")
    assert not said, "a native-MXFP4 checkpoint must not warn"

    # A format with no per-layer type table at all must not trip over it either.
    expert_banks._warn_stale_q2k_ud_banks(
        SimpleNamespace(quant_format="bf16", quant_types=None), "/models/dense"
    )
    assert not said

    down[26] = down[42] = GGML_Q2_K
    stale = SimpleNamespace(
        quant_format="q2_k_ud",
        quant_types={"gate_up": [GGML_IQ2_XS] * layers, "down": down},
    )
    expert_banks._warn_stale_q2k_ud_banks(stale, "/models/v2")
    assert len(said) == 1
    assert "[26, 42]" in said[0] and "ft checkpoint" in said[0]


# --------------------------------------------------------------------------- #
# the Q2_K encoder, and what it cost
# --------------------------------------------------------------------------- #


def _fake_mxfp4_rows(n: int, seed: int) -> np.ndarray:
    """``[n, 1088]`` of plausible MXFP4 rows: random E2M1 nibbles under a per-block
    E8M0 scale drawn from a narrow band, so the dequantized values look like weights
    rather than like a dynamic-range stress test."""
    rng = np.random.default_rng(seed)
    nb = I // 32
    blocks = np.empty((n * nb, 17), dtype=np.uint8)
    blocks[:, 0] = rng.integers(118, 127, size=n * nb, dtype=np.uint8)
    blocks[:, 1:] = rng.integers(0, 256, size=(n * nb, 16), dtype=np.uint8)
    return blocks.reshape(n, nb * 17)


def test_q2_k_encoder_round_trips():
    """Not on the load path any more, so this is the only thing keeping it honest."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((64, 256)) * 0.02).astype(np.float32)
    q = quantize_q2_k(x)
    assert q.shape == (64, 84) and q.dtype == np.uint8

    import gguf.quants
    from gguf import GGMLQuantizationType

    back = gguf.quants.dequantize(
        np.ascontiguousarray(q), GGMLQuantizationType(GGML_Q2_K)
    ).astype(np.float32)
    # 2 bits per weight with a 4-bit step/offset per 16 buys ~0.27 relative RMS on
    # gaussian input -- that IS Q2_K, and it is why sending layers 26 and 42 through
    # it was expensive. A broken bit layout or a lost ALS refinement lands well
    # outside this band in one direction or the other.
    err = _rel_rms(x, back)
    print(f"\n  Q2_K round trip on gaussian input: relative RMS {err:.4f}")
    assert 0.20 < err < 0.32, f"Q2_K round-trip relative RMS {err:.4f}"


def test_native_mxfp4_beats_the_q2_k_reencode_it_replaced():
    """The whole point of the change, measured on the same bytes.

    Native MXFP4 reconstructs its own rows EXACTLY -- it is the stored format, so the
    error is zero by construction -- while the old path's Q2_K re-encode of those rows
    carried a relative RMS around 0.28. That gap is the precision unsloth spent on
    layers 26 and 42 and the loader used to discard.
    """
    import gguf.quants
    from gguf import GGMLQuantizationType

    packed = _fake_mxfp4_rows(96, seed=1)
    q2k, ref = _mxfp4_rows_to_q2_k(packed)

    native = gguf.quants.dequantize(
        np.ascontiguousarray(packed), GGMLQuantizationType(GGML_MXFP4)
    ).astype(np.float32)
    assert np.array_equal(native, ref), "native MXFP4 is not exact on its own rows"

    reencoded = gguf.quants.dequantize(
        np.ascontiguousarray(q2k), GGMLQuantizationType(GGML_Q2_K)
    ).astype(np.float32)
    err = _rel_rms(ref, reencoded)
    print(f"\n  Q2_K re-encode of MXFP4 rows: relative RMS {err:.4f}; native: 0.0000")
    assert 0.15 < err < 0.45, (
        f"the old re-encode path's error is {err:.4f}, not the ~0.28 this change was "
        "sized against -- re-measure before trusting the quality claim"
    )


# --------------------------------------------------------------------------- #
# per-layer copy widths
# --------------------------------------------------------------------------- #


def _bank_stub(layers: int):
    """Zero-strided stand-ins: only ``.shape`` is read."""
    return {
        "gate_up": [torch.empty((0, 2 * I, GU_PITCH), dtype=torch.uint8)] * layers,
        "down": [torch.empty((0, H, DN_PITCH), dtype=torch.uint8)] * layers,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the gguf kernel module")
def test_copy_widths_are_the_native_row_bytes():
    layers = 43
    qtypes = {
        "gate_up": [GGML_IQ2_XS] * layers,
        "down": [GGML_IQ3_XXS] * layers,
    }
    qtypes["gate_up"][26] = GGML_IQ3_XXS
    qtypes["down"][26] = GGML_MXFP4
    qtypes["down"][42] = GGML_MXFP4

    got = q2k_ud_layer_copy_bytes(_bank_stub(layers), qtypes)
    assert got is not None
    # gate_up: 42 layers of IQ2_XS at 1184, one of IQ3_XXS whose native row IS the pitch.
    assert got["gate_up"][26] == GU_PITCH
    assert got["gate_up"][0] == got["gate_up"][42] == 1184
    assert sum(w == 1184 for w in got["gate_up"]) == 42
    # down: 41 layers of IQ3_XXS at 784, and the two MXFP4 layers at the full pitch.
    assert got["down"][26] == got["down"][42] == DN_PITCH
    assert got["down"][0] == 784
    assert sum(w == 784 for w in got["down"]) == 41
    assert all(w % 16 == 0 for ws in got.values() for w in ws)


def test_decode_narrows_only_the_bank_where_it_pays():
    """The table above is honest about both banks; the DECODE path deliberately uses
    only part of it, while the prefill DMA uses all of it. Both are measured facts, not
    structural ones -- tests/kernels/test_pitched_index_copy.py holds the measurements.
    """
    from freetoken.models.deepseek_v4.gguf_experts import DECODE_NARROW_BANKS

    assert tuple(DECODE_NARROW_BANKS) == ("gate_up",)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the gguf kernel module")
def test_copy_widths_bail_out_on_a_type_the_kernels_cannot_size():
    """A partial table is harder to reason about than none, so one unknown type drops
    the whole declaration and every bank goes back to full-pitch copies."""
    layers = 4
    qtypes = {"gate_up": [GGML_IQ2_XS] * layers, "down": [GGML_IQ3_XXS] * layers}
    qtypes["gate_up"][2] = 250  # not a ggml type
    assert q2k_ud_layer_copy_bytes(_bank_stub(layers), qtypes) is None
