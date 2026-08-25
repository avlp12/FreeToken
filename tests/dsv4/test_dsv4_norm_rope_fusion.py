"""Bit-identity and capture-safety for the fused DSV4 RMSNorm+RoPE decode stage.

Two fusions are covered:

  * folding the ``freqs_cis.index_select(0, pos)`` gather into the rope kernel, and
  * collapsing ``rms_norm`` + that rope into one launch.

Both are bit-identical by construction: the folded gather reads the same table
entries, and the fused stage reuses the same Triton reduction (same ``BLOCK_D``,
same ``num_warps``) and reproduces the bf16 rounding the reference gets from
storing the norm result before the rope kernel reloads it. So every assertion is
``torch.equal``.

Nothing here touches the global RNG -- tests/dsv4/test_hc_fused.py draws some of
its inputs from the ambient CUDA stream and its tolerances notice.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused norm+rope is a CUDA Triton kernel"
)

DEV = "cuda"
EPS = 1e-6
MAXSEQ = 4096


def _gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def _table(rope_dim, seed=0):
    """A dense random rotation table -- harsher than real YaRN freqs, which are
    smooth in position and would hide an off-by-one in the folded gather."""
    g = _gen(seed)
    ang = torch.rand(MAXSEQ, rope_dim // 2, device=DEV, generator=g) * 6.283185307
    return torch.polar(torch.ones_like(ang), ang)


def _reference(x, w, table, pos, rope_dim, inverse):
    from freetoken.kernel.triton.dsv4.norm import rms_norm
    from freetoken.kernel.triton.dsv4.rope import rope_decode_inplace

    y = rms_norm(x, w, EPS)
    rope_decode_inplace(y[..., -rope_dim:], table.index_select(0, pos), inverse)
    return y


def _case(shape, D, rope_dim, has_w, heads, inverse=False, dtype=torch.bfloat16, seed=0):
    g = _gen(seed)
    x = torch.randn(*shape, device=DEV, dtype=dtype, generator=g)
    w = torch.randn(D, device=DEV, dtype=dtype, generator=g) if has_w else None
    table = _table(rope_dim, seed)
    pos = torch.randint(
        0, MAXSEQ, (shape[0],), device=DEV, dtype=torch.int64, generator=g
    )
    return x, w, table, pos


# --------------------------------------------------------------------------- #
# the folded frequency gather, on the standalone rope kernel
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("shape,heads", [((1, 1, 64), 1), ((4, 1, 64), 1),
                                         ((1, 1, 16, 64), 16), ((3, 1, 8, 64), 8)])
@pytest.mark.parametrize("inverse", [False, True])
def test_folded_gather_matches_index_select(shape, heads, inverse):
    from freetoken.kernel.triton.dsv4.rope import rope_decode_inplace

    rd = 64
    x, _, table, pos = _case(shape, rd, rd, False, heads, seed=11)
    a = x.clone()
    b = x.clone()
    rope_decode_inplace(a, table.index_select(0, pos), inverse)
    rope_decode_inplace(b, table, inverse, positions=pos)
    assert torch.equal(a, b)


def test_folded_gather_actually_uses_the_position():
    """A kernel that ignored `positions` and used the program id would still pass
    the B=1 pos=0 case, so pin it against a position the row index cannot be."""
    from freetoken.kernel.triton.dsv4.rope import rope_decode_inplace

    rd = 64
    table = _table(rd, 3)
    x = torch.randn(1, 1, rd, device=DEV, dtype=torch.bfloat16, generator=_gen(4))
    got = rope_decode_inplace(x.clone(), table, False, positions=
                              torch.tensor([1234], device=DEV, dtype=torch.int64))
    want = rope_decode_inplace(x.clone(), table[1234:1235], False)
    wrong = rope_decode_inplace(x.clone(), table[0:1], False)
    assert torch.equal(got, want)
    assert not torch.equal(got, wrong)


# --------------------------------------------------------------------------- #
# the fused norm+rope stage
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "shape,D,heads,has_w",
    [
        ((1, 1, 64, 512), 512, 64, False),   # q, production shape
        ((4, 1, 64, 512), 512, 64, False),   # q, batched decode
        ((1, 1, 512), 512, 1, True),         # latent kv, production shape
        ((8, 1, 512), 512, 1, True),         # latent kv, batched
        ((1, 1, 576), 576, 1, True),         # D not a power of two
        ((1, 1, 64), 64, 1, True),           # the whole row is rope
        ((1, 1, 16, 512), 512, 16, True),    # heads + weight together
    ],
)
def test_fused_stage_is_bit_identical(shape, D, heads, has_w):
    from freetoken.models.deepseek_v4.ops import rms_norm_rope_decode

    rd = 64
    x, w, table, pos = _case(shape, D, rd, has_w, heads, seed=21)
    ref = _reference(x, w, table, pos, rd, False)
    fus = rms_norm_rope_decode(x, w, EPS, table, pos, rd, heads=heads)
    assert fus.shape == ref.shape and fus.dtype == ref.dtype
    assert torch.equal(fus, ref)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_stage_inverse_and_dtype(dtype):
    from freetoken.models.deepseek_v4.ops import rms_norm_rope_decode

    rd = 64
    x, w, table, pos = _case((2, 1, 512), 512, rd, True, 1, dtype=dtype, seed=31)
    for inverse in (False, True):
        ref = _reference(x, w, table, pos, rd, inverse)
        fus = rms_norm_rope_decode(x, w, EPS, table, pos, rd, inverse=inverse)
        assert torch.equal(fus, ref), f"inverse={inverse} dtype={dtype}"


@pytest.mark.parametrize("seed", range(6))
def test_fused_stage_over_many_draws(seed):
    from freetoken.models.deepseek_v4.ops import rms_norm_rope_decode

    rd = 64
    x, w, table, pos = _case((1, 1, 512), 512, rd, True, 1, seed=100 + seed)
    assert torch.equal(
        rms_norm_rope_decode(x, w, EPS, table, pos, rd),
        _reference(x, w, table, pos, rd, False),
    )


def test_fused_stage_leaves_the_input_alone():
    """rms_norm returns a fresh tensor; the fused stage must too, or the residual
    the caller still holds would be rotated out from under it."""
    from freetoken.models.deepseek_v4.ops import rms_norm_rope_decode

    rd = 64
    x, w, table, pos = _case((1, 1, 512), 512, rd, True, 1, seed=41)
    before = x.clone()
    out = rms_norm_rope_decode(x, w, EPS, table, pos, rd)
    assert torch.equal(x, before)
    assert out.data_ptr() != x.data_ptr()


def test_head_rows_share_their_token_position():
    """Every head of a token rotates by that token's position, not by its row
    index -- the mapping a flattened [B*H, D] grid has to reconstruct."""
    from freetoken.models.deepseek_v4.ops import rms_norm_rope_decode

    rd, D, H = 64, 512, 8
    x, w, table, _ = _case((2, 1, H, D), D, rd, True, H, seed=51)
    pos = torch.tensor([700, 3], device=DEV, dtype=torch.int64)
    fus = rms_norm_rope_decode(x, w, EPS, table, pos, rd, heads=H)
    ref = _reference(x, w, table, pos, rd, False)
    assert torch.equal(fus, ref)
    # and swapping the two tokens' positions must change both tokens' outputs
    other = rms_norm_rope_decode(
        x, w, EPS, table, pos.flip(0), rd, heads=H
    )
    assert not torch.equal(fus[0], other[0])
    assert not torch.equal(fus[1], other[1])


# --------------------------------------------------------------------------- #
# escape hatch + capture
# --------------------------------------------------------------------------- #


def test_unfused_env_gate_selects_the_reference_path(monkeypatch):
    from freetoken.kernel.triton.dsv4 import norm_rope
    from freetoken.models.deepseek_v4.ops import rms_norm_rope_decode

    assert not norm_rope.unfused_norm_rope()
    rd = 64
    x, w, table, pos = _case((1, 1, 512), 512, rd, True, 1, seed=61)
    fused = rms_norm_rope_decode(x, w, EPS, table, pos, rd)
    monkeypatch.setenv("FREETOKEN_UNFUSED_NORM_ROPE", "1")
    assert norm_rope.unfused_norm_rope()
    unfused = rms_norm_rope_decode(x, w, EPS, table, pos, rd)
    assert torch.equal(fused, unfused)


def test_fused_stage_is_cuda_graph_capturable():
    from freetoken.models.deepseek_v4.ops import rms_norm_rope_decode

    rd, D, H = 64, 512, 8
    x, w, table, _ = _case((1, 1, H, D), D, rd, True, H, seed=71)
    xbuf = x.clone()
    posbuf = torch.zeros(1, device=DEV, dtype=torch.int64)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            rms_norm_rope_decode(xbuf, w, EPS, table, posbuf, rd, heads=H)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = rms_norm_rope_decode(xbuf, w, EPS, table, posbuf, rd, heads=H)

    # Replay at a position and an input the capture never saw. `positions` is read
    # on the device, so the graph must rotate by the buffer's value at replay.
    x2 = torch.randn(1, 1, H, D, device=DEV, dtype=torch.bfloat16, generator=_gen(72))
    xbuf.copy_(x2)
    posbuf.copy_(torch.tensor([2049], device=DEV, dtype=torch.int64))
    graph.replay()
    torch.cuda.synchronize()
    replayed = out.clone()

    eager = rms_norm_rope_decode(
        x2, w, EPS, table, torch.tensor([2049], device=DEV, dtype=torch.int64),
        rd, heads=H,
    )
    assert torch.equal(replayed, eager)


def test_launch_count_collapses():
    from torch.autograd.profiler_util import DeviceType

    from freetoken.models.deepseek_v4.ops import rms_norm_rope_decode

    rd, D, H = 64, 512, 8
    x, w, table, pos = _case((1, 1, H, D), D, rd, True, H, seed=81)

    def count(fn):
        fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            fn()
            torch.cuda.synchronize()
        return sum(
            e.count
            for e in prof.key_averages()
            if e.device_type == DeviceType.CUDA and e.self_device_time_total > 0
        )

    n_fused = count(lambda: rms_norm_rope_decode(x, w, EPS, table, pos, rd, heads=H))
    n_ref = count(lambda: _reference(x, w, table, pos, rd, False))
    assert n_fused == 1, f"expected 1 launch, got {n_fused}"
    assert n_ref == 3, f"expected the 3-launch reference, got {n_ref}"
