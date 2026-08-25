"""Fused FP8 decode GEMV (act-quant prologue + split-k epilogue) parity.

The fused kernel replaces three launches (``_act_quant_fp8_kernel`` ->
``_fp8_act_gemv_splitk_kernel`` -> ``_splitk_reduce_kernel``) with one. Its
activation quantization must be BIT-IDENTICAL to the materialized one, and at the
shipped ``SPLIT_K == 1`` configs the whole projection must be bit-identical to the
unfused path (the epilogue is then a plain store of the same accumulator).
"""

import pytest
import torch

triton = pytest.importorskip("triton")

from freetoken.kernel.triton.dsv4.fp8_linear import (  # noqa: E402
    act_quant_fp8,
    block_fp8_linear,
    fused_fp8_gemv,
    grouped_block_fp8_linear,
    _decode_cfg,
    _fused_cfg,
)
from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

FP8 = torch.float8_e4m3fn

# Every distinct (N, K) the DSV4 decode step runs, plus a couple of odd ones.
DECODE_SHAPES = [
    (1024, 4096),   # wq_a
    (32768, 1024),  # wq_b
    (512, 4096),    # wkv
    (4096, 8192),   # wo_b
    (2048, 4096),   # shared w1 / w3
    (4096, 2048),   # shared w2
    (8192, 1024),   # indexer wq_b
    (128, 128),     # smallest legal
    (256, 512),
]


def _weights(N, K, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    w = (torch.randn(N, K, generator=g, device=device) * 0.05).to(FP8)
    sb = torch.randint(
        112, 142, (N // 128, K // 128), generator=g, dtype=torch.uint8, device=device
    )
    return w, sb


def _acts(K, kind, device="cuda", seed=1, rows=1):
    g = torch.Generator(device=device).manual_seed(seed)
    if kind == "normal":
        x = torch.randn(rows, K, generator=g, device=device)
    elif kind == "tiny":  # below act_quant's 1e-4 amax floor -> the clamp branch
        x = torch.randn(rows, K, generator=g, device=device) * 1e-8
    elif kind == "huge":
        x = torch.randn(rows, K, generator=g, device=device) * 1e4
    elif kind == "zeros":
        x = torch.zeros(rows, K, device=device)
    elif kind == "mixed":  # per-128 blocks spanning ~40 binades
        x = torch.randn(rows, K, generator=g, device=device)
        scales = torch.logspace(-20, 20, K // 128, base=2.0, device=device)
        x = (x.view(rows, K // 128, 128) * scales[None, :, None]).reshape(rows, K)
    elif kind == "denormal":  # e4m3 subnormal grid after scaling
        x = torch.randn(rows, K, generator=g, device=device) * 448.0 * 2.0**-9
        x[:, ::7] *= 448.0
    elif kind == "spike":  # one huge value per block, rest ~0 -> maximal quant error
        x = torch.full((rows, K), 1e-6, device=device)
        x[:, ::128] = 1e3
    else:
        raise ValueError(kind)
    return x.to(torch.bfloat16)


KINDS = ["normal", "tiny", "huge", "zeros", "mixed", "denormal", "spike"]


# --------------------------------------------------------------------------------------
# 1. The prologue reproduces act_quant_fp8 bit-for-bit.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("K", [128, 1024, 4096])
def test_prologue_quant_is_bit_identical(kind, K):
    """A one-hot FP8 weight with unit block scales makes the GEMV an identity read of
    the quantized activation, so the output exposes ``a_fp8[k] * scale[k // 128]``
    directly. Compare against the materialized ``act_quant_fp8``."""
    N = K
    w = torch.zeros(N, K, device="cuda")
    w.fill_diagonal_(1.0)
    w = w.to(FP8)
    sb = torch.full((N // 128, K // 128), 127, dtype=torch.uint8, device="cuda")

    x = _acts(K, kind)
    a_fp8, sa = act_quant_fp8(x, 128)
    ref = (
        a_fp8[0].to(torch.float32)
        * torch.exp2(sa[0].to(torch.float32) - 127.0).repeat_interleave(128)
    ).to(torch.float32)

    got = fused_fp8_gemv(
        x, e4m3_kernel_view(w), sb, torch.float32, cfg=(16, 1, 1, 2)
    )
    assert torch.equal(got, ref), (
        kind, K, (got - ref).abs().max().item(), (got != ref).sum().item()
    )


# --------------------------------------------------------------------------------------
# 2. Whole projection: fused == unfused, bit-for-bit, at matching (BLOCK_N, SPLIT_K).
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("N,K", DECODE_SHAPES)
@pytest.mark.parametrize("kind", KINDS)
def test_fused_matches_unfused_bitwise_same_cfg(N, K, kind):
    w, sb = _weights(N, K)
    x = _acts(K, kind)
    bn, sk, nw = _decode_cfg(N, K)
    if N % bn:
        pytest.skip("cfg BLOCK_N does not divide N")
    ref = block_fp8_linear(x, w, sb, fused=False)
    got = fused_fp8_gemv(x, e4m3_kernel_view(w), sb, torch.bfloat16, cfg=(bn, sk, nw, 3))
    assert torch.equal(got, ref[0]), (
        N, K, kind, (got.float() - ref[0].float()).abs().max().item()
    )


# --------------------------------------------------------------------------------------
# 3. Whole projection at the SHIPPED fused config (which may retune BLOCK_N/SPLIT_K).
#    A different SPLIT_K reassociates the fp32 accumulation, so this is a tolerance
#    check, not a bitwise one -- unless the shipped config happens to match.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("N,K", DECODE_SHAPES)
@pytest.mark.parametrize("kind", KINDS)
def test_fused_matches_unfused_shipped_cfg(N, K, kind):
    w, sb = _weights(N, K)
    x = _acts(K, kind)
    ref = block_fp8_linear(x, w, sb, fused=False)[0].float()
    got = block_fp8_linear(x, w, sb, fused=True)[0].float()
    bn, sk, nw, ns = _fused_cfg(N, K)
    if (bn, sk) == _decode_cfg(N, K)[:2]:
        assert torch.equal(got, ref), (N, K, kind)
        return
    # Both sides round the fp32 accumulator to bf16; a reassociated split-k can land on
    # the other side of a tie, so allow two bf16 ulps at the output's peak magnitude.
    scale = ref.abs().max().clamp_min(1e-30)
    assert (got - ref).abs().max() <= 2.0 ** -7 * scale, (
        N, K, kind, (got - ref).abs().max().item(), scale.item()
    )


# --------------------------------------------------------------------------------------
# 4. Grouped (block-diagonal) form -- the wo_a shape.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("G,R,D", [(8, 1024, 4096), (2, 256, 512), (4, 128, 128)])
@pytest.mark.parametrize("kind", ["normal", "mixed", "spike"])
def test_grouped_matches_per_group(G, R, D, kind):
    """The block-diagonal form must be bit-identical to G independent GEMVs at the same
    config -- each output row sees the same rows, the same k loop and the same scales."""
    w, sb = _weights(G * R, D)
    wk = e4m3_kernel_view(w)
    x = _acts(D, kind, rows=G)
    cfg = (8, min(4, D // 128), 1, 3)
    ref = torch.cat([
        fused_fp8_gemv(x[g], wk[g * R:(g + 1) * R], sb[g * R // 128:(g + 1) * R // 128],
                       torch.bfloat16, cfg=cfg)
        for g in range(G)
    ])
    got = fused_fp8_gemv(x, wk, sb, torch.bfloat16, group_rows=R, cfg=cfg)
    assert torch.equal(got, ref), (G, R, D, kind,
                                   (got.float() - ref.float()).abs().max().item())


def test_grouped_module_wrapper_shapes():
    """``grouped_block_fp8_linear`` decode (one fused launch) == its prefill fallback
    (one block GEMM per group) on the same single row."""
    G, R, D = 8, 1024, 4096
    w, sb = _weights(G * R, D)
    x = _acts(D, "normal", rows=G).reshape(1, 1, G, D)
    got = grouped_block_fp8_linear(x, w, sb, R)
    assert got.shape == (1, 1, G * R)
    two = grouped_block_fp8_linear(x.expand(1, 2, G, D).contiguous(), w, sb, R)
    assert two.shape == (1, 2, G * R)
    rel = (got[0, 0].float() - two[0, 0].float()).abs().max() / two.float().abs().max()
    assert rel < 1e-2, rel.item()


def test_grouped_fp8_vs_bf16_einsum_sanity():
    """wo_a's production shape: the FP8 grouped GEMV against the bf16 einsum it replaces.
    A loose bound -- the real numeric-delta measurement lives in the bench script; this
    only catches a wired-up-wrong kernel."""
    G, R, D = 8, 1024, 4096
    torch.manual_seed(0)
    from freetoken.models.deepseek_v4.weight import _dequant_fp8_block, quantize_fp8_block

    w_bf16 = (torch.randn(G * R, D, device="cuda") * 0.02).to(torch.bfloat16)
    w, sb = quantize_fp8_block(w_bf16.float())
    deq = _dequant_fp8_block(w, sb)  # what the bf16 path would actually hold
    x = _acts(D, "normal", rows=G)
    ref = torch.einsum("gd,grd->gr", x.float(), deq.view(G, R, D).float()).flatten()
    got = grouped_block_fp8_linear(x, w, sb, R).float()
    rel = (got - ref).abs().max() / ref.abs().max()
    assert rel < 0.05, rel.item()


# --------------------------------------------------------------------------------------
# 5. Graph capture: fixed shapes, no host branching on data.
# --------------------------------------------------------------------------------------
def test_fused_gemv_is_graph_capturable():
    N, K = 2048, 4096
    w, sb = _weights(N, K)
    wk = e4m3_kernel_view(w)
    x = _acts(K, "normal")
    out_eager = fused_fp8_gemv(x, wk, sb, torch.bfloat16).clone()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fused_fp8_gemv(x, wk, sb, torch.bfloat16)
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_graph = fused_fp8_gemv(x, wk, sb, torch.bfloat16)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out_graph, out_eager)

    x2 = _acts(K, "mixed", seed=7)
    x.copy_(x2)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out_graph, fused_fp8_gemv(x, wk, sb, torch.bfloat16))
