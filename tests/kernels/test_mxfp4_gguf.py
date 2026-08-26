"""MXFP4 (ggml type 39) support in the vendored ggml CUDA kernels.

unsloth's UD-Q2_K_XL quantises the DeepSeek-V4-Flash ``down`` tensors of layers 26
and 42 as MXFP4 -- 4.25 bpw against IQ3_XXS's 3.06 -- precisely because those two
layers are the sensitive ones. Until these kernels existed the loader dequantised
those tensors and re-encoded them to Q2_K RTN to fit the bank pitch, which threw
the extra precision away. So the property under test is not "MXFP4 roughly works"
but "MXFP4 is carried byte-for-byte and decoded exactly".

Three things get their own test:

* **Dequant is exact, not close.** An MXFP4 value is a power-of-two E8M0 scale
  times one of eight E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}. Every one of
  those has at most two mantissa bits, so ``scale * value`` is *exactly*
  representable in bf16 -- there is no rounding anywhere in the chain and the
  reference can be compared with ``torch.equal``. An ``allclose`` here would pass
  with the doubled-LUT ``* 0.5f`` compensation dropped on a subset of codes.

* **The padded row pitch composes.** block_mxfp4 is 17 bytes: alignof 1, and the
  first type in this bank whose blocks do not tile a 16 B-multiple pitch. The
  kernel must address the row base in bytes and then tile blocks from there,
  reading exactly ``blocks_per_row`` of them and never the padding tail.

* **The batched launcher is bit-identical.** MXFP4 is the first batched type with
  qk != QK_K, so the ``iby = i * (qk / QK8_1)`` activation walk takes a different
  shape (stride 1 over 64 blocks instead of stride 8 over 8).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

GGML_MXFP4 = 39
GGML_IQ3_XXS = 18

QK_MXFP4 = 32
MXFP4_BLOCK_BYTES = 17  # 1 e8m0 byte + 16 nibble bytes
IQ3_XXS_BLOCK_BYTES = 98
QK_K = 256

# The production down-bank geometry: 2048 columns, so a native MXFP4 row is
# 2048 / 32 * 17 = 1088 B and a native IQ3_XXS row is 2048 / 256 * 98 = 784 B.
H = 2048
MXFP4_NATIVE = H // QK_MXFP4 * MXFP4_BLOCK_BYTES        # 1088
IQ3_NATIVE = H // QK_K * IQ3_XXS_BLOCK_BYTES            # 784

# E2M1 code -> magnitude. Codes 8..15 are the negatives of 0..7.
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)


# --------------------------------------------------------------------------- #
# numpy reference
# --------------------------------------------------------------------------- #


def _mxfp4_reference(blocks: np.ndarray) -> np.ndarray:
    """``[nb, 17]`` uint8 -> ``[nb, 32]`` float64, the MXFP4 spec by hand.

    Nibble packing follows ggml: byte ``j`` of ``qs`` holds element ``j`` in its
    low nibble and element ``j + 16`` in its high nibble.
    """
    assert blocks.dtype == np.uint8 and blocks.shape[1] == MXFP4_BLOCK_BYTES
    e = blocks[:, 0].astype(np.int64)
    qs = blocks[:, 1:]

    # E8M0: code 0 denotes 2^-127, code x denotes 2^(x-127). (255 is NaN in the
    # OCP spec and never appears in a real GGUF; we never generate it.)
    scale = np.where(e == 0, 2.0**-127, 2.0 ** (e.astype(np.float64) - 127.0))

    def lut(codes: np.ndarray) -> np.ndarray:
        return np.where(codes >= 8, -E2M1[codes & 7], E2M1[codes & 7])

    out = np.empty((blocks.shape[0], QK_MXFP4), dtype=np.float64)
    out[:, :16] = lut(qs & 0x0F)
    out[:, 16:] = lut(qs >> 4)
    return out * scale[:, None]


def _make_mxfp4(nb: int, seed: int, emin: int = 100, emax: int = 150) -> np.ndarray:
    """``[nb, 17]`` uint8 of random MXFP4 blocks.

    Exponents are kept in a narrow band so the products stay well inside bf16's
    range: an overflow to +/-inf would make an exact comparison vacuous. Nibbles
    are fully random -- all 16 E2M1 codes are legal, including both zeros.
    """
    rng = np.random.default_rng(seed)
    blocks = np.empty((nb, MXFP4_BLOCK_BYTES), dtype=np.uint8)
    blocks[:, 0] = rng.integers(emin, emax + 1, size=nb, dtype=np.uint8)
    blocks[:, 1:] = rng.integers(0, 256, size=(nb, MXFP4_BLOCK_BYTES - 1), dtype=np.uint8)
    return blocks


# --------------------------------------------------------------------------- #
# 1. dequant
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dequantize_matches_numpy_exactly(dtype):
    from freetoken.kernel.gguf import ggml_dequantize

    rows, nb_per_row = 37, H // QK_MXFP4
    blocks = _make_mxfp4(rows * nb_per_row, seed=0, emin=118, emax=134)
    ref = _mxfp4_reference(blocks).reshape(rows, H)

    w = torch.from_numpy(blocks.reshape(rows, -1)).cuda()
    got = ggml_dequantize(w, GGML_MXFP4, rows, H, dtype)

    want = torch.from_numpy(ref).to(dtype).cuda()
    assert got.shape == want.shape
    assert torch.isfinite(got).all()
    if not torch.equal(got, want):
        bad = (got != want).nonzero()
        i, j = bad[0].tolist()
        raise AssertionError(
            f"{bad.shape[0]} of {rows * H} elements differ; first at [{i},{j}]: "
            f"got {got[i, j].item()} want {want[i, j].item()} "
            f"(e={blocks[i * nb_per_row + j // 32, 0]}, "
            f"qs={blocks[i * nb_per_row + j // 32, 1 + (j % 32) % 16]:#04x})"
        )


def test_dequantize_covers_every_e2m1_code():
    """All 16 nibble codes, at one fixed scale -- the exhaustive check the random
    test only reaches statistically, and the one that pins the sign convention and
    the zero/negative-zero codes."""
    from freetoken.kernel.gguf import ggml_dequantize

    nb = H // QK_MXFP4
    blocks = np.zeros((nb, MXFP4_BLOCK_BYTES), dtype=np.uint8)
    blocks[:, 0] = 127  # scale exactly 1.0
    # qs byte j = (code_hi << 4) | code_lo; sweep every (lo, hi) pair over 16 bytes
    # x 4 blocks = 64 combinations >= 256 across the row set.
    for b in range(nb):
        for j in range(16):
            lo = (b * 16 + j) % 16
            hi = ((b * 16 + j) // 16) % 16
            blocks[b, 1 + j] = (hi << 4) | lo

    ref = _mxfp4_reference(blocks).reshape(1, H)
    w = torch.from_numpy(blocks.reshape(1, -1)).cuda()
    got = ggml_dequantize(w, GGML_MXFP4, 1, H, torch.bfloat16)
    want = torch.from_numpy(ref).to(torch.bfloat16).cuda()
    assert torch.equal(got, want)
    # The test is only meaningful if it actually exercised the whole table.
    assert len(set(want.flatten().float().tolist())) == 15  # +0 and -0 collapse


# --------------------------------------------------------------------------- #
# fixtures for the GEMV tests
# --------------------------------------------------------------------------- #

NROWS = 64
TOP_K = 6


def _mxfp4_bank(num_experts: int, nrows: int, pitch: int, seed: int) -> torch.Tensor:
    """``[E, nrows, pitch]`` uint8: native MXFP4 rows plus random padding tail.

    The tail is deliberately random rather than zero -- the whole point of the
    pitched-copy design in the loader is that those bytes are never read, and a
    zero tail would hide a kernel that read them.
    """
    assert pitch >= MXFP4_NATIVE
    nb = H // QK_MXFP4
    blocks = _make_mxfp4(num_experts * nrows * nb, seed=seed).reshape(num_experts, nrows, -1)
    rng = np.random.default_rng(seed + 5000)
    bank = rng.integers(0, 256, size=(num_experts, nrows, pitch), dtype=np.uint8)
    bank[:, :, :MXFP4_NATIVE] = blocks
    return torch.from_numpy(bank).cuda()


def _iq3_rows(num_experts: int, nrows: int, seed: int) -> torch.Tensor:
    """``[E, nrows, 784]`` of natively packed IQ3_XXS rows, scale bytes forced
    finite so a random bit pattern cannot hand us NaN and make the comparison
    vacuous. Generated once at the native width so the same content can be laid
    out at two different pitches."""
    nb = H // QK_K
    g = torch.Generator().manual_seed(seed)
    rows = torch.randint(0, 256, (num_experts, nrows, IQ3_NATIVE), generator=g, dtype=torch.uint8)
    d = (0.01 + 0.04 * torch.rand(num_experts, nrows, nb, generator=g)).to(torch.float16)
    d_bytes = d.view(torch.uint8).reshape(num_experts, nrows, nb, 2)
    for b in range(nb):
        rows[:, :, b * IQ3_XXS_BLOCK_BYTES : b * IQ3_XXS_BLOCK_BYTES + 2] = d_bytes[:, :, b]
    return rows


def _at_pitch(rows: torch.Tensor, pitch: int, seed: int) -> torch.Tensor:
    """Re-lay ``rows`` at ``pitch``, filling the tail with random garbage."""
    e, n, native = rows.shape
    assert pitch >= native
    rng = np.random.default_rng(seed)
    bank = torch.from_numpy(rng.integers(0, 256, size=(e, n, pitch), dtype=np.uint8))
    bank[:, :, :native] = rows
    return bank.cuda()


def _make_x(rows: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(rows, H, generator=g, device="cuda") * 0.5).to(torch.bfloat16)


def _ids(tokens: int, top_k: int, num_experts: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(
        0, num_experts, (tokens, top_k), generator=g, device="cuda", dtype=torch.int32
    )


def _vec(x, bank, ids, top_k, tokens, qtype, pitch, nrows=NROWS):
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    y = ggml_moe_a8_vec(x, bank, ids, top_k, qtype, nrows, tokens, pitch)
    torch.cuda.synchronize()
    return y


def _vec_batched(x, bank, ids, top_k, tokens, qtype, pitch, n, num_experts, nrows=NROWS):
    from freetoken.kernel.gguf import ggml_moe_a8_vec_batched
    from freetoken.moe.fused_q2_k_ud import _expert_group_perm

    perm = _expert_group_perm(ids, num_experts, n)
    y = ggml_moe_a8_vec_batched(
        x, bank, ids, perm, top_k, qtype, nrows, tokens, pitch, n, None
    )
    torch.cuda.synchronize()
    return y


def _sane(y: torch.Tensor) -> None:
    assert torch.isfinite(y).all(), "kernel produced non-finite output"
    assert y.abs().sum().item() > 0, "kernel output is entirely zero -- test is vacuous"


def _assert_identical(ref: torch.Tensor, got: torch.Tensor) -> None:
    if torch.equal(ref, got):
        return
    bad = (ref != got).any(dim=1).nonzero(as_tuple=True)[0]
    delta = (ref.to(torch.float32) - got.to(torch.float32)).abs().max().item()
    raise AssertionError(
        f"{bad.numel()} of {ref.shape[0]} routed rows differ; "
        f"first at routed row {bad[0].item()}; max abs delta {delta}"
    )


# --------------------------------------------------------------------------- #
# 2. GEMV: pitch independence
# --------------------------------------------------------------------------- #


# 1088 is the native (and shipped) pitch; 1120 pads by two blocks' worth and is
# NOT a multiple of 17 (1120 % 17 == 15), which is the case a block-granular
# pitch implementation would get wrong; 1152 pads further still.
@pytest.mark.parametrize("pitch", [MXFP4_NATIVE, 1120, 1152])
def test_padded_pitch_matches_tight_bank(pitch):
    """Same rows, different row stride -> byte-identical output.

    ``row_pitch_bytes = 0`` means "tightly packed at the native width", so the
    tight bank is the reference the padded ones have to reproduce.
    """
    tokens, num_experts = 512, 8
    tight = _mxfp4_bank(num_experts, NROWS, MXFP4_NATIVE, seed=101)
    padded = _mxfp4_bank(num_experts, NROWS, pitch, seed=101)
    # Same weights, different stride: the seed drives the block content, and the
    # native prefix is written after the random fill.
    assert torch.equal(tight[:, :, :MXFP4_NATIVE], padded[:, :, :MXFP4_NATIVE])

    x = _make_x(tokens, seed=102)
    ids = _ids(tokens, TOP_K, num_experts, seed=103)

    ref = _vec(x, tight, ids, TOP_K, tokens, GGML_MXFP4, 0)
    got = _vec(x, padded, ids, TOP_K, tokens, GGML_MXFP4, pitch)
    _sane(ref)
    _assert_identical(ref, got)


def test_padding_tail_is_never_read():
    """Two banks identical in their native prefix, adversarially different in the
    tail. If the kernel ever ran past ``blocks_per_row`` the outputs would differ.
    """
    tokens, num_experts, pitch = 256, 8, 1152
    a = _mxfp4_bank(num_experts, NROWS, pitch, seed=111)
    b = a.clone()
    b[:, :, MXFP4_NATIVE:] = 0xFF  # e8m0 0xFF is NaN; a stray read would poison the sum
    assert torch.equal(a[:, :, :MXFP4_NATIVE], b[:, :, :MXFP4_NATIVE])

    x = _make_x(tokens, seed=112)
    ids = _ids(tokens, TOP_K, num_experts, seed=113)

    ya = _vec(x, a, ids, TOP_K, tokens, GGML_MXFP4, pitch)
    yb = _vec(x, b, ids, TOP_K, tokens, GGML_MXFP4, pitch)
    _sane(ya)
    _assert_identical(ya, yb)


def test_iq3_xxs_still_matches_at_the_widened_down_pitch():
    """The shipped down bank raises its pitch to 1088 for ALL 43 layers so the two
    MXFP4 layers fit. The 41 IQ3_XXS layers now sit in a bank whose pitch is 1088
    against a 784 B native row -- and 1088 % 98 != 0, so this is the byte-granular
    case, not a block-granular one."""
    assert MXFP4_NATIVE % IQ3_XXS_BLOCK_BYTES != 0
    tokens, num_experts = 256, 8
    rows = _iq3_rows(num_experts, NROWS, seed=121)
    tight = _at_pitch(rows, IQ3_NATIVE, seed=1211)
    wide = _at_pitch(rows, MXFP4_NATIVE, seed=1212)
    assert torch.equal(tight[:, :, :IQ3_NATIVE], wide[:, :, :IQ3_NATIVE])

    x = _make_x(tokens, seed=122)
    ids = _ids(tokens, TOP_K, num_experts, seed=123)

    ref = _vec(x, tight, ids, TOP_K, tokens, GGML_IQ3_XXS, 0)
    got = _vec(x, wide, ids, TOP_K, tokens, GGML_IQ3_XXS, MXFP4_NATIVE)
    _sane(ref)
    _assert_identical(ref, got)


# --------------------------------------------------------------------------- #
# 3. GEMV: numeric sanity against the dequantised weights
# --------------------------------------------------------------------------- #


def test_gemv_tracks_the_dequantised_reference():
    """Exact-match tests above prove self-consistency; this one proves the vec_dot
    computes the RIGHT dot product.

    The kernel quantises the activations to q8_1 first, so the comparison against
    an fp32 matmul of the dequantised weights is necessarily approximate -- but the
    tolerance is tight enough to catch a wrong LUT, a dropped 0.5f, or a misread
    e8m0 byte, all of which move the result by tens of percent.
    """
    from freetoken.kernel.gguf import ggml_dequantize

    tokens, num_experts, top_k = 64, 4, 2
    bank = _mxfp4_bank(num_experts, NROWS, MXFP4_NATIVE, seed=131)
    x = _make_x(tokens, seed=132)
    ids = _ids(tokens, top_k, num_experts, seed=133)

    got = _vec(x, bank, ids, top_k, tokens, GGML_MXFP4, MXFP4_NATIVE)

    w = ggml_dequantize(
        bank.reshape(num_experts * NROWS, MXFP4_NATIVE), GGML_MXFP4,
        num_experts * NROWS, H, torch.float32,
    ).reshape(num_experts, NROWS, H)
    xf = x.to(torch.float32)
    ref = torch.stack(
        [xf[t] @ w[int(ids[t, k])].T for t in range(tokens) for k in range(top_k)]
    )

    _sane(got)
    err = (got.to(torch.float32) - ref).norm() / ref.norm()
    assert err < 0.02, f"relative L2 error {err.item():.4f} is too large for a q8_1 GEMV"


# --------------------------------------------------------------------------- #
# 4. batched launcher
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_batched_is_bit_identical(n):
    from freetoken.kernel.gguf import ggml_moe_vec_batched_supported

    assert ggml_moe_vec_batched_supported(GGML_MXFP4)

    tokens, num_experts, pitch = 1024, 16, MXFP4_NATIVE
    bank = _mxfp4_bank(num_experts, NROWS, pitch, seed=141)
    x = _make_x(tokens, seed=142)
    ids = _ids(tokens, TOP_K, num_experts, seed=143)

    ref = _vec(x, bank, ids, TOP_K, tokens, GGML_MXFP4, pitch)
    got = _vec_batched(x, bank, ids, TOP_K, tokens, GGML_MXFP4, pitch, n, num_experts)
    _sane(ref)
    assert ref.shape == got.shape == (tokens * TOP_K, NROWS)
    _assert_identical(ref, got)


@pytest.mark.parametrize("n", [4, 8])
def test_batched_with_padded_pitch_and_top_k_one(n):
    """The down GEMV's actual shape: tokens = T * top_k routed rows at top_k = 1,
    over a bank whose pitch exceeds the native row."""
    routed, num_experts, pitch = 4096, 16, 1152
    bank = _mxfp4_bank(num_experts, NROWS, pitch, seed=151)
    x = _make_x(routed, seed=152)
    ids = _ids(routed, 1, num_experts, seed=153)

    ref = _vec(x, bank, ids, 1, routed, GGML_MXFP4, pitch)
    got = _vec_batched(x, bank, ids, 1, routed, GGML_MXFP4, pitch, n, num_experts)
    _sane(ref)
    _assert_identical(ref, got)


# --------------------------------------------------------------------------- #
# 5. gridDim.z slicing
# --------------------------------------------------------------------------- #


def test_z_slicing_past_the_grid_cap():
    """gridDim.z caps at 65535. At top_k = 6 that bites at 10923 tokens, so the
    launcher's slicing loop has to hand each launch its own ``z_offset`` -- and the
    kernel has to derive ``token = z / topk`` from the ABSOLUTE z, not blockIdx.z.
    A missing offset shows up as the tail of the output being a copy of the head.
    """
    tokens, num_experts, pitch = 12000, 8, MXFP4_NATIVE
    routed = tokens * TOP_K
    assert routed > 65535

    bank = _mxfp4_bank(num_experts, 16, pitch, seed=161)
    x = _make_x(tokens, seed=162)
    ids = _ids(tokens, TOP_K, num_experts, seed=163)

    got = _vec(x, bank, ids, TOP_K, tokens, GGML_MXFP4, pitch, nrows=16)
    _sane(got)
    assert got.shape == (routed, 16)

    # Rows past the cap must match a launch that never crossed it. Re-running the
    # tail as its own small problem is the cheapest independent reference.
    tail_start = 65535 // TOP_K + 1          # first token whose z exceeds the cap
    n_tail = tokens - tail_start
    tail = _vec(
        x[tail_start:], bank, ids[tail_start:].contiguous(), TOP_K, n_tail,
        GGML_MXFP4, pitch, nrows=16,
    )
    _assert_identical(tail, got[tail_start * TOP_K :])


@pytest.mark.parametrize("n", [4])
def test_batched_z_slicing_past_the_grid_cap(n):
    """Same cap, counted in GROUPS on the batched path."""
    tokens, num_experts, pitch = 48000, 8, MXFP4_NATIVE
    assert tokens * TOP_K // n > 65535

    bank = _mxfp4_bank(num_experts, 16, pitch, seed=171)
    x = _make_x(tokens, seed=172)
    ids = _ids(tokens, TOP_K, num_experts, seed=173)

    ref = _vec(x, bank, ids, TOP_K, tokens, GGML_MXFP4, pitch, nrows=16)
    got = _vec_batched(x, bank, ids, TOP_K, tokens, GGML_MXFP4, pitch, n, num_experts, nrows=16)
    _sane(ref)
    _assert_identical(ref, got)


# --------------------------------------------------------------------------- #
# 6. type metadata
# --------------------------------------------------------------------------- #


def test_block_geometry_agrees_across_the_c_and_python_tables():
    from freetoken.kernel.gguf import ggml_type_block_bytes, ggml_type_block_elems
    from freetoken.models.gguf.dequant import BLOCK_SHAPE, GGML_MXFP4 as PY_MXFP4

    assert PY_MXFP4 == GGML_MXFP4
    elems, nbytes = BLOCK_SHAPE[PY_MXFP4]
    assert (elems, nbytes) == (QK_MXFP4, MXFP4_BLOCK_BYTES)
    assert ggml_type_block_bytes(GGML_MXFP4) == nbytes
    assert ggml_type_block_elems(GGML_MXFP4) == elems
    # ...and that is exactly the shipped down-row width.
    assert H // elems * nbytes == MXFP4_NATIVE == 1088
