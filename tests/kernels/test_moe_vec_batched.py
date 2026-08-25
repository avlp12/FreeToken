"""Weight-reuse batching for the grouped-expert GEMV: it must change nothing.

``moe_vec_q`` gives one CUDA block to each (routed row, weight row) pair, so at an
8192-token prefill chunk with top_k = 6 every expert's weight matrix is re-read
~192 times per layer -- 20.4 TB of HBM traffic per chunk, 89% of prefill wall
time. ``moe_vec_q_batched`` hands one block N routed rows that share an expert and
calls ``vec_dot_*_q8_1`` N times against the SAME weight block pointer, so the
weight row is fetched once per N rows.

That is a scheduling change and nothing else. Each output element is the same dot
product accumulated in the same order, through the same warp-reduction shuffle
sequence, into the same dst slot. So the bar here is ``torch.equal``, not
``allclose``: an exact-equality test is the only kind that can distinguish "we
reordered the work" from "we reordered the arithmetic", and the second would be a
silent accuracy regression nobody would notice until the model got dumber.

The other thing worth failing over is the -1 padding sentinel. Each expert's run
is padded up to a multiple of N, and a padding lane that forgets to skip its store
writes ``dst[-1 * nrows + row]`` -- straight off the front of the output
allocation, into whatever the caching allocator handed out before it. Silent
memory corruption, not a crash. ``test_padding_lanes_never_store`` places the
output inside a poisoned guard band so that write has somewhere visible to land.

Geometry mirrors the production q2_k_ud banks: rows padded to one uniform byte
pitch shared across quant types, so the pitch is NOT a whole number of blocks of
either type living in it (400 % 74 != 0 for IQ2_XS, 400 % 98 != 0 for IQ3_XXS) --
exactly like the shipped 1568 B gate_up bank.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

QK_K = 256
GGML_IQ2_XS = 17
GGML_IQ3_XXS = 18
BLOCK_BYTES = {GGML_IQ2_XS: 74, GGML_IQ3_XXS: 98}

H = 1024        # ncols -- 4 super-blocks per row
PITCH = 400     # >= max native row (4 * 98 = 392), 16 B-multiple, not a block multiple
NROWS = 64      # output features per expert
TOP_K = 6

BATCH_WIDTHS = (4, 8, 16)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def _make_bank(qtype: int, num_experts: int, nrows: int = NROWS, seed: int = 0) -> torch.Tensor:
    """``[E, nrows, PITCH]`` uint8 bank of ``qtype`` rows with padded trailing bytes.

    Random bytes throughout -- the IQ grid/sign/scale nibbles are total over their
    lookup tables -- except the fp16 super-block scale ``d`` at each block head,
    which is written explicitly so a random bit pattern cannot hand us NaN/Inf and
    make an exact comparison vacuous. The pad bytes stay random: the kernel must
    never read them.
    """
    blk = BLOCK_BYTES[qtype]
    nb = H // QK_K
    g = torch.Generator().manual_seed(seed)
    bank = torch.randint(0, 256, (num_experts, nrows, PITCH), generator=g, dtype=torch.uint8)
    d = (0.01 + 0.04 * torch.rand(num_experts, nrows, nb, generator=g)).to(torch.float16)
    d_bytes = d.view(torch.uint8).reshape(num_experts, nrows, nb, 2)
    for b in range(nb):
        bank[:, :, b * blk : b * blk + 2] = d_bytes[:, :, b]
    return bank.cuda()


def _make_x(rows: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(rows, H, generator=g, device="cuda") * 0.5).to(torch.bfloat16)


def _uniform_ids(tokens: int, top_k: int, num_experts: int, seed: int = 2) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(
        0, num_experts, (tokens, top_k), generator=g, device="cuda", dtype=torch.int32
    )


def _skewed_ids(tokens: int, top_k: int, num_experts: int, seed: int = 3) -> torch.Tensor:
    """One expert takes ~90% of the routes; most experts get zero.

    Zero-count experts contribute a zero-length padded run, which is the case that
    breaks a grouping built from ``cumsum`` over counts if the padding arithmetic
    is off by a run.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    hot = num_experts // 2
    ids = torch.full((tokens, top_k), hot, device="cuda", dtype=torch.int32)
    r = torch.rand(tokens, top_k, generator=g, device="cuda")
    cold = torch.tensor([1, 2, num_experts - 1], device="cuda", dtype=torch.int32)
    pick = torch.randint(0, 3, (tokens, top_k), generator=g, device="cuda")
    return torch.where(r < 0.1, cold[pick], ids).contiguous()


# --------------------------------------------------------------------------- #
# runners
# --------------------------------------------------------------------------- #


def _unbatched(x, bank, ids, top_k, tokens, qtype, nrows=NROWS):
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    y = ggml_moe_a8_vec(x, bank, ids, top_k, qtype, nrows, tokens, PITCH)
    torch.cuda.synchronize()
    return y


def _batched(x, bank, ids, top_k, tokens, qtype, n, num_experts, nrows=NROWS, out=None):
    from freetoken.kernel.gguf import ggml_moe_a8_vec_batched
    from freetoken.moe.fused_q2_k_ud import _expert_group_perm

    perm = _expert_group_perm(ids, num_experts, n)
    y = ggml_moe_a8_vec_batched(
        x, bank, ids, perm, top_k, qtype, nrows, tokens, PITCH, n, out
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
# 1. the headline: bit-identical to the unbatched kernel
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", BATCH_WIDTHS)
@pytest.mark.parametrize("qtype", [GGML_IQ2_XS, GGML_IQ3_XXS])
def test_batched_is_bit_identical(n, qtype):
    """Both q2_k_ud bank types, every instantiated batch width."""
    tokens, num_experts = 1024, 16
    bank = _make_bank(qtype, num_experts)
    x = _make_x(tokens)
    ids = _uniform_ids(tokens, TOP_K, num_experts)

    ref = _unbatched(x, bank, ids, TOP_K, tokens, qtype)
    got = _batched(x, bank, ids, TOP_K, tokens, qtype, n, num_experts)
    _sane(ref)
    assert ref.shape == got.shape == (tokens * TOP_K, NROWS)
    _assert_identical(ref, got)


@pytest.mark.parametrize("n", BATCH_WIDTHS)
def test_routed_rows_not_divisible_by_n(n):
    """Routed-row count coprime-ish to N, and per-expert runs that never land on a
    multiple of N -- so every expert contributes real padding lanes."""
    tokens, top_k, num_experts = 101, 3, 7
    routed = tokens * top_k  # 303: not a multiple of 4, 8 or 16
    assert routed % n != 0
    bank = _make_bank(GGML_IQ3_XXS, num_experts, seed=11)
    x = _make_x(tokens, seed=12)
    ids = _uniform_ids(tokens, top_k, num_experts, seed=13)

    ref = _unbatched(x, bank, ids, top_k, tokens, GGML_IQ3_XXS)
    got = _batched(x, bank, ids, top_k, tokens, GGML_IQ3_XXS, n, num_experts)
    _sane(ref)
    _assert_identical(ref, got)


@pytest.mark.parametrize("n", BATCH_WIDTHS)
def test_skewed_expert_distribution(n):
    """Most experts get 0 rows, one gets ~90% of them.

    The pathological shape for the grouping: long runs of empty experts either
    side of one enormous run, so a padded-offset bug shows up as a misrouted
    expert rather than a mis-sized launch.
    """
    tokens, num_experts = 2048, 64
    bank = _make_bank(GGML_IQ2_XS, num_experts, seed=21)
    x = _make_x(tokens, seed=22)
    ids = _skewed_ids(tokens, TOP_K, num_experts, seed=23)

    counts = torch.bincount(ids.reshape(-1).long(), minlength=num_experts)
    assert (counts == 0).sum().item() >= num_experts - 5, "distribution is not skewed enough"

    ref = _unbatched(x, bank, ids, TOP_K, tokens, GGML_IQ2_XS)
    got = _batched(x, bank, ids, TOP_K, tokens, GGML_IQ2_XS, n, num_experts)
    _sane(ref)
    _assert_identical(ref, got)


@pytest.mark.parametrize("n", BATCH_WIDTHS)
def test_small_case(n):
    """Fewer routed rows than a single group at N = 16: the whole launch is one
    mostly-padding group."""
    tokens, top_k, num_experts = 3, 2, 4
    bank = _make_bank(GGML_IQ3_XXS, num_experts, seed=31)
    x = _make_x(tokens, seed=32)
    ids = _uniform_ids(tokens, top_k, num_experts, seed=33)

    ref = _unbatched(x, bank, ids, top_k, tokens, GGML_IQ3_XXS)
    got = _batched(x, bank, ids, top_k, tokens, GGML_IQ3_XXS, n, num_experts)
    _sane(ref)
    _assert_identical(ref, got)


@pytest.mark.parametrize("n", BATCH_WIDTHS)
def test_down_projection_shape(n):
    """The second GEMV of the layer runs with tokens = T * top_k and top_k = 1, so
    ``token = z / topk`` degenerates to the identity -- a different index path
    through the kernel than the top_k = 6 cases above, and the one the shared
    permutation has to serve unchanged."""
    routed, num_experts = 4096, 16
    bank = _make_bank(GGML_IQ2_XS, num_experts, seed=41)
    x = _make_x(routed, seed=42)
    ids = _uniform_ids(routed, 1, num_experts, seed=43)

    ref = _unbatched(x, bank, ids, 1, routed, GGML_IQ2_XS)
    got = _batched(x, bank, ids, 1, routed, GGML_IQ2_XS, n, num_experts)
    _sane(ref)
    _assert_identical(ref, got)


def test_group_count_past_the_grid_z_cap():
    """gridDim.z still caps at 65535 -- it just counts GROUPS now. At N = 4 that
    bites at 262140 routed rows, so the launcher's slicing loop has to keep
    working on the batched path too."""
    n, tokens, num_experts = 4, 44000, 16
    routed = tokens * TOP_K
    assert routed // n > 65535
    bank = _make_bank(GGML_IQ2_XS, num_experts, seed=51)
    x = _make_x(tokens, seed=52)
    ids = _uniform_ids(tokens, TOP_K, num_experts, seed=53)

    ref = _unbatched(x, bank, ids, TOP_K, tokens, GGML_IQ2_XS)
    got = _batched(x, bank, ids, TOP_K, tokens, GGML_IQ2_XS, n, num_experts)
    _sane(ref)
    _assert_identical(ref, got)


# --------------------------------------------------------------------------- #
# 2. the padding sentinel must never store
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", BATCH_WIDTHS)
def test_padding_lanes_never_store(n):
    """Poison a guard band around the output and prove nothing outside the real
    routed rows was touched.

    A dropped ``z[j] >= 0`` guard writes ``dst[-1 * nrows + row]``, i.e. the row
    immediately BEFORE the output -- which is why the output here is a view into
    the middle of a larger poisoned buffer rather than a fresh allocation. A plain
    ``torch::zeros`` output could not catch it: the corruption lands outside the
    tensor, in whatever the caching allocator happened to hand out next door.

    The forward guard catches the mirror-image bug (a group index running past the
    end of ``perm``'s real content).
    """
    tokens, num_experts = 512, 16
    routed = tokens * TOP_K
    guard = 64
    poison = -12345.0

    bank = _make_bank(GGML_IQ3_XXS, num_experts, seed=61)
    x = _make_x(tokens, seed=62)
    # Skewed routing maximises the number of padding lanes (many short runs).
    ids = _skewed_ids(tokens, TOP_K, num_experts, seed=63)

    big = torch.full((routed + 2 * guard, NROWS), poison, device="cuda", dtype=torch.bfloat16)
    dst = big[guard : guard + routed]
    assert dst.is_contiguous()

    got = _batched(x, bank, ids, TOP_K, tokens, GGML_IQ3_XXS, n, num_experts, out=dst)
    assert got.data_ptr() == dst.data_ptr(), "`out` was not written in place"

    pv = torch.tensor(poison, device="cuda", dtype=torch.bfloat16)
    before, after = big[:guard], big[guard + routed :]
    assert (before == pv).all(), (
        f"{(before != pv).any(dim=1).sum().item()} rows BEFORE the output were "
        "written -- a padding lane stored at a negative routed-row index"
    )
    assert (after == pv).all(), "rows AFTER the output were written"

    # And every real row was produced -- otherwise "nothing was corrupted" would
    # be trivially satisfied by a kernel that wrote nothing at all.
    assert not (dst == pv).all(dim=1).any(), "some routed rows were never written"
    ref = _unbatched(x, bank, ids, TOP_K, tokens, GGML_IQ3_XXS)
    _sane(ref)
    _assert_identical(ref, dst)


# --------------------------------------------------------------------------- #
# 3. the permutation itself
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", BATCH_WIDTHS)
def test_perm_groups_share_one_expert_and_cover_every_row(n):
    """The kernel reads the expert id from the first non-padding lane and applies
    it to all N. That is only sound because the host guarantees the grouping, so
    the guarantee gets its own test rather than being implied by the value checks.
    """
    from freetoken.moe.fused_q2_k_ud import _expert_group_perm

    tokens, num_experts = 777, 32
    routed = tokens * TOP_K
    ids = _skewed_ids(tokens, TOP_K, num_experts, seed=71)
    perm = _expert_group_perm(ids, num_experts, n)

    assert perm.dtype == torch.int32
    assert perm.numel() % n == 0
    # Fixed worst-case length: no host sync was needed to size it.
    assert perm.numel() >= routed
    assert perm.numel() <= routed + num_experts * (n - 1) + n

    flat = ids.reshape(-1).long()
    real = perm[perm >= 0].long()
    assert real.numel() == routed, "perm does not cover every routed row exactly once"
    assert torch.equal(real.sort().values, torch.arange(routed, device=real.device))

    # Every aligned group holds one expert (padding lanes excluded).
    groups = perm.reshape(-1, n)
    e = torch.where(groups >= 0, flat[groups.clamp(min=0)], torch.full_like(groups.long(), -1))
    lead = e.max(dim=1).values           # -1 for wholly-padding groups
    mixed = ((e != lead.unsqueeze(1)) & (e >= 0)).any(dim=1)
    assert not mixed.any(), f"{mixed.sum().item()} groups mix experts"

    # Padding is trailing within a group -- not required by the kernel, but it is
    # what the builder promises and what makes the `lead` lookup cheap.
    is_pad = groups < 0
    assert torch.equal(is_pad, is_pad.sort(dim=1, descending=False).values)


# --------------------------------------------------------------------------- #
# 4. end-to-end through the layer wrapper, both GEMVs on one permutation
# --------------------------------------------------------------------------- #


def test_fused_experts_prefill_matches_unbatched(monkeypatch):
    """``fused_experts_q2k_ud`` builds ONE permutation and hands it to both the
    gate_up GEMV (tokens = T, top_k = 6) and the down GEMV (tokens = T * 6,
    top_k = 1). If those two index spaces were not actually the same thing, this
    is where it would show.
    """
    from freetoken.moe import fused_q2_k_ud as mod

    tokens, num_experts, inter = 256, 16, 256
    gate_up = _make_bank(GGML_IQ2_XS, num_experts, nrows=2 * inter, seed=81)
    # down reads `inter` columns -> 1 super-block; pitch 80 >= 74, 80 % 74 != 0.
    g = torch.Generator().manual_seed(82)
    down = torch.randint(0, 256, (num_experts, H, 80), generator=g, dtype=torch.uint8)
    d = (0.01 + 0.04 * torch.rand(num_experts, H, generator=g)).to(torch.float16)
    down[:, :, :2] = d.view(torch.uint8).reshape(num_experts, H, 2)
    down = down.cuda()

    x = _make_x(tokens, seed=83)
    ids = _uniform_ids(tokens, TOP_K, num_experts, seed=84)
    w = torch.rand(tokens, TOP_K, device="cuda", dtype=torch.float32)

    def run(batch_n, is_prefill):
        monkeypatch.setattr(mod, "_PREFILL_BATCH", batch_n)
        out = mod.fused_experts_q2k_ud(
            x, gate_up, down, w, ids, GGML_IQ2_XS, GGML_IQ2_XS, 7.0, is_prefill=is_prefill
        )
        torch.cuda.synchronize()
        return out

    ref = run(0, True)          # knob off -> unbatched
    _sane(ref)
    for n in BATCH_WIDTHS:
        _assert_identical(ref, run(n, True))
    # Decode must not batch even with the knob set.
    _assert_identical(ref, run(8, False))
