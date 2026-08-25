"""Prefill dequant-GEMM: decode each expert once, then a grouped bf16 GEMM.

Unlike ``test_moe_vec_batched.py``, the bar here is NOT ``torch.equal``. This path
deliberately changes the arithmetic: the vec kernel quantizes the activation to
q8_1 (8 bits per 32-element block) to feed an integer dot product, while this one
dequantizes the WEIGHT to bf16 and leaves the activation alone. So the two answers
differ by roughly the vec path's activation quantization error, and this path is
the more accurate of the two.

Threshold. q8_1 stores each 32-element activation block as int8 against a
per-block scale, i.e. a uniform quantization error of about ``amax / 254`` per
element, ~0.23% of the block's amax RMS for gaussian data. Summed over a row of
K terms with independent signs those errors add in quadrature while the signal
adds coherently, so the relative L2 of the OUTPUT lands at the same few-tenths-of-
a-percent order -- measured 4-6e-3 across all three q2_k_ud types at both
production column counts. ``REL_L2_MAX = 2e-2`` is therefore ~4x the observed
value: loose enough that seed and shape wobble cannot flake it, tight enough that
a real defect (a misrouted expert, an inverted permutation, a truncated bank
slice) lands one to three ORDERS of magnitude above it. Those defects are what
``test_wrong_expert_is_far_outside_tolerance`` calibrates the bar against.

Geometry mirrors the shipped q2_k_ud banks: rows padded out to one uniform byte
pitch shared across quant types, so the pitch is not a whole number of blocks of
either type living in it.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

QK_K = 256
GGML_Q2_K = 10
GGML_IQ2_XS = 17
GGML_IQ3_XXS = 18
QTYPES = (GGML_IQ2_XS, GGML_IQ3_XXS, GGML_Q2_K)
QNAME = {GGML_Q2_K: "Q2_K", GGML_IQ2_XS: "IQ2_XS", GGML_IQ3_XXS: "IQ3_XXS"}
BLOCK_BYTES = {GGML_Q2_K: 84, GGML_IQ2_XS: 74, GGML_IQ3_XXS: 98}

# Q2_K puts its fp16 super-block scale pair at the END of the block; the IQ types
# put a single fp16 ``d`` at the head.
SCALE_OFF = {GGML_Q2_K: 80, GGML_IQ2_XS: 0, GGML_IQ3_XXS: 0}
SCALE_LEN = {GGML_Q2_K: 4, GGML_IQ2_XS: 2, GGML_IQ3_XXS: 2}

H = 1024        # hidden / gate_up ncols -- 4 super-blocks per row
I = 512         # intermediate / down ncols -- 2 super-blocks per row
PITCH_H = 400   # >= max native H row (4 * 98 = 392), 16 B multiple, not a block multiple
PITCH_I = 208   # >= max native I row (2 * 98 = 196), 16 B multiple, not a block multiple
NROWS = 64
TOP_K = 6

REL_L2_MAX = 2e-2


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def _make_bank(qtype, num_experts, ncols, pitch, nrows=NROWS, seed=0):
    """``[E, nrows, pitch]`` uint8 bank of ``qtype`` rows, tail explicitly ZERO.

    The tail matters here in a way it does not for the vec kernel: this path
    slices the native prefix off every row to hand ``ggml_dequantize`` a tightly
    packed bank, and asserts the discarded tail really is padding. Random bytes
    in the payload (the IQ grid/sign/scale nibbles are total over their lookup
    tables), explicit fp16 super-block scales so no random bit pattern can hand
    us a NaN and make the comparison vacuous.
    """
    blk = BLOCK_BYTES[qtype]
    nb = ncols // QK_K
    native = nb * blk
    assert native <= pitch and pitch % 16 == 0
    g = torch.Generator().manual_seed(seed)
    bank = torch.zeros(num_experts, nrows, pitch, dtype=torch.uint8)
    bank[:, :, :native] = torch.randint(
        0, 256, (num_experts, nrows, native), generator=g, dtype=torch.uint8
    )
    d = (0.01 + 0.04 * torch.rand(num_experts, nrows, nb, generator=g)).to(torch.float16)
    db = d.view(torch.uint8).reshape(num_experts, nrows, nb, 2)
    off, ln = SCALE_OFF[qtype], SCALE_LEN[qtype]
    for b in range(nb):
        lo = b * blk + off
        # Q2_K's ``dm`` is two fp16s (delta, min); write the same value into both
        # so the min term is bounded rather than a random pattern.
        bank[:, :, lo : lo + ln] = db[:, :, b].repeat(1, 1, ln // 2)
    return bank.cuda()


def _make_x(rows, ncols, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(rows, ncols, generator=g, device="cuda") * 0.5).to(torch.bfloat16)


def _uniform_ids(tokens, top_k, num_experts, seed=2):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(
        0, num_experts, (tokens, top_k), generator=g, device="cuda", dtype=torch.int32
    )


def _skewed_ids(tokens, top_k, num_experts, seed=3):
    """One expert takes ~90% of the routes; most experts get ZERO rows.

    Zero-count experts produce repeated entries in the cumulative-offset vector
    ``_grouped_mm`` consumes. The routed experts are also clustered into two
    narrow bands (``1..3`` and ``num_experts // 2``) so that whole expert TILES
    between and after them are empty and the plan has to drop them rather than
    launch a dequant for nothing -- which stays true for every tile size, unlike
    a spread that happens to touch one expert per tile.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    hot = num_experts // 2
    ids = torch.full((tokens, top_k), hot, device="cuda", dtype=torch.int32)
    r = torch.rand(tokens, top_k, generator=g, device="cuda")
    cold = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)
    pick = torch.randint(0, 3, (tokens, top_k), generator=g, device="cuda")
    return torch.where(r < 0.1, cold[pick], ids).contiguous()


def _rel_l2(ref: torch.Tensor, got: torch.Tensor) -> float:
    a = ref.to(torch.float32)
    b = got.to(torch.float32)
    return ((a - b).norm() / a.norm()).item()


def _sane(y: torch.Tensor) -> None:
    assert torch.isfinite(y).all(), "output is not finite"
    assert y.abs().sum().item() > 0, "output is entirely zero -- the test is vacuous"


# --------------------------------------------------------------------------- #
# runners
# --------------------------------------------------------------------------- #


def _vec(x, bank, ids, top_k, tokens, qtype, pitch, nrows=NROWS):
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    y = ggml_moe_a8_vec(x, bank, ids, top_k, qtype, nrows, tokens, pitch)
    torch.cuda.synchronize()
    return y


def _dequant(x, bank, ids, top_k, qtype, num_experts, tile=16):
    """One GEMM through the production plan + grouped GEMM, unpermuted here."""
    from freetoken.moe import prefill_dequant_gemm as dq

    plan = dq.RoutePlan(ids, num_experts, tile)
    idx = plan.order // top_k if top_k > 1 else plan.order
    a = x.index_select(0, idx)
    sorted_out = dq.grouped_expert_gemm(a, bank, qtype, plan)
    out = torch.empty_like(sorted_out)
    out.index_copy_(0, plan.order, sorted_out)
    torch.cuda.synchronize()
    return out


# --------------------------------------------------------------------------- #
# 0. the geometry the whole thing rests on
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("qtype", QTYPES)
def test_native_row_bytes_matches_the_block_structs(qtype):
    """``ggml_type_block_bytes`` must report ``sizeof(block_q_t)``, not the MMQ
    tile height ``ggml_moe_get_block_size`` returns (0 for every IQ type, 4 for
    Q2_K). Driving the bank slice off the wrong one truncates real payload into
    plausible-looking garbage, so the two are pinned apart here."""
    from freetoken.kernel.gguf import (
        ggml_moe_get_block_size,
        ggml_type_block_bytes,
        ggml_type_block_elems,
        ggml_type_row_bytes,
    )

    assert ggml_type_block_bytes(qtype) == BLOCK_BYTES[qtype]
    assert ggml_type_block_elems(qtype) == QK_K
    assert ggml_type_row_bytes(qtype, H) == (H // QK_K) * BLOCK_BYTES[qtype]
    assert ggml_type_row_bytes(qtype, I) == (I // QK_K) * BLOCK_BYTES[qtype]
    assert ggml_type_block_bytes(qtype) != ggml_moe_get_block_size(qtype)

    # And it agrees with the loader's own table, which is what actually LAID OUT
    # the bank (Q2_K rows are re-encoded at load, so it is absent there).
    from freetoken.models.gguf.dequant import BLOCK_SHAPE

    if qtype in BLOCK_SHAPE:
        assert BLOCK_SHAPE[qtype] == (ggml_type_block_elems(qtype), ggml_type_block_bytes(qtype))


def test_nonzero_pad_tail_is_rejected():
    """The slice discards the row tail. If the derived native width were too
    small we would be discarding real payload -- so the path proves the tail is
    dead before trusting the geometry, and that proof has to actually fire."""
    from freetoken.moe import prefill_dequant_gemm as dq

    bank = _make_bank(GGML_IQ2_XS, 4, H, PITCH_H, seed=101)
    native = dq.native_row_bytes(GGML_IQ2_XS, H)
    assert native < PITCH_H
    bank[0, 0, native] = 0xFF

    dq._PAD_CHECKED.clear()
    ids = _uniform_ids(8, 1, 4, seed=102)
    x = _make_x(8, H, seed=103)
    with pytest.raises(AssertionError, match="not padding"):
        _dequant(x, bank, ids, 1, GGML_IQ2_XS, 4)
    dq._PAD_CHECKED.clear()


# --------------------------------------------------------------------------- #
# 1. accuracy vs the vec path, per qtype
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("qtype", QTYPES)
@pytest.mark.parametrize("ncols,pitch", [(H, PITCH_H), (I, PITCH_I)])
def test_accuracy_vs_vec_path(qtype, ncols, pitch, capsys):
    """Both bank geometries of the layer (gate_up reads H columns, down reads I),
    all three shipped quant types."""
    tokens, num_experts = 1024, 32
    bank = _make_bank(qtype, num_experts, ncols, pitch, seed=200 + qtype)
    x = _make_x(tokens, ncols, seed=201)
    ids = _uniform_ids(tokens, TOP_K, num_experts, seed=202)

    ref = _vec(x, bank, ids, TOP_K, tokens, qtype, pitch)
    got = _dequant(x, bank, ids, TOP_K, qtype, num_experts)
    _sane(ref)
    _sane(got)
    assert ref.shape == got.shape == (tokens * TOP_K, NROWS)
    rel = _rel_l2(ref, got)
    with capsys.disabled():
        print(f"  rel-L2 {QNAME[qtype]:<8} ncols={ncols:<5} {rel:.4e}")
    assert rel < REL_L2_MAX, f"{QNAME[qtype]} ncols={ncols}: rel-L2 {rel:.4e}"


@pytest.mark.parametrize("qtype", QTYPES)
def test_down_projection_shape(qtype):
    """The layer's second GEMM: tokens = T * top_k, top_k = 1, so the routed row
    IS the activation row and the plan's ``order // top_k`` degenerates to the
    identity."""
    routed, num_experts = 4096, 32
    bank = _make_bank(qtype, num_experts, I, PITCH_I, seed=210 + qtype)
    x = _make_x(routed, I, seed=211)
    ids = _uniform_ids(routed, 1, num_experts, seed=212)

    ref = _vec(x, bank, ids, 1, routed, qtype, PITCH_I)
    got = _dequant(x, bank, ids, 1, qtype, num_experts)
    _sane(got)
    assert _rel_l2(ref, got) < REL_L2_MAX


# --------------------------------------------------------------------------- #
# 2. skew, empty experts, unaligned row counts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tile", [8, 16, 32])
def test_skewed_distribution_with_empty_experts(tile):
    """Most experts get zero rows -- repeated cumulative offsets for
    ``_grouped_mm`` and whole expert TILES with no rows at all, which the plan
    must drop rather than launch."""
    from freetoken.moe import prefill_dequant_gemm as dq

    tokens, num_experts = 2048, 128
    bank = _make_bank(GGML_IQ2_XS, num_experts, H, PITCH_H, seed=301)
    x = _make_x(tokens, H, seed=302)
    ids = _skewed_ids(tokens, TOP_K, num_experts, seed=303)

    counts = torch.bincount(ids.reshape(-1).long(), minlength=num_experts)
    assert (counts == 0).sum().item() >= num_experts - 5, "distribution is not skewed enough"

    plan = dq.RoutePlan(ids, num_experts, tile)
    n_tiles = -(-num_experts // tile)
    assert len(plan.tiles()) < n_tiles, "no tile was empty -- the case is not exercised"

    ref = _vec(x, bank, ids, TOP_K, tokens, GGML_IQ2_XS, PITCH_H)
    got = _dequant(x, bank, ids, TOP_K, GGML_IQ2_XS, num_experts, tile=tile)
    _sane(got)
    assert _rel_l2(ref, got) < REL_L2_MAX


@pytest.mark.parametrize("tile", [8, 16, 32])
def test_expert_count_not_divisible_by_tile(tile):
    """Expert count coprime to the tile, so the last tile is short, AND a routed
    row count that is not a multiple of anything."""
    tokens, top_k, num_experts = 101, 3, 37
    routed = tokens * top_k  # 303
    assert num_experts % tile != 0
    assert routed % tile != 0
    bank = _make_bank(GGML_IQ3_XXS, num_experts, H, PITCH_H, seed=311)
    x = _make_x(tokens, H, seed=312)
    ids = _uniform_ids(tokens, top_k, num_experts, seed=313)

    ref = _vec(x, bank, ids, top_k, tokens, GGML_IQ3_XXS, PITCH_H)
    got = _dequant(x, bank, ids, top_k, GGML_IQ3_XXS, num_experts, tile=tile)
    _sane(got)
    assert got.shape == (routed, NROWS)
    assert _rel_l2(ref, got) < REL_L2_MAX


def test_single_expert_takes_everything():
    """Degenerate extreme: one expert, every routed row, so one tile carries the
    whole GEMM and every other tile is empty."""
    tokens, num_experts = 512, 64
    bank = _make_bank(GGML_Q2_K, num_experts, H, PITCH_H, seed=321)
    x = _make_x(tokens, H, seed=322)
    ids = torch.full((tokens, TOP_K), 40, device="cuda", dtype=torch.int32)

    ref = _vec(x, bank, ids, TOP_K, tokens, GGML_Q2_K, PITCH_H)
    got = _dequant(x, bank, ids, TOP_K, GGML_Q2_K, num_experts)
    _sane(got)
    assert _rel_l2(ref, got) < REL_L2_MAX


# --------------------------------------------------------------------------- #
# 3. the unpermute direction
# --------------------------------------------------------------------------- #


def test_unpermute_is_a_scatter_not_a_gather():
    """The single easiest bug to ship here.

    ``sorted_out[i]`` holds routed row ``order[i]``, so the unpermute is
    ``out[order[i]] = sorted_out[i]`` -- ``index_copy_``. The inverted form,
    ``sorted_out.index_select(0, order)``, applies the permutation a second time
    instead of undoing it: same shape, same values, same norm, plausible-looking
    output. Only a comparison against a per-row reference catches it.

    Routing is arranged so ``order`` is far from an involution (an involution
    would make the two directions agree and the test vacuous), and the assertion
    is checked BOTH ways: the correct direction must pass, and the inverted one
    must fail by orders of magnitude.
    """
    tokens, num_experts, top_k = 512, 32, TOP_K
    qtype = GGML_IQ2_XS
    bank = _make_bank(qtype, num_experts, H, PITCH_H, seed=401)
    x = _make_x(tokens, H, seed=402)
    ids = _uniform_ids(tokens, top_k, num_experts, seed=403)

    from freetoken.moe import prefill_dequant_gemm as dq

    plan = dq.RoutePlan(ids, num_experts, 16)
    order = plan.order
    # ``order`` must not be its own inverse, or the mutation below is a no-op.
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    assert not torch.equal(inv, order), "order is an involution -- the test cannot distinguish"

    a = x.index_select(0, order // top_k)
    sorted_out = dq.grouped_expert_gemm(a, bank, qtype, plan)

    right = torch.empty_like(sorted_out)
    right.index_copy_(0, order, sorted_out)          # out[order[i]] = sorted[i]
    wrong = sorted_out.index_select(0, order)        # out[i] = sorted[order[i]]
    torch.cuda.synchronize()

    ref = _vec(x, bank, ids, top_k, tokens, qtype, PITCH_H)
    _sane(ref)
    rel_right = _rel_l2(ref, right)
    rel_wrong = _rel_l2(ref, wrong)
    assert rel_right < REL_L2_MAX, f"correct direction failed: {rel_right:.4e}"
    assert rel_wrong > 0.5, (
        f"the INVERTED unpermute only scores {rel_wrong:.4e} -- this test cannot "
        "distinguish scatter from gather and is worthless as written"
    )
    assert not torch.equal(right, wrong)


def test_wrong_expert_is_far_outside_tolerance():
    """Calibrates REL_L2_MAX: a single misrouted expert has to blow past it.

    A threshold nothing can fail is not a test. Perturbing the routing of ~3% of
    the rows must move the rel-L2 orders of magnitude above the bar, which is
    what makes ``< 2e-2`` evidence of correctness rather than of a loose bar.
    """
    tokens, num_experts = 512, 32
    qtype = GGML_IQ3_XXS
    bank = _make_bank(qtype, num_experts, H, PITCH_H, seed=411)
    x = _make_x(tokens, H, seed=412)
    ids = _uniform_ids(tokens, TOP_K, num_experts, seed=413)

    ref = _vec(x, bank, ids, TOP_K, tokens, qtype, PITCH_H)
    good = _dequant(x, bank, ids, TOP_K, qtype, num_experts)
    assert _rel_l2(ref, good) < REL_L2_MAX

    bad_ids = ids.clone()
    bad_ids[:16] = (bad_ids[:16] + 1) % num_experts
    bad = _dequant(x, bank, bad_ids, TOP_K, qtype, num_experts)
    rel_bad = _rel_l2(ref, bad)
    assert rel_bad > 10 * REL_L2_MAX, (
        f"misrouting 3% of the rows only moved rel-L2 to {rel_bad:.4e}; "
        f"REL_L2_MAX={REL_L2_MAX} is not discriminating anything"
    )


def test_plan_covers_every_routed_row_exactly_once():
    """``order`` is a permutation and its per-expert runs match the counts.

    Value tests catch a broken permutation only through the answer; this catches
    it in the object, which is what a failure report needs to point at.
    """
    from freetoken.moe import prefill_dequant_gemm as dq

    tokens, num_experts, tile = 777, 32, 16
    ids = _skewed_ids(tokens, TOP_K, num_experts, seed=421)
    routed = tokens * TOP_K
    plan = dq.RoutePlan(ids, num_experts, tile)

    assert plan.routed == routed
    assert torch.equal(
        plan.order.sort().values, torch.arange(routed, device=plan.order.device)
    )
    flat = ids.reshape(-1).long()
    # Sorted by expert, stably.
    e = flat[plan.order]
    assert torch.equal(e, e.sort(stable=True).values), "order does not group by expert"

    counts = torch.bincount(flat, minlength=num_experts)
    assert torch.equal(plan.cum_i32.long(), counts.cumsum(0))

    # Tile boundaries agree with the cumulative counts, and cover [0, routed).
    tiles = plan.tiles()
    assert tiles[0][2] == 0 or all(t[2] > 0 for t in tiles)
    assert tiles[-1][3] == routed
    prev_end = 0
    for e0, e1, r0, r1 in tiles:
        assert r0 >= prev_end
        assert r1 == int(counts[:e1].sum().item())
        assert r0 == int(counts[:e0].sum().item())
        prev_end = r1


# --------------------------------------------------------------------------- #
# 4. tile size must not change the answer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tile", [1, 4, 8, 16, 32, 64])
def test_tile_size_is_a_scheduling_knob_only(tile):
    """Tiling only decides how many experts are decoded at a time. Different tiles
    partition the SAME grouped GEMM differently, but each routed row still meets
    the same weight matrix in the same order, so the result is bit-identical
    across tile sizes -- ``torch.equal``, not ``allclose``."""
    tokens, num_experts = 512, 64
    qtype = GGML_IQ2_XS
    bank = _make_bank(qtype, num_experts, H, PITCH_H, seed=501)
    x = _make_x(tokens, H, seed=502)
    ids = _skewed_ids(tokens, TOP_K, num_experts, seed=503)

    ref = _dequant(x, bank, ids, TOP_K, qtype, num_experts, tile=64)
    got = _dequant(x, bank, ids, TOP_K, qtype, num_experts, tile=tile)
    _sane(got)
    assert torch.equal(ref, got), f"tile={tile} changed the arithmetic"


# --------------------------------------------------------------------------- #
# 5. the wrapper: flags, fallback, and decode
# --------------------------------------------------------------------------- #


def _layer_banks(num_experts, inter, seed):
    """A (gate_up, down) pair with the production shape relationship:
    gate_up is ``[E, 2I, pitch(H)]`` and down is ``[E, H, pitch(I)]``."""
    gate_up = _make_bank(GGML_IQ2_XS, num_experts, H, PITCH_H, nrows=2 * inter, seed=seed)
    down = _make_bank(GGML_IQ3_XXS, num_experts, I, PITCH_I, nrows=H, seed=seed + 1)
    return gate_up, down


def _run_layer(monkeypatch, x, gate_up, down, w, ids, *, is_prefill, **flags):
    from freetoken.moe import fused_q2_k_ud as mod
    from freetoken.moe import prefill_dequant_gemm as dq

    for k, v in flags.items():
        monkeypatch.setattr(dq, k, v)
    out = mod.fused_experts_q2k_ud(
        x, gate_up, down, w, ids, GGML_IQ2_XS, GGML_IQ3_XXS, 7.0, is_prefill=is_prefill
    )
    torch.cuda.synchronize()
    return out


def test_fused_experts_prefill_dequant_gemm(monkeypatch):
    """End to end through ``fused_experts_q2k_ud``: one plan, both GEMMs, the
    layer staying in sorted order between them, and the final unpermute."""
    tokens, num_experts, inter = 512, 32, I
    gate_up, down = _layer_banks(num_experts, inter, 601)
    x = _make_x(tokens, H, seed=603)
    ids = _uniform_ids(tokens, TOP_K, num_experts, seed=604)
    w = torch.rand(tokens, TOP_K, device="cuda", dtype=torch.float32)

    ref = _run_layer(monkeypatch, x, gate_up, down, w, ids, is_prefill=True, ENABLED=False)
    got = _run_layer(
        monkeypatch, x, gate_up, down, w, ids, is_prefill=True,
        ENABLED=True, MIN_TOKENS=0, TILE=16,
    )
    _sane(ref)
    _sane(got)
    assert ref.shape == got.shape == (tokens, H)
    assert _rel_l2(ref, got) < REL_L2_MAX


def test_min_tokens_fallback_is_byte_identical(monkeypatch):
    """Below MIN_TOKENS the wrapper must return to the vec path EXACTLY -- not
    approximately. A fallback that quietly rounds differently would make the
    crossover a numerics cliff instead of a scheduling one."""
    tokens, num_experts, inter = 256, 32, I
    gate_up, down = _layer_banks(num_experts, inter, 611)
    x = _make_x(tokens, H, seed=613)
    ids = _uniform_ids(tokens, TOP_K, num_experts, seed=614)
    w = torch.rand(tokens, TOP_K, device="cuda", dtype=torch.float32)

    ref = _run_layer(monkeypatch, x, gate_up, down, w, ids, is_prefill=True, ENABLED=False)
    # MIN_TOKENS above the chunk -> fallback fires.
    fell_back = _run_layer(
        monkeypatch, x, gate_up, down, w, ids, is_prefill=True,
        ENABLED=True, MIN_TOKENS=tokens + 1,
    )
    _sane(ref)
    assert torch.equal(ref, fell_back), "the MIN_TOKENS fallback is not the vec path"

    # ...and one token higher it does NOT fire, which is what makes the equality
    # above a statement about the threshold rather than about nothing.
    taken = _run_layer(
        monkeypatch, x, gate_up, down, w, ids, is_prefill=True,
        ENABLED=True, MIN_TOKENS=tokens,
    )
    assert not torch.equal(ref, taken), "MIN_TOKENS == tokens should take the GEMM path"
    assert _rel_l2(ref, taken) < REL_L2_MAX


def test_decode_is_untouched(monkeypatch):
    """``is_prefill=False`` must never reach the dequant-GEMM, whatever the flags
    say. Decode is CUDA-graph captured and bit-identical by contract."""
    from freetoken.moe import fused_q2_k_ud as mod

    tokens, num_experts, inter = 4, 32, I
    gate_up, down = _layer_banks(num_experts, inter, 621)
    x = _make_x(tokens, H, seed=623)
    ids = _uniform_ids(tokens, TOP_K, num_experts, seed=624)
    w = torch.rand(tokens, TOP_K, device="cuda", dtype=torch.float32)

    baseline = _run_layer(
        monkeypatch, x, gate_up, down, w, ids, is_prefill=False, ENABLED=False
    )
    forced = _run_layer(
        monkeypatch, x, gate_up, down, w, ids, is_prefill=False,
        ENABLED=True, MIN_TOKENS=0, TILE=8,
    )
    _sane(baseline)
    assert torch.equal(baseline, forced), "decode took the prefill dequant-GEMM path"

    # And a tripwire on the wrapper itself: the GEMM entry point is never called.
    from freetoken.moe import prefill_dequant_gemm as dq

    called = []
    monkeypatch.setattr(dq, "ENABLED", True)
    monkeypatch.setattr(dq, "MIN_TOKENS", 0)
    monkeypatch.setattr(
        dq, "grouped_expert_gemm", lambda *a, **k: called.append(1) or (_ for _ in ()).throw(
            AssertionError("decode reached grouped_expert_gemm")
        )
    )
    mod.fused_experts_q2k_ud(
        x, gate_up, down, w, ids, GGML_IQ2_XS, GGML_IQ3_XXS, 7.0, is_prefill=False
    )
    torch.cuda.synchronize()
    assert not called


def test_unsupported_geometry_falls_back(monkeypatch):
    """A bank whose native row does not fit its pitch cannot be sliced tight. The
    wrapper must notice and take the GEMV rather than assert in the layer path."""
    from freetoken.moe import prefill_dequant_gemm as dq

    assert not dq.supported(GGML_IQ3_XXS, H, 300)   # 4 * 98 = 392 > 300
    assert not dq.supported(999, H, PITCH_H)        # no dequant kernel for the type
    assert dq.supported(GGML_IQ3_XXS, H, PITCH_H)

    tokens, num_experts, inter = 256, 32, I
    gate_up, down = _layer_banks(num_experts, inter, 631)
    x = _make_x(tokens, H, seed=633)
    ids = _uniform_ids(tokens, TOP_K, num_experts, seed=634)
    w = torch.rand(tokens, TOP_K, device="cuda", dtype=torch.float32)

    ref = _run_layer(monkeypatch, x, gate_up, down, w, ids, is_prefill=True, ENABLED=False)
    # Claim a type with no dequant kernel for gate_up -> unsupported -> GEMV.
    from freetoken.moe import fused_q2_k_ud as mod

    monkeypatch.setattr(dq, "ENABLED", True)
    monkeypatch.setattr(dq, "MIN_TOKENS", 0)
    monkeypatch.setattr(dq, "supported", lambda *a: False)
    got = mod.fused_experts_q2k_ud(
        x, gate_up, down, w, ids, GGML_IQ2_XS, GGML_IQ3_XXS, 7.0, is_prefill=True
    )
    torch.cuda.synchronize()
    assert torch.equal(ref, got)
