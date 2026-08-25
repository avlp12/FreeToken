"""The fused decode index kernels must be BIT-IDENTICAL to the torch composition.

These are ADDRESSES. An off-by-one does not raise and does not produce a NaN -- it
silently points attention at another token's KV, or writes a compressed block over a
live row. So every derived tensor is compared with ``torch.equal`` (never
``allclose``) against the pre-fusion path, which is kept reachable as
``_reference_window_ctx`` / the ``ictx is None`` branches / ``FREETOKEN_UNFUSED_INDEX=1``.

The states cover the edges the addressing actually has:

* the ring's early-decode masking (``pos < window_size``: ring columns ``j > pos``
  have no token yet) and the exact wrap point ``pos == win - 1`` / ``pos == win``;
* 128-token page boundaries, where ``prev_window_slot`` lands on the PREVIOUS page
  and the compress-state carry has to follow it into a different ring block;
* compress-ratio boundaries -- and the ratios DIFFER PER LAYER (4 and 128 in
  DSV4-Flash), so ``pos % ratio``, the block-completion flag and the live block
  count are checked against both;
* ``full_loc == -1`` (the capture buffer's fill, and any position the window has
  slid past), which must floor to compressed row ``-1`` and translate to window slot
  ``-1`` -- C truncation would give ``0``, i.e. row 0 of another request's KV;
* positions at and next to ``max_seq_len - 1``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Batch, Context, Req, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.kvcache.dsv4_cost_model import dsv4_pool_sizes, ring_size_for_ratio
from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache
from freetoken.models.deepseek_v4.args import DeepseekV4Args

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused index kernels are CUDA")

P = 128
MRR = 2
MAX_SEQ = 1024
RATIOS = (0, 4, 128)
DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Every edge the addressing has, in one list (see the module docstring).
POSITIONS = [
    0, 1, 2, 3, 4, 5, 7, 8,            # early decode + ratio-4 block boundaries
    126, 127, 128, 129, 130,           # window wrap + page boundary
    252, 255, 256, 257,                # ratio-128 block boundary + page boundary
    383, 384, 511, 512,                # further page boundaries
    MAX_SEQ - 2, MAX_SEQ - 1,          # near max_seq_len
]


# --------------------------------------------------------------------------- #
# stack
# --------------------------------------------------------------------------- #
def _args(ratios=RATIOS):
    return DeepseekV4Args(
        max_batch_size=MRR + 1, dim=256, n_layers=len(ratios), n_heads=4,
        q_lora_rank=128, o_lora_rank=128, o_groups=2, moe_inter_dim=128,
        n_routed_experts=4, n_activated_experts=2, vocab_size=128,
        index_n_heads=2, index_head_dim=128, index_topk=8,
        compress_ratios=tuple(ratios), max_seq_len=MAX_SEQ,
        head_dim=512, rope_head_dim=64, window_size=P,
    )


def _stack(ratios=RATIOS):
    """A real pool + backend + ctx, with row 0's whole ``MAX_SEQ`` history bound."""
    args = _args(ratios)
    num_pages = MAX_SEQ // P + 1
    sizes = dsv4_pool_sizes(num_pages=num_pages + 1, args=args, swa_ratio=1.0, P=P)
    pool = DSV4PagedKVCache(sizes=sizes, args=args, device=DEV, P=P, n_scratch=MRR + 1)
    pool._init_paged_state(MRR, True)
    pt = torch.zeros(MRR + 1, MAX_SEQ, dtype=torch.int32, device=DEV)
    pt[MRR].fill_(num_pages * P)  # engine dummy-row convention
    pt[0, :MAX_SEQ] = torch.arange(MAX_SEQ, dtype=torch.int32, device=DEV)
    for pg in range(MAX_SEQ // P):
        pool.bind_window_pages(pg * P, pg * P)
    pool.full_loc_map = pt

    try:
        ctx = get_global_ctx()
    except AssertionError:
        ctx = Context(page_size=P)
        set_global_ctx(ctx)
    ctx.kv_cache = pool

    from freetoken.attention.dsv4_sparse import DSV4SparseAttnBackend

    backend = DSV4SparseAttnBackend(SimpleNamespace(dsv4_args=args))
    ctx.attn_backend = backend
    return backend, pool, pt, args


def _decode_batch(rows, positions):
    reqs = [
        Req(input_ids=torch.zeros(1, dtype=torch.int32), table_idx=int(t), cached_len=0,
            output_len=1, uid=i, sampling_params=SamplingParams(), cache_handle=None)
        for i, t in enumerate(rows)
    ]
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = reqs
    batch.active_table_idx = torch.tensor(rows, dtype=torch.int64, device=DEV)
    batch.positions = torch.tensor(positions, dtype=torch.int64, device=DEV)
    return batch


def _snapshots(pool, extra_minus_one=True):
    """Synthetic whole-history snapshots, one per case, including a ``-1`` row.

    ``-1`` is the capture buffer's fill and the sentinel for a position the window
    has slid past; both the window translate and the compressed-row floor-div must
    carry it through as ``-1``.
    """
    ident = torch.arange(MAX_SEQ, dtype=torch.int64, device=DEV)
    cases = {
        "identity": ident.clone(),
        "offset": ident + P,
        "dummy": torch.full((MAX_SEQ,), (MAX_SEQ // P) * P, dtype=torch.int64, device=DEV),
    }
    if extra_minus_one:
        holed = ident.clone()
        holed[: 2 * P] = -1  # the window slid past the first two pages
        holed[900:] = -1     # never staged
        cases["minus_one"] = holed
    return cases


# --------------------------------------------------------------------------- #
# 1. window ring context
# --------------------------------------------------------------------------- #
def _ref_window_ctx(pos, rows, snap, f2w, win):
    """Verbatim pre-fusion composition (mirrors ``_reference_window_ctx``)."""
    def translate(x):
        return f2w[x.to(torch.int64)]

    bs = pos.shape[0]
    j = torch.arange(win, device=pos.device)
    window_slots = translate(snap[rows, pos])
    prev_window_slots = translate(snap[rows, (pos - 1).clamp_min(0)])
    p = pos[:, None] - ((pos[:, None] - j[None, :]) % win)
    ws = translate(snap[rows[:, None], p.clamp(min=0)])
    ring = torch.where(p >= 0, ws, torch.full_like(ws, -1))
    topk = torch.where(
        j[None, :] <= pos[:, None], ring, torch.full_like(ring, -1)
    ).view(bs, 1, win)
    return window_slots, prev_window_slots, topk


def test_window_ring_ctx_is_bit_identical():
    from freetoken.kernel.triton.dsv4.decode_index import window_ring_ctx

    _, pool, _, _ = _stack()
    f2w = pool.full_to_window
    for name, hist in _snapshots(pool).items():
        snap = torch.stack([hist, hist.roll(P)])
        rows = torch.arange(2, device=DEV)
        for p in POSITIONS:
            pos = torch.tensor([p, max(0, p - 1)], dtype=torch.int64, device=DEV)
            got = window_ring_ctx(pos, rows, snap, f2w, P)
            want = _ref_window_ctx(pos, rows, snap, f2w, P)
            for g, w, lbl in zip(got, want, ("window_slots", "prev_window_slots", "topk")):
                assert torch.equal(g, w), f"{lbl} mismatch: snap={name} pos={p}"


def test_window_ring_ctx_masks_positions_with_no_token_yet():
    """``pos < win`` leaves ring columns ``j > pos`` with no token: they MUST be -1,
    not the stale slot the modulo would otherwise resolve to."""
    from freetoken.kernel.triton.dsv4.decode_index import window_ring_ctx

    _, pool, _, _ = _stack()
    snap = torch.arange(MAX_SEQ, dtype=torch.int64, device=DEV).unsqueeze(0)
    rows = torch.zeros(1, dtype=torch.int64, device=DEV)
    for p in (0, 1, 5, P - 2, P - 1, P):
        pos = torch.tensor([p], dtype=torch.int64, device=DEV)
        _, _, topk = window_ring_ctx(pos, rows, snap, pool.full_to_window, P)
        assert int((topk[0, 0] >= 0).sum()) == min(p + 1, P), f"pos={p}"


# --------------------------------------------------------------------------- #
# 2. per-ratio decode index context
# --------------------------------------------------------------------------- #
def _ref_index_ctx(pos, rows, ws, pws, snap, ratio, ring, Pw, cap, cmp_base, idx_base):
    """The per-layer torch cluster the fused kernel replaces, op for op."""
    ar = torch.arange(ring, device=pos.device)
    valid = (pos + 1) // ratio
    ref = {
        "idx_mod": pos % ratio,
        "should": ((pos + 1) % ratio == 0),
        "carry_prev": torch.div(pws, Pw, rounding_mode="floor")[:, None] * ring + ar,
        "carry_cur": torch.div(ws, Pw, rounding_mode="floor")[:, None] * ring + ar,
        "freq_idx": (pos + 1 - ratio).clamp_min(0),
        "valid": valid,
        "cmp_counts": valid.clamp(max=cap).to(torch.int32).view(-1, 1),
    }
    row_of_block = torch.div(snap[rows, pos], ratio, rounding_mode="floor")
    completed = ref["should"]
    ref["cmp_dst_attn"] = torch.where(completed, row_of_block, rows + cmp_base)
    if idx_base is not None:
        ref["cmp_dst_idx"] = torch.where(completed, row_of_block, rows + idx_base)
    if ratio == 4:
        ref["ovl_slot"] = ratio + ref["idx_mod"]
    return ref


@pytest.mark.parametrize("ratio", [4, 128])
def test_decode_index_ctx_is_bit_identical(ratio):
    from freetoken.kernel.triton.dsv4.decode_index import decode_index_ctx, window_ring_ctx

    _, pool, _, _ = _stack()
    ring = ring_size_for_ratio(ratio)
    L = RATIOS.index(ratio)
    cmp_base = pool.cmp_scratch_base[L]
    idx_base = pool.idx_scratch_base[L] if ratio == 4 else None
    cap = 8 if ratio == 4 else MAX_SEQ // ratio
    for name, hist in _snapshots(pool).items():
        snap = torch.stack([hist, hist.roll(P)])
        rows = torch.arange(2, device=DEV)
        for p in POSITIONS:
            pos = torch.tensor([p, max(0, p - 1)], dtype=torch.int64, device=DEV)
            ws, pws, _ = window_ring_ctx(pos, rows, snap, pool.full_to_window, P)
            got = decode_index_ctx(
                pos, rows, ws, pws, snap, ratio=ratio, ring_size=ring, P=P, cap=cap,
                cmp_base=cmp_base, idx_base=idx_base, overlap=ratio == 4,
            )
            want = _ref_index_ctx(
                pos, rows, ws, pws, snap, ratio, ring, P, cap, cmp_base, idx_base
            )
            for k, w in want.items():
                g = getattr(got, k) if k != "should" else got.should
                assert torch.equal(g, w), f"{k} mismatch: ratio={ratio} snap={name} pos={p}"


def test_minus_one_full_loc_floors_to_row_minus_one():
    """``floor(-1 / ratio) == -1``. C truncation gives 0, which is a LIVE row."""
    from freetoken.kernel.triton.dsv4.decode_index import decode_index_ctx

    _, pool, _, _ = _stack()
    snap = torch.full((1, MAX_SEQ), -1, dtype=torch.int64, device=DEV)
    rows = torch.zeros(1, dtype=torch.int64, device=DEV)
    for ratio in (4, 128):
        pos = torch.tensor([ratio - 1], dtype=torch.int64, device=DEV)  # a completing block
        ws = torch.zeros(1, dtype=torch.int64, device=DEV)
        ctx = decode_index_ctx(
            pos, rows, ws, ws, snap, ratio=ratio, ring_size=ring_size_for_ratio(ratio),
            P=P, cap=4, cmp_base=99, idx_base=None, overlap=ratio == 4,
        )
        assert bool(ctx.should[0]) and int(ctx.cmp_dst_attn[0]) == -1


# --------------------------------------------------------------------------- #
# 3. compressed picks -> global rows
# --------------------------------------------------------------------------- #
def _ref_topk_idxs(picks, valid, rows, snap, wtopk, ratio, offset, identity_k):
    """``indexer_select_decode`` -> ``blocks_to_global`` -> ``cat`` -> ``int()``."""
    if picks is None:
        blk = torch.arange(identity_k, device=wtopk.device)
        blocks = torch.where(
            blk[None, :] < valid[:, None], blk[None, :], -1
        ).view(wtopk.shape[0], 1, identity_k)
    else:
        blocks = torch.where(picks >= valid[:, None, None], -1, picks + offset)
    safe = blocks.clamp_min(0)
    full_at = snap[rows[:, None, None], safe * ratio]
    g = torch.div(full_at, ratio, rounding_mode="floor")
    cmp = torch.where(blocks < 0, torch.full_like(g, -1), g)
    return torch.cat([wtopk, cmp], dim=-1).int()


@pytest.mark.parametrize("ratio", [4, 128])
def test_cmp_topk_to_global_is_bit_identical(ratio):
    from freetoken.kernel.triton.dsv4.decode_index import cmp_topk_to_global, window_ring_ctx

    _, pool, _, _ = _stack()
    gen = torch.Generator(device="cpu").manual_seed(7)
    for name, hist in _snapshots(pool).items():
        snap = torch.stack([hist, hist.roll(P)])
        rows = torch.arange(2, device=DEV)
        for p in POSITIONS:
            pos = torch.tensor([p, max(0, p - 1)], dtype=torch.int64, device=DEV)
            _, _, wtopk = window_ring_ctx(pos, rows, snap, pool.full_to_window, P)
            n_stage = (p + 1) // ratio
            valid = (pos + 1) // ratio
            # positional selection (the indexer-less ratio class)
            got = cmp_topk_to_global(
                None, valid, rows, snap, wtopk, ratio=ratio, identity_k=n_stage
            )
            want = _ref_topk_idxs(None, valid, rows, snap, wtopk, ratio, 0, n_stage)
            assert torch.equal(got, want), f"identity picks: ratio={ratio} {name} pos={p}"
            if n_stage == 0:
                continue
            # indexer picks: random, plus the exact "past the live count" boundary
            k = min(8, n_stage)
            picks = torch.randint(0, n_stage, (2, 1, k), generator=gen).to(DEV)
            picks[:, 0, 0] = n_stage - 1
            picks[0, 0, -1] = max(0, int(valid[0]) - 1)
            picks[1, 0, -1] = min(n_stage - 1, int(valid[1]))  # first DEAD block
            got = cmp_topk_to_global(picks, valid, rows, snap, wtopk, ratio=ratio)
            want = _ref_topk_idxs(picks, valid, rows, snap, wtopk, ratio, 0, None)
            assert torch.equal(got, want), f"indexer picks: ratio={ratio} {name} pos={p}"


# --------------------------------------------------------------------------- #
# 4. end to end: a real Attention.decode_step, fused vs unfused
# --------------------------------------------------------------------------- #
def _make_attention(layer_id, args, ratio, pool):
    from freetoken.models.deepseek_v4.attention import Attention

    attn = Attention(layer_id, args, compress_ratio=ratio).to(DEV)
    g = torch.Generator(device="cpu").manual_seed(1234 + layer_id)
    for p in attn.parameters():
        if p.dtype in (torch.float32, torch.bfloat16):
            p.data.copy_(torch.randn(p.shape, generator=g).to(p.dtype) * 0.02)
        elif p.dtype == torch.float8_e4m3fn:
            p.data.copy_(torch.randn(p.shape, generator=g).to(torch.float8_e4m3fn))
        elif p.dtype == torch.float8_e8m0fnu:
            p.data.copy_(torch.ones(p.shape).to(torch.float8_e8m0fnu))
    attn.attn_sink.data.zero_()
    attn.bind(pool, DEV)
    return attn


def _pool_state(pool, L):
    out = [pool.window_pool[L].clone()]
    for lst in (pool.cmp_pool, pool.idx_pool):
        if lst[L] is not None:
            out.append(lst[L].clone())
    for lst in (pool.state_ring, pool.indexer_state_ring):
        if lst[L] is not None:
            out.append(lst[L].buffer.clone())
    return out


def _restore(pool, L, state):
    it = iter(state)
    pool.window_pool[L].copy_(next(it))
    for lst in (pool.cmp_pool, pool.idx_pool):
        if lst[L] is not None:
            lst[L].copy_(next(it))
    for lst in (pool.state_ring, pool.indexer_state_ring):
        if lst[L] is not None:
            lst[L].buffer.copy_(next(it))


@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("p", [0, 3, 4, 127, 128, 129, 255, 256, 511, MAX_SEQ - 1])
def test_decode_step_end_to_end_matches_the_unfused_path(ratio, p):
    """Same output AND same pool side effects (window KV, compressed KV, indexer KV,
    both compress-state rings) -- an address that drifts shows up in one or the other."""
    backend, pool, _, args = _stack()
    L = RATIOS.index(ratio)
    attn = _make_attention(L, args, ratio, pool)
    batch = _decode_batch([0, MRR], [p, min(p, 63)])
    backend.prepare_metadata(batch)
    with get_global_ctx().forward_batch(batch):
        pos = batch.positions
        rows = torch.arange(2, device=DEV)
        md = batch.attn_metadata
        md.full_snapshot()
        torch.manual_seed(11)
        x = torch.randn(2, 1, args.dim, device=DEV, dtype=torch.bfloat16)
        cap = MAX_SEQ - 1
        clean = _pool_state(pool, L)

        wctx_ref = md._reference_window_ctx(pos, rows)
        out_ref = attn.decode_step(x, pos, rows, cap, wctx_ref, None)
        state_ref = _pool_state(pool, L)

        _restore(pool, L, clean)
        wctx = md.window_ctx(pos, rows)
        ictx = backend.index_ctx(pos, rows, wctx, cap)
        out_fused = attn.decode_step(x, pos, rows, cap, wctx, ictx)

        for a, b, lbl in zip(wctx, wctx_ref, ("ws", "prev_ws", "topk")):
            assert torch.equal(a, b), f"{lbl} drifted (ratio={ratio}, pos={p})"
        assert torch.equal(out_fused, out_ref), f"attention output (ratio={ratio}, pos={p})"
        for i, (a, b) in enumerate(zip(_pool_state(pool, L), state_ref)):
            assert torch.equal(a, b), f"pool buffer {i} (ratio={ratio}, pos={p})"


@pytest.mark.parametrize("ratio", [4, 128])
def test_fused_decode_step_captures_and_replays_in_a_cuda_graph(ratio):
    """The fusion must stay INSIDE the captured graph.

    A replay reads the staged snapshot and the live ``positions`` buffer, so the
    fused index kernels have to be graph nodes -- fixed shapes, no host branch on
    device data, no sync. If any of them were hoisted out (or cached), a replay would
    freeze at the capture-time ring slots and every later token would attend to the
    wrong window page. Proved by replaying at a DIFFERENT position and demanding the
    eager result for that position.
    """
    backend, pool, _, args = _stack()
    L = RATIOS.index(ratio)
    attn = _make_attention(L, args, ratio, pool)
    backend.init_capture_graph(max_seq_len=MAX_SEQ, bs_list=[2])
    batch = _decode_batch([0, MRR], [300, 12])
    backend.prepare_for_replay(batch)
    with get_global_ctx().forward_batch(batch):
        md = batch.attn_metadata
        cap = md.stage_width - 1
        rows = torch.arange(2, device=DEV)
        pos_buf = torch.tensor([300, 12], dtype=torch.int64, device=DEV)
        torch.manual_seed(5)
        x_buf = torch.randn(2, 1, args.dim, device=DEV, dtype=torch.bfloat16)

        def body():
            wctx = md.window_ctx(pos_buf, rows)
            ictx = backend.index_ctx(pos_buf, rows, wctx, cap)
            return attn.decode_step(x_buf, pos_buf, rows, cap, wctx, ictx)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(side)

        clean = _pool_state(pool, L)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = body()

        # replay at a position the capture never saw
        pos_buf.copy_(torch.tensor([431, 128], dtype=torch.int64, device=DEV))
        _restore(pool, L, clean)
        graph.replay()
        torch.cuda.synchronize()
        replayed, state_replay = out.clone(), _pool_state(pool, L)

        _restore(pool, L, clean)
        eager = body()

        assert torch.equal(replayed, eager), "replay diverged from eager at the new position"
        for i, (a, b) in enumerate(zip(state_replay, _pool_state(pool, L))):
            assert torch.equal(a, b), f"pool buffer {i} diverged on replay"


def test_unfused_env_gate_disables_the_fused_path(monkeypatch):
    """``FREETOKEN_UNFUSED_INDEX=1`` must reach the reference composition, so an
    addressing regression can be bisected without reverting the fusion."""
    from freetoken.attention import dsv4_sparse

    backend, pool, _, args = _stack()
    monkeypatch.setenv("FREETOKEN_UNFUSED_INDEX", "1")
    assert dsv4_sparse.unfused_index()
    batch = _decode_batch([0, MRR], [300, 12])
    backend.prepare_metadata(batch)
    with get_global_ctx().forward_batch(batch):
        pos, rows = batch.positions, torch.arange(2, device=DEV)
        md = batch.attn_metadata
        assert backend.index_ctx(pos, rows, md.window_ctx(pos, rows), MAX_SEQ - 1) == {}
        monkeypatch.delenv("FREETOKEN_UNFUSED_INDEX")
        assert torch.equal(md.window_ctx(pos, rows)[2], md._reference_window_ctx(pos, rows)[2])
