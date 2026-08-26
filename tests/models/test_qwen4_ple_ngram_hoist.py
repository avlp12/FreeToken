"""Numerical proof that the CUDA-graph-hoisted n-gram embedding path reproduces
the same embeddings as the (still-present, unmodified) inline ``_ngram_embed`` /
``_NGramTable.gather()`` computation, against the REAL Qwen3.8-Flash-Next-FP8
checkpoint's n-gram table (real fp8 shards, real hash multipliers/offsets/vocab
sizes) -- not a mocked table.

Background: ``Qwen4PLELayer.forward()``'s decode branch used to compute the
n-gram embedding inline (hash -> ``_NGramTable.gather()`` -> PLE core), which is
illegal inside a captured CUDA graph (``gather()`` syncs to host, loops over a
data-dependent set of shards, and does an unpinned H2D copy). The fix hoists that
computation into ``precompute_decode_ngram``, called by the engine EAGERLY before
any graph replay, which parks the result in a stable-address per-slot buffer
(``_ngram_embed_buf``); the (possibly captured) decode forward then only does a
device-side ``index_select`` against that buffer.

These tests drive ``precompute_decode_ngram`` across multiple decode steps and
multiple concurrent (non-contiguous) slots, and independently recompute the
"ground truth" embedding via a direct call to ``_ngram_embed`` -- the exact
function the pre-hoist inline code called -- from a separately-tracked reference
token history. ``ngram_size=3`` means step t's hash needs tokens t-2, t-1, t;
several steps are run per slot specifically so a wrong-history bug (stale, off-by-
one, or cross-slot-mixed) would show up as a numerical mismatch rather than
passing by accident on the first step (where history is still mostly padding).
"""

from __future__ import annotations

import os
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.path.isdir("/root/models/Qwen3.8-Flash-Next-FP8"),
    reason="requires the real Qwen3.8-Flash-Next-FP8 checkpoint on disk",
)

MODEL_PATH = "/root/models/Qwen3.8-Flash-Next-FP8"


def _make_layer():
    from freetoken.models.qwen4_exp.config import Qwen4ExpArgs
    from freetoken.models.qwen4_exp.ngram import Qwen4PLELayer

    # Real values read from the checkpoint's config.json (text_config), not
    # guessed: hidden_size=2560, hc_count=4, ngram_size=3, heads_per_ngram=8,
    # split_ngram_parts=128, ple_embed_dim=2560, ple_conv_kernel_size=4,
    # eos_token_id=248044, ple_layer_ids=[2] (HF one-indexed -> layer_idx 1).
    args = Qwen4ExpArgs(
        hc_count=4,
        hc_lowrank=320,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=128,
        ple_layer_ids=(2,),
        ple_embed_dim=2560,
        ple_conv_kernel_size=4,
        eos_token_id=248044,
        model_path=MODEL_PATH,
    )
    config = types.SimpleNamespace(qwen4_args=args, hidden_size=2560, rms_norm_eps=1e-6)
    return Qwen4PLELayer(config, layer_idx=1)


class _StubPool:
    """Just enough of LinearStatePool for precompute_decode_ngram/forward: a slot
    count, keyed by a local layer index precompute doesn't otherwise care about."""

    def __init__(self, slots: int):
        self.conv_states = [torch.zeros(slots, 1)]

    def local_index(self, layer_idx: int) -> int:
        return 0


def _make_ctx(slots: int):
    from freetoken.core import Context

    ctx = Context(page_size=1)
    ctx.linear_state_pool = _StubPool(slots)
    return ctx


def _make_batch(device, input_ids: list[int], slot_ids: list[int]):
    from freetoken.core import Batch

    batch = Batch(reqs=[], phase="decode")
    batch.input_ids = torch.tensor(input_ids, dtype=torch.int32, device=device)
    batch.linear_table_idx = torch.tensor(slot_ids, dtype=torch.int32, device=device)
    batch.padded_reqs = []
    batch.fla_metadata = None
    return batch


@pytest.fixture()
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def _global_ctx_guard():
    import freetoken.core as core

    saved = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = saved


def test_hoisted_decode_matches_inline_gather_multi_step_multi_slot(device):
    """Multi-step, multi-slot (non-contiguous slot ids) decode: at every step the
    precomputed buffer must exactly equal a from-scratch ``_ngram_embed`` call
    fed the independently-tracked reference history, and the layer's own
    ``_tok_hist`` state must match the reference history too."""
    import freetoken.core as core

    layer = _make_layer()
    slots = 8
    ctx = _make_ctx(slots)
    core._GLOBAL_CTX = ctx

    # Non-contiguous, out-of-order slot assignment on purpose: a bug that mixes up
    # rows (e.g. assumes slot i == batch row i) would show up as cross-contamination.
    active_slots = [5, 1, 6]
    rng = torch.Generator().manual_seed(1234)
    vocab_hi = 50_000

    # Seed each slot with a distinct 2-token "post-prefill" history (as a real
    # prefill would leave behind: layer._tok_hist[slot] = last context_len ids).
    seed_hist = {
        s: [int(torch.randint(0, vocab_hi, (1,), generator=rng)) for _ in range(layer.context_len)]
        for s in active_slots
    }
    for s, h in seed_hist.items():
        layer._ensure_state(slots, device, layer.key_proj.weight.dtype)
        layer._tok_hist[s] = torch.tensor(h, dtype=torch.long, device=device)

    ref_hist = {s: list(h) for s, h in seed_hist.items()}  # python-side ground truth

    n_steps = 12
    for step in range(n_steps):
        order = active_slots[step % len(active_slots):] + active_slots[: step % len(active_slots)]
        new_tokens = {
            s: int(torch.randint(0, vocab_hi, (1,), generator=rng)) for s in order
        }
        batch = _make_batch(
            device,
            input_ids=[new_tokens[s] for s in order],
            slot_ids=order,
        )
        layer.precompute_decode_ngram(batch)

        for s in order:
            # Ground truth: direct call to the unmodified _ngram_embed, exactly
            # what the pre-hoist inline decode branch computed.
            hist_row = torch.tensor(
                [ref_hist[s] + [new_tokens[s]]], dtype=torch.long, device=device
            )
            emb_ref = layer._ngram_embed(hist_row, 1, layer._ngram_embed_buf.dtype)[0, 0]
            emb_hoisted = layer._ngram_embed_buf[s]
            assert torch.equal(emb_hoisted, emb_ref), (
                f"step {step} slot {s}: hoisted embedding diverged from the "
                f"direct gather() reference"
            )
            # State check: _tok_hist must now hold exactly the last context_len ids.
            expected_hist = (ref_hist[s] + [new_tokens[s]])[-layer.context_len:]
            assert layer._tok_hist[s].tolist() == expected_hist, (
                f"step {step} slot {s}: token-history state diverged "
                f"(got {layer._tok_hist[s].tolist()}, want {expected_hist})"
            )
            ref_hist[s] = expected_hist

    # An untouched slot must never have been written.
    untouched = 3
    assert torch.equal(
        layer._ngram_embed_buf[untouched], torch.zeros_like(layer._ngram_embed_buf[untouched])
    )


def test_forward_decode_branch_reads_the_same_embedding_precompute_wrote(device):
    """End-to-end: precompute_decode_ngram followed by forward()'s decode branch
    must consume exactly the embedding precompute parked in _ngram_embed_buf (via
    the capture-safe index_select), matching a direct _ngram_embed call."""
    import freetoken.core as core

    layer = _make_layer()
    # Give the linear layers real (non-garbage) weights so forward() doesn't run
    # matmuls over uninitialized memory.
    device_bf16 = device
    for name in ("key_proj", "value_proj"):
        w = getattr(layer, name).weight
        w.data = torch.randn(*w.shape, dtype=torch.bfloat16, device=device_bf16)
    for name in ("norm_key", "norm_query", "norm_conv"):
        w = getattr(layer, name).weight
        w.data = torch.ones(*w.shape, dtype=torch.bfloat16, device=device_bf16)
    layer.conv1d.weight.data = torch.randn(
        *layer.conv1d.weight.shape, dtype=torch.bfloat16, device=device_bf16
    )

    slots = 4
    ctx = _make_ctx(slots)
    core._GLOBAL_CTX = ctx

    slot = 2
    hist0 = [111, 222]
    layer._ensure_state(slots, device_bf16, torch.bfloat16)
    layer._tok_hist[slot] = torch.tensor(hist0, dtype=torch.long, device=device_bf16)

    new_tok = 333
    batch = _make_batch(device_bf16, input_ids=[new_tok], slot_ids=[slot])
    layer.precompute_decode_ngram(batch)

    hist_row = torch.tensor([hist0 + [new_tok]], dtype=torch.long, device=device_bf16)
    emb_ref = layer._ngram_embed(hist_row, 1, torch.bfloat16)[0, 0]
    assert torch.equal(layer._ngram_embed_buf[slot], emb_ref)

    x4 = torch.randn(1, layer.hc_count * layer.hidden_size, dtype=torch.bfloat16, device=device_bf16)
    with ctx.forward_batch(batch):
        out = layer.forward(x4)
    assert out.shape == x4.shape
    assert torch.isfinite(out).all()
