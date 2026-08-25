"""Grouped expert GEMM over native GGUF IQ-quant banks (DeepSeek-V4 UD-Q2_K_XL).

The ``q2_k_ud`` sibling of :mod:`freetoken.moe.fused_q4_0`: the experts are streamed
to the GPU as packed block bytes and dequantized *inside* ``ggml_moe_a8_vec`` -- no
bf16 expert copy is materialized. Two things differ from the Q4_0 wrapper:

* **Pitch.** ``q2_k_ud`` banks pad every row to one uniform IQ3_XXS width, so both
  GEMVs pass ``row_pitch_bytes``. It is read off the bank tensors rather than
  hardcoded, so a geometry change cannot silently desynchronize the two.
* **Per-layer quant type.** The banks hold rows of several ggml types (IQ2_XS,
  IQ3_XXS, Q2_K) at that shared pitch, so the caller supplies the type each of its
  layer's two banks decodes as instead of the wrapper naming a constant.

The activation is DeepSeek-V4's CLAMPED SwiGLU (``fused_swiglu(gate_up, limit)``),
not the generic ``silu_and_mul`` the Q4_0 path uses; ``gate_up`` is gate-major,
matching how the loader lays the bank out.

MMQ (``ggml_moe_a8``) is not an option here at any batch size: it is not
pitch-aware and rejects a nonzero ``row_pitch_bytes``. Prefill therefore runs the
same vector kernel -- but it runs the WEIGHT-REUSE BATCHED variant of it.

Prefill weight reuse
--------------------
``moe_vec_q`` gives one CUDA block to each (routed row, weight row) pair, and each
block streams its expert's weight row from HBM. At an 8192-token chunk with
top_k = 6 that is 49152 routed rows over 256 experts, so every expert's weight
matrix is re-read ~192 times per layer -- 20.4 TB per chunk, ~69% of the card's
HBM roofline, 89% of prefill wall time.

``ggml_moe_a8_vec_batched`` amortizes that: N routed rows that share an expert are
computed by one block, which reads the weight row once and feeds it to N different
activation rows. Each output is still the same dot product accumulated in the same
order, so the batched path is BIT-IDENTICAL to the unbatched one -- see
``csrc/gguf/moe_vec_batched.cuh`` and ``tests/kernels/test_moe_vec_batched.py``.

The N rows must share an expert, which is what ``_expert_group_perm`` below
arranges. That permutation is built ONCE per call and drives BOTH GEMVs: the
gate_up call runs with (tokens = T, top_k = 6) and the down call with
(tokens = T * 6, top_k = 1), but a routed-row index means the same thing in each,
so one grouping is valid for both.

Decode (T = 1) keeps taking the unbatched path untouched: there is nothing to
amortize when the handful of routed rows mostly hold distinct experts, and the
extra sort would be pure overhead on the latency-critical path.

``FREETOKEN_MOE_PREFILL_BATCH`` sets N (default 8). ``0`` or ``1`` restores the
unbatched prefill kernel for A/B comparison.

Prefill dequant-GEMM
--------------------
Batching fixed the HBM traffic but not the ALU work: one block still re-runs the
IQ2_XS codebook lookups once per routed row it serves, so each expert is still
DECODED ~24 times per layer. At a big enough chunk the answer is to stop decoding
per row entirely -- decode a tile of experts once into a bf16 scratch matrix and
run a grouped bf16 GEMM (:mod:`freetoken.moe.prefill_dequant_gemm`). That is the
default for chunks of at least ``FREETOKEN_PREFILL_DEQUANT_MIN_TOKENS`` tokens;
``FREETOKEN_PREFILL_DEQUANT_GEMM=0`` restores the batched GEMV above exactly.

It is the one path here that is NOT bit-identical to the unbatched kernel: the
activation stays bf16 instead of being quantized to q8_1, which removes error
rather than adding it. Decode is untouched -- ``is_prefill=False`` never reaches
either prefill branch.
"""

from __future__ import annotations

import os
import warnings

import torch

from freetoken.moe import prefill_dequant_gemm as _dq

# Batch widths the CUDA side instantiates (MOE_VEC_BATCH_WIDTHS in
# moe_vec_batched.cuh). 0/1 mean "no batching".
_BATCH_WIDTHS = (2, 4, 8, 16)
_DEFAULT_BATCH = 8


def _read_batch_env() -> int:
    raw = os.getenv("FREETOKEN_MOE_PREFILL_BATCH")
    if raw is None:
        return _DEFAULT_BATCH
    raw = raw.strip()
    try:
        n = int(raw)
    except ValueError:
        warnings.warn(
            f"FREETOKEN_MOE_PREFILL_BATCH={raw!r} is not an integer; "
            f"using the default {_DEFAULT_BATCH}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _DEFAULT_BATCH
    if n in (0, 1):
        return 0
    if n not in _BATCH_WIDTHS:
        warnings.warn(
            f"FREETOKEN_MOE_PREFILL_BATCH={n} is not one of {_BATCH_WIDTHS} "
            f"(0/1 disable batching); using the default {_DEFAULT_BATCH}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _DEFAULT_BATCH
    return n


# Read once at import, like ``offload_cache._FUSED_COPY``: the per-layer path must
# not pay an environment lookup, and the value must not shift mid-run underneath a
# captured CUDA graph.
_PREFILL_BATCH = _read_batch_env()


def _expert_group_perm(topk_ids: torch.Tensor, num_experts: int, n: int) -> torch.Tensor:
    """Routed-row indices grouped ``n``-per-expert, padded with -1.

    Returns an int32 tensor whose every aligned run of ``n`` entries belongs to a
    single expert. Values are indices into the flattened ``[tokens * top_k]``
    routed-row range; -1 marks alignment padding, which the kernel computes but
    never stores.

    The length is a FIXED worst-case bound -- ``routed + num_experts * (n - 1)``
    rounded up to a multiple of ``n`` -- not the exact padded total. The exact
    total is data-dependent, and reading it would mean a device->host sync in the
    per-layer path, the very stall class this change exists to remove. The surplus
    entries stay -1 and their blocks exit on the first instruction; at the
    production shape (routed 49152, 256 experts, n = 8) the slack is 3.6% of the
    groups. A fixed length is also what keeps the launch shape independent of the
    routing data.

    Everything here is vectorized and sync-free. Note ``scatter_add_`` rather than
    ``torch.bincount``: bincount has to learn the max element to size its output,
    which syncs.
    """
    dev = topk_ids.device
    flat = topk_ids.reshape(-1).long()
    routed = flat.numel()

    # Stable sort, so the grouping is deterministic run to run and a failure is
    # reproducible. `order` holds routed-row indices ordered by expert.
    sorted_experts, order = torch.sort(flat, stable=True)

    counts = torch.zeros(num_experts, dtype=torch.long, device=dev)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    padded_counts = ((counts + (n - 1)) // n) * n

    # Position i in `order` sits at `starts[e] + within` and must land at
    # `padded_starts[e] + within`. The `within` cancels, leaving a per-expert shift.
    starts = torch.cumsum(counts, 0) - counts
    padded_starts = torch.cumsum(padded_counts, 0) - padded_counts
    shift = padded_starts - starts

    bound = routed + num_experts * (n - 1)
    bound = -(-bound // n) * n
    perm = torch.full((bound,), -1, dtype=torch.int32, device=dev)
    dest = torch.arange(routed, device=dev) + shift[sorted_experts]
    perm[dest] = order.to(torch.int32)
    return perm


def _dequant_gemm_experts(
    hidden_states: torch.Tensor,  # [T, H]
    gate_up_q: torch.Tensor,
    down_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_up_qtype: int,
    down_qtype: int,
    swiglu_limit: float,
    fused_swiglu,
) -> torch.Tensor:
    """Prefill path: decode each expert once per chunk, then a grouped bf16 GEMM.

    The whole layer stays in EXPERT-SORTED routed-row order between the two GEMMs.
    The gate_up call needs the activations gathered (routed row r reads token
    ``r // top_k``); the down call does not, because ``inter`` is already indexed
    by sorted position and the down projection's activation for routed row r IS
    row r. So the permutation is paid once on the way in and once on the way out,
    and the two GEMMs chain directly.

    See :mod:`freetoken.moe.prefill_dequant_gemm` for the pitch slicing, the
    ``_grouped_mm`` layout, and the single host wait.
    """
    num_tokens = hidden_states.shape[0]
    h = down_q.shape[1]  # hidden
    top_k = topk_ids.shape[1]

    plan = _dq.RoutePlan(topk_ids, gate_up_q.shape[0], _dq.TILE)
    # Enqueue the gather BEFORE waiting on the plan's boundary transfer (inside
    # grouped_expert_gemm -> plan.tiles()), so the GPU is busy across the wait.
    a = hidden_states.index_select(0, plan.order // top_k if top_k > 1 else plan.order)

    gate_up = _dq.grouped_expert_gemm(a, gate_up_q, gate_up_qtype, plan)
    del a
    inter = fused_swiglu(gate_up, swiglu_limit)
    del gate_up
    out_sorted = _dq.grouped_expert_gemm(inter, down_q, down_qtype, plan)
    del inter

    # Unpermute. ``out_sorted[i]`` holds routed row ``plan.order[i]``, so this is
    # a SCATTER (``out[order[i]] = out_sorted[i]``), not a gather -- inverting it
    # silently returns a permuted-but-plausible answer.
    out = torch.empty_like(out_sorted)
    out.index_copy_(0, plan.order, out_sorted)
    del out_sorted
    out = out.reshape(num_tokens, top_k, h)
    out.mul_(topk_weights.reshape(num_tokens, top_k, 1).to(out.dtype))
    return out.sum(dim=1)


def fused_experts_q2k_ud(
    hidden_states: torch.Tensor,  # [T, H]
    gate_up_q: torch.Tensor,  # [num_slots, 2I, gate_up_pitch] uint8
    down_q: torch.Tensor,  # [num_slots, H, down_pitch] uint8
    topk_weights: torch.Tensor,  # [T, top_k]
    topk_ids: torch.Tensor,  # [T, top_k] int32 -> bank row
    gate_up_qtype: int,
    down_qtype: int,
    swiglu_limit: float,
    *,
    is_prefill: bool = False,
) -> torch.Tensor:
    """Routed-expert output summed over the top-k routes (excludes the shared expert).

    ``topk_ids`` already index the bank rows: cache slots on the decode path,
    materialized layer positions (position == expert id) on the streaming prefill path.

    ``is_prefill`` selects the prefill kernel -- the dequant-GEMM path when the
    chunk is big enough for it (see :mod:`freetoken.moe.prefill_dequant_gemm`),
    otherwise the weight-reuse batched GEMV, which is bit-identical to the
    unbatched one. The dequant-GEMM path is NOT bit-identical: it keeps the
    activation in bf16 instead of quantizing it to q8_1, which is a small
    accuracy gain, not a loss.
    """
    from freetoken.kernel.gguf import (
        ggml_moe_a8_vec,
        ggml_moe_a8_vec_batched,
        ggml_moe_vec_batched_supported,
    )
    from freetoken.kernel.triton.dsv4.fused_moe import fused_swiglu

    num_tokens = hidden_states.shape[0]
    n2 = gate_up_q.shape[1]  # 2 * intermediate
    h = down_q.shape[1]  # hidden
    top_k = topk_ids.shape[1]

    if is_prefill and _dq.ENABLED and num_tokens >= _dq.MIN_TOKENS:
        # Decode never reaches here, and neither does a short chunk: decoding all
        # 256 experts to serve a few hundred routed rows loses to the GEMV.
        i = n2 // 2
        if _dq.supported(int(gate_up_qtype), h, gate_up_q.shape[2]) and _dq.supported(
            int(down_qtype), i, down_q.shape[2]
        ):
            return _dequant_gemm_experts(
                hidden_states, gate_up_q, down_q, topk_weights, topk_ids,
                int(gate_up_qtype), int(down_qtype), swiglu_limit, fused_swiglu,
            )

    batch_n = _PREFILL_BATCH if is_prefill else 0
    if (
        batch_n
        and num_tokens * top_k >= batch_n
        and ggml_moe_vec_batched_supported(int(gate_up_qtype))
        and ggml_moe_vec_batched_supported(int(down_qtype))
    ):
        # One permutation, both GEMVs -- the routed-row index space is shared.
        perm = _expert_group_perm(topk_ids, gate_up_q.shape[0], batch_n)
    else:
        perm = None

    # gate_up: [T*top_k, 2I] -> clamped swiglu -> [T*top_k, I]
    if perm is None:
        gate_up = ggml_moe_a8_vec(
            hidden_states, gate_up_q, topk_ids, top_k, int(gate_up_qtype), n2, num_tokens,
            gate_up_q.shape[2],
        )
    else:
        gate_up = ggml_moe_a8_vec_batched(
            hidden_states, gate_up_q, topk_ids, perm, top_k, int(gate_up_qtype), n2,
            num_tokens, gate_up_q.shape[2], batch_n,
        )
    inter = fused_swiglu(gate_up, swiglu_limit)
    # down: each of the T*top_k intermediate rows uses its own expert id.
    if perm is None:
        out = ggml_moe_a8_vec(
            inter, down_q, topk_ids, 1, int(down_qtype), h, num_tokens * top_k,
            down_q.shape[2],
        )
    else:
        out = ggml_moe_a8_vec_batched(
            inter, down_q, topk_ids, perm, 1, int(down_qtype), h, num_tokens * top_k,
            down_q.shape[2], batch_n,
        )
    # In place, deliberately. ``out`` is a fresh kernel allocation that nothing
    # else aliases, and an out-of-place ``*`` would hold a SECOND
    # [T, top_k, H] tensor live alongside it -- 805 MB of avoidable transient peak
    # at a 16 K chunk, on a card where prefill already runs close enough to the
    # 32 GiB ceiling that the allocator starts thrashing. Same multiply, same
    # order, so the bit-identical equality tests are unaffected.
    out = out.reshape(num_tokens, top_k, h)
    out.mul_(topk_weights.reshape(num_tokens, top_k, 1).to(out.dtype))
    return out.sum(dim=1)


__all__ = ["fused_experts_q2k_ud"]
