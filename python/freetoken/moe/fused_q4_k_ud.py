"""Grouped expert GEMM over native GGUF unsloth UD-Q4_K_XL banks (Qwen4-Exp /
Qwen3.8-Flash-Next).

The ``q4_k_ud`` sibling of :mod:`freetoken.moe.fused_q2_k_ud`, generalized for
Qwen4-Exp's mixed Q4_K/Q5_K gate_up and Q5_1/Q8_0 down banks (see
:mod:`freetoken.models.qwen4_exp.gguf_experts`). Same idea as the DSV4 module:
the experts are streamed to the GPU as packed block bytes and dequantized
*inside* ``ggml_moe_a8_vec`` -- no bf16 expert copy is materialized -- and every
row is padded to one uniform per-bank pitch, addressed via ``row_pitch_bytes``.

Two differences from ``fused_q2_k_ud``:

* **Activation.** Qwen4-Exp's routed experts use plain (unclamped) SwiGLU --
  ``silu_and_mul`` over uninterleaved gate/up halves, the same activation the
  shared expert and every other non-DSV4 GGUF format in this repo use (see
  :mod:`freetoken.moe.fused_q4_0`) -- not DeepSeek-V4's clamped variant, so
  there is no ``swiglu_limit`` here.
* **No weight-reuse prefill batching.** ``ggml_moe_a8_vec_batched`` only has
  kernels instantiated for Q2_K/IQ2_XS/IQ3_XXS/MXFP4 (the DSV4 ``q2_k_ud``
  types); Q4_K/Q5_K/Q5_1/Q8_0 have no batched instantiation, so
  ``ggml_moe_vec_batched_supported`` always reports False for every type this
  format uses and the batched path would be dead code. The prefill
  dequant-GEMM path (:mod:`freetoken.moe.prefill_dequant_gemm`) IS available
  unchanged -- ``dequantize.cuh``'s ``ggml_get_to_cuda`` already covers all
  four types -- and is tried first for large chunks exactly as it is for
  ``q2_k_ud``.

MMQ (``ggml_moe_a8``) is not an option here at any batch size, same reason as
``q2_k_ud``: it is not pitch-aware and rejects a nonzero ``row_pitch_bytes``.
"""

from __future__ import annotations

import torch

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from freetoken.moe import prefill_dequant_gemm as _dq

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}


def _dequant_gemm_experts(
    hidden_states: torch.Tensor,  # [T, H]
    gate_up_q: torch.Tensor,
    down_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_up_qtype: int,
    down_qtype: int,
    act_fn,
) -> torch.Tensor:
    """Prefill path: decode each expert once per chunk, then a grouped bf16 GEMM.

    Structurally identical to ``fused_q2_k_ud._dequant_gemm_experts`` (see that
    docstring for the sorted-order bookkeeping); the only difference is a plain
    ``act_fn(gate_up)`` call instead of a clamped swiglu.
    """
    num_tokens = hidden_states.shape[0]
    h = down_q.shape[1]  # hidden
    top_k = topk_ids.shape[1]

    plan = _dq.RoutePlan(topk_ids, gate_up_q.shape[0], _dq.TILE)
    a = hidden_states.index_select(0, plan.order // top_k if top_k > 1 else plan.order)

    gate_up = _dq.grouped_expert_gemm(a, gate_up_q, gate_up_qtype, plan)
    del a
    inter = act_fn(gate_up)
    del gate_up
    out_sorted = _dq.grouped_expert_gemm(inter, down_q, down_qtype, plan)
    del inter

    # Unpermute: out_sorted[i] holds routed row plan.order[i] -> a SCATTER.
    out = torch.empty_like(out_sorted)
    out.index_copy_(0, plan.order, out_sorted)
    del out_sorted
    out = out.reshape(num_tokens, top_k, h)
    out.mul_(topk_weights.reshape(num_tokens, top_k, 1).to(out.dtype))
    return out.sum(dim=1)


def fused_experts_q4k_ud(
    hidden_states: torch.Tensor,  # [T, H]
    gate_up_q: torch.Tensor,  # [num_slots, 2I, gate_up_pitch] uint8
    down_q: torch.Tensor,  # [num_slots, H, down_pitch] uint8
    topk_weights: torch.Tensor,  # [T, top_k]
    topk_ids: torch.Tensor,  # [T, top_k] int32 -> bank row
    gate_up_qtype: int,
    down_qtype: int,
    activation: str,
    *,
    is_prefill: bool = False,
) -> torch.Tensor:
    """Routed-expert output summed over the top-k routes.

    ``topk_ids`` already index the bank rows: cache slots on the decode path,
    materialized layer positions (position == expert id) on the streaming
    prefill path.

    ``is_prefill`` tries the dequant-GEMM path first when the chunk is big
    enough for it (see :mod:`freetoken.moe.prefill_dequant_gemm`), else the
    plain (unbatched) GEMV -- identical structure to ``fused_q2_k_ud``, minus
    the weight-reuse batching branch (no batched kernel exists for these
    types; see the module docstring).
    """
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")

    num_tokens = hidden_states.shape[0]
    n2 = gate_up_q.shape[1]  # 2 * intermediate
    h = down_q.shape[1]  # hidden
    top_k = topk_ids.shape[1]

    if is_prefill and _dq.ENABLED and num_tokens >= _dq.MIN_TOKENS:
        i = n2 // 2
        if _dq.supported(int(gate_up_qtype), h, gate_up_q.shape[2]) and _dq.supported(
            int(down_qtype), i, down_q.shape[2]
        ):
            return _dequant_gemm_experts(
                hidden_states, gate_up_q, down_q, topk_weights, topk_ids,
                int(gate_up_qtype), int(down_qtype), act_fn,
            )

    # gate_up: [T*top_k, 2I] -> activation -> [T*top_k, I]
    gate_up = ggml_moe_a8_vec(
        hidden_states, gate_up_q, topk_ids, top_k, int(gate_up_qtype), n2, num_tokens,
        gate_up_q.shape[2],
    )
    inter = act_fn(gate_up)
    # down: each of the T*top_k intermediate rows uses its own expert id.
    out = ggml_moe_a8_vec(
        inter, down_q, topk_ids, 1, int(down_qtype), h, num_tokens * top_k,
        down_q.shape[2],
    )
    # In place, deliberately -- see fused_q2_k_ud's identical note: avoids a
    # second [T, top_k, H] transient alongside a fresh, unaliased allocation.
    out = out.reshape(num_tokens, top_k, h)
    out.mul_(topk_weights.reshape(num_tokens, top_k, 1).to(out.dtype))
    return out.sum(dim=1)


__all__ = ["fused_experts_q4k_ud"]
