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
same vector kernel -- correct, just slower than a grouped GEMM would be.
"""

from __future__ import annotations

import torch


def fused_experts_q2k_ud(
    hidden_states: torch.Tensor,  # [T, H]
    gate_up_q: torch.Tensor,  # [num_slots, 2I, gate_up_pitch] uint8
    down_q: torch.Tensor,  # [num_slots, H, down_pitch] uint8
    topk_weights: torch.Tensor,  # [T, top_k]
    topk_ids: torch.Tensor,  # [T, top_k] int32 -> bank row
    gate_up_qtype: int,
    down_qtype: int,
    swiglu_limit: float,
) -> torch.Tensor:
    """Routed-expert output summed over the top-k routes (excludes the shared expert).

    ``topk_ids`` already index the bank rows: cache slots on the decode path,
    materialized layer positions (position == expert id) on the streaming prefill path.
    """
    from freetoken.kernel.gguf import ggml_moe_a8_vec
    from freetoken.kernel.triton.dsv4.fused_moe import fused_swiglu

    num_tokens = hidden_states.shape[0]
    n2 = gate_up_q.shape[1]  # 2 * intermediate
    h = down_q.shape[1]  # hidden
    top_k = topk_ids.shape[1]

    # gate_up: [T*top_k, 2I] -> clamped swiglu -> [T*top_k, I]
    gate_up = ggml_moe_a8_vec(
        hidden_states, gate_up_q, topk_ids, top_k, int(gate_up_qtype), n2, num_tokens,
        gate_up_q.shape[2],
    )
    inter = fused_swiglu(gate_up, swiglu_limit)
    # down: each of the T*top_k intermediate rows uses its own expert id.
    out = ggml_moe_a8_vec(
        inter, down_q, topk_ids, 1, int(down_qtype), h, num_tokens * top_k,
        down_q.shape[2],
    )
    out = out.reshape(num_tokens, top_k, h) * topk_weights.reshape(num_tokens, top_k, 1).to(
        out.dtype
    )
    return out.sum(dim=1)


__all__ = ["fused_experts_q2k_ud"]
