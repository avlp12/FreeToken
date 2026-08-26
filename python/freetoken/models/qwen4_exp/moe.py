from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    make_moe_layer,
    silu_and_mul,
)
from freetoken.moe.prefetch import get_prefetcher as _get_moe_prefetcher
from freetoken.moe.routing_trace import get_tracer as _get_routing_tracer

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class _LookaheadGate:
    """Adapts a Qwen4ExpMoE layer's plain softmax router (a bare ``LinearReplicated``
    -- this model has no ``Gate`` class of its own, unlike DSV4) to the DSV4-style
    ``Gate.forward`` interface the process-wide lookahead-Gate registry
    (``freetoken/moe/prefetch.py``) and the routing tracer
    (``freetoken/moe/routing_trace.py``) both call a registered gate through:
    ``gate(hidden, input_ids) -> (weights, indices)``, or ``(weights, indices,
    sel_scores)`` with ``return_scores=True``.

    Reproduces exactly the softmax -> top-k -> renormalize chain the REAL routing
    decision runs (``freetoken/moe/fused.py``'s ``fused_topk``/``_torch_fused_topk``:
    top-10 is not a power of 2, so the real path is unconditionally the pure-torch
    formula this mirrors, never the fused triton_kernels path), so a prediction here
    is exactly what the target layer's own router would pick given this input.  Purely
    additive and read-only against ``self._linear`` (the same ``LinearReplicated`` the
    owning layer's real ``self.gate.forward`` call already uses) -- it never feeds back
    into the real routing call, so it cannot change a single output token.

    Deliberately NOT a ``BaseOP``: it owns no parameters (it borrows the already-loaded
    linear projection) and must never appear in ``Qwen4ExpMoE``'s ``state_dict``/
    ``load_state_dict`` walk -- which is also why the owning layer stores it as
    ``self._lookahead_gate`` (leading underscore: both walks in ``freetoken.layers.base
    .BaseOP`` skip such names outright, so the type wouldn't matter either, but a plain
    object keeps it unambiguous).
    """

    hash = False  # qwen4_exp has no hash router; the prefetcher never truncation-exempts it

    def __init__(self, linear_gate: LinearReplicated, top_k: int, renormalize: bool):
        self._linear = linear_gate
        self._top_k = top_k
        self._renormalize = renormalize

    def __call__(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        return_scores: bool = False,
        want_int32: bool = False,
    ):
        logits = self._linear.forward(x)
        probs = torch.softmax(logits.float(), dim=-1)
        # Pre-renorm selection score (analogous to DSV4 Gate.forward's `sel_scores`):
        # the softmax probability of each chosen expert BEFORE the renorm below. This
        # model has no e-score bias to gather ahead of, so topk on `probs` directly
        # is both the selection and the score.
        sel_scores, indices = torch.topk(probs, self._top_k, dim=-1)
        weights = sel_scores
        if self._renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        out = (weights, indices)
        if return_scores:
            out = out + (sel_scores,)
        if want_int32:
            out = out + (indices.to(torch.int32).contiguous(),)
        return out


class _SharedExpert(BaseOP):
    """Always-on shared SwiGLU expert. Deliberately bf16-only: the Qwen3.8-FP8 release
    whitelists the shared expert out of quantization (no weight_scale_inv in the
    checkpoint), so unlike qwen3_5 this must NOT follow config.expert_quant."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size, [intermediate_size, intermediate_size], has_bias=False
        )
        self.down_proj = LinearRowParallel(intermediate_size, hidden_size, has_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class Qwen4ExpMoE(BaseOP):
    """512 routed experts (fp8_block banks, offload cache) top-10 + sigmoid-gated shared
    expert. Router: softmax over ALL experts -> top-k -> renormalize (norm_topk_prob=True
    per the transformers Qwen4ExpTextConfig default; HF evaluates shared expert first)."""

    def __init__(self, config: ModelConfig, layer_id: int | None = None):
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            weight_format="fp8_block",
        )
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _SharedExpert(
            config.hidden_size, config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

        # In-graph L+1/L+2 expert prefetch: publish this layer's router so layer L-1
        # can run it one layer ahead (freetoken/moe/prefetch.py). Registration is
        # unconditional and free (one dict entry, no tensors), which keeps it
        # independent of when --moe-prefetch is read; whether anything actually
        # prefetches is decided later, per layer, by ``self.experts.prefetcher``
        # (set by ``attach_offload_moe_cache`` only when the cache allocated prefetch
        # state). See _LookaheadGate above for why this needs an adapter rather than
        # registering ``self.gate`` directly, the way DSV4's ``Gate`` module can.
        self._layer_id = layer_id
        self._lookahead_gate = None
        self._routing_trace = None
        if layer_id is not None:
            self._lookahead_gate = _LookaheadGate(
                self.gate,
                top_k=config.num_experts_per_tok,
                renormalize=config.norm_topk_prob,
            )
            _get_moe_prefetcher().register_gate(
                layer_id,
                self._lookahead_gate,
                n_layers=config.num_layers,
                top_k=config.num_experts_per_tok,
                n_experts=config.num_experts,
            )
            # Research instrumentation, off unless FREETOKEN_ROUTING_TRACE was set at
            # process start -- see freetoken/moe/routing_trace.py.
            self._routing_trace = _get_routing_tracer()
            if self._routing_trace is not None:
                self._routing_trace.register_gate(
                    layer_id,
                    self._lookahead_gate,
                    n_layers=config.num_layers,
                    top_k=config.num_experts_per_tok,
                    n_experts=config.num_experts,
                )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Router + shared expert BEFORE the routed experts: the fused MoE kernel may
        # write into hidden_states in place (same ordering note as qwen3_5).
        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        shared = shared * torch.sigmoid(self.shared_expert_gate.forward(hidden_states))

        if self.experts.prefetcher is not None or self._routing_trace is not None:
            # ``hidden_states`` is the exact tensor ``self.gate`` (THIS layer's own
            # router) just consumed -- the same tensor the prefetcher replays layer
            # L+1's Gate on, which is what makes that a genuine one-layer-early
            # question rather than a re-derivation of L+1's own input. Mirrors
            # DSV4's MoE.forward (models/deepseek_v4/moe.py) exactly; the hyper-
            # connection mixer has already collapsed the 4x2560 residual stream
            # into this hidden_size-wide tensor by the time it reaches here (see
            # models/qwen4_exp/model.py's ``mixed, inject =
            # self.mlp_hyper_connection.forward(x4)``), so it is the layer's actual
            # router input, not the raw 4-wide stream.
            flat_ids = get_global_ctx().batch.input_ids.flatten()
        if self.experts.prefetcher is not None:
            self.experts.prefetch_ctx = (hidden_states, flat_ids)
        if self._routing_trace is not None:
            self._maybe_trace_routing(hidden_states, flat_ids)

        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        return (routed + shared).view(num_tokens, hidden_dim)

    def _maybe_trace_routing(self, x: torch.Tensor, flat_ids: torch.Tensor) -> None:
        """Routing-trace tap (only reached with FREETOKEN_ROUTING_TRACE set).

        Recomputes this layer's actual top-k/weights via ``self._lookahead_gate``
        (the identical softmax -> top-k -> renormalize formula ``self.experts.forward``
        runs internally through ``fused_topk``) purely so the tracer has the real
        decision to score lookahead predictions against -- it does not feed back into
        real routing. Decode only, matching DSV4's tap (models/deepseek_v4/moe.py's
        ``MoE._maybe_trace_routing``): prefill materializes routing over the whole
        prompt and a speculative verify block rides the prefill phase.
        """
        tracer = self._routing_trace
        assert tracer is not None
        if get_global_ctx().batch.is_prefill:
            return
        if not tracer.traces_layer(self._layer_id):
            return
        assert self._lookahead_gate is not None
        weights, indices = self._lookahead_gate(x, flat_ids)
        tracer.record(self._layer_id, x, flat_ids, weights, indices)


__all__ = ["Qwen4ExpMoE"]
