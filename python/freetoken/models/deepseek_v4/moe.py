"""DSV4 MoE: sqrtsoftplus/hash router, shared SwiGLU expert, offloaded FP4 routed
experts (GPU slot-cache / cpu / hybrid decode paths)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator
from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
from freetoken.kernel.triton.dsv4.router import fused_router, unfused_router
from freetoken.kernel.triton.dsv4.swiglu import fused_swiglu
from freetoken.layers import OffloadMoELayer
from freetoken.moe.prefetch import get_prefetcher as _get_moe_prefetcher
from freetoken.moe.routing_trace import get_tracer as _get_routing_tracer

from .args import DeepseekV4Args
from .layers import Linear

from .parallel import div_tp, tp_size


def _fuse_router(score_func: str) -> bool:
    """Whether ``Gate.forward``'s tail runs as the single fused kernel.

    The kernel implements the sqrtsoftplus scoring function only -- the one every
    DSV4 checkpoint uses. ``softmax`` (which also skips the renorm) and
    ``sigmoid`` stay on the torch chain: no checkpoint exercises them, so a
    second and third kernel specialization would be untested code on a path that
    is not hot. Read per call, not at import: the escape hatch has to be usable
    from a test that flips the env var after the module is loaded, and the read
    is a dict lookup on a path that is already doing a GEMV.
    """
    return score_func not in ("softmax", "sigmoid") and not unfused_router()


class Gate(nn.Module):
    """MoE router: sqrtsoftplus scoring + hash routing (first ``n_hash_layers``)."""

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        super().__init__()
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim, dtype=torch.bfloat16), requires_grad=False)
        if self.hash:
            self.tid2eid = nn.Parameter(
                torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int64), requires_grad=False
            )
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32), requires_grad=False)

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        return_scores: bool = False,
        want_int32: bool = False,
    ):
        """Route ``x``: ``(weights, indices)``, plus ``sel_scores`` / int32 ids on request.

        ``want_int32`` additionally returns the expert ids as a fresh contiguous
        int32 tensor -- the form ``MoE.forward`` has to hand ``routed_forward``
        anyway. The fused kernel stores it in the same launch it stores the int64
        ids, so asking for it costs nothing and saves the caller a cast; the
        buffer is exclusively owned, so the offload cache's in-place expert-id ->
        slot-id rewrite is safe on it.
        """
        scores = bf16_linear_fp32(x, self.weight)
        if _fuse_router(self.score_func):
            weights, indices, idx32, sel_scores = fused_router(
                scores,
                self.bias,
                self.tid2eid if self.hash else None,
                input_ids if self.hash else None,
                top_k=self.topk,
                route_scale=self.route_scale,
                renormalize=True,
                want_int32=want_int32,
                want_sel=return_scores,
            )
            out = (weights, indices)
            if return_scores:
                out = out + (sel_scores,)
            if want_int32:
                out = out + (idx32,)
            return out
        return self._reference_route(
            scores, input_ids, return_scores=return_scores, want_int32=want_int32
        )

    def _reference_route(
        self,
        scores: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        return_scores: bool = False,
        want_int32: bool = False,
    ):
        """Pre-fusion torch composition of the router tail (nine launches).

        Reached under ``FREETOKEN_UNFUSED_ROUTER=1``, and unconditionally for the
        ``softmax``/``sigmoid`` scoring functions, which no DSV4 checkpoint uses
        and which the fused kernel therefore does not implement. It is also the
        reference the bit-identity test compares against, so it must stay a
        faithful transcription of the original chain -- do not "simplify" it.
        """
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        # Pre-renorm selection score: original_scores (post score_func, PRE the
        # e-score load-balancing bias) gathered at the chosen indices. This is the
        # same numerator `weights` below is built from, captured before the
        # per-step renorm/route_scale -- so it stays >= 0 by construction
        # (softplus/sigmoid/softmax are all non-negative) and, unlike `weights`
        # (forced to sum to route_scale every step), is not warped by what else
        # got selected that step. That is what makes it usable for cross-expert /
        # cross-step thresholding (e.g. a lookahead prefetcher's "how confident is
        # this pick" question), whereas `weights` only answers "how much of the
        # combine does this pick get THIS step". `return_scores` gates the extra
        # gather (already computed either way, so free) behind an explicit opt-in
        # so the real per-layer routing call (`MoE.forward`'s `self.gate(x,
        # flat_ids)`, on the CUDA-graph-captured decode path) is unchanged in
        # both signature and cost; only the routing tracer's lookahead calls pass
        # `return_scores=True`. See freetoken/moe/routing_trace.py.
        sel_scores = original_scores.gather(1, indices)
        weights = sel_scores
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        out = (weights, indices)
        if return_scores:
            out = out + (sel_scores,)
        if want_int32:
            out = out + (indices.to(torch.int32).contiguous(),)
        return out


class Expert(nn.Module):
    """Dense SwiGLU expert (the shared expert; routed experts are offloaded FP4).

    Under TP the intermediate dimension splits: ``w1``/``w3`` are column-parallel and
    ``w2`` is row-parallel, so the output is a partial sum. ``MoE.forward`` owns the
    single all-reduce that completes it together with the routed half.
    """

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float):
        super().__init__()
        inter_local = div_tp(inter_dim, "moe_inter_dim", multiple_of=128)
        self.w1 = Linear(dim, inter_local, kind="fp8")
        self.w2 = Linear(inter_local, dim, kind="fp8")
        self.w3 = Linear(dim, inter_local, kind="fp8")
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = fused_swiglu(self.w1(x), self.w3(x), self.swiglu_limit, x.dtype)
        return self.w2(h)


class DSV4OffloadMoELayer(OffloadMoELayer):
    """Routed FP4 experts on the shared offload cache: the base whole-layer
    streaming prefill (grouped inline-dequant GEMM for dense chunks, GEMV
    below the route crossover) and slot-cache / cpu / hybrid decode paths
    (per-route dequant GEMV)."""

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        super().__init__(
            layer_id=layer_id,
            num_experts=args.n_routed_experts,
            top_k=args.n_activated_experts,
            hidden_size=args.dim,
            intermediate_size=args.moe_inter_dim,
            renormalize=True,
            activation="silu",
        )
        self.swiglu_limit = args.swiglu_limit

    def _maybe_all_reduce(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Suppress the base class's per-call collective.

        The routed and the shared expert are both partial sums over the same output
        dim, so DSV4 adds them first and all-reduces ONCE in ``MoE.forward``. Letting
        the base reduce here would cost a second collective per layer and would also
        double-count the shared expert.
        """
        return hidden_states

    def _prefill_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Whole-layer streaming moves all num_experts rows per layer; a small
        # chunk touches at most T*top_k of them, so below that crossover the
        # decode-style on-demand slot path strictly moves fewer bytes (and
        # keeps short-prompt slot residency -- hence hybrid decode's GPU/CPU
        # route split -- unchanged). Mixing modes across chunks is safe: the
        # streaming buffers disown their borrowed slots on invalidation.
        cache = self.offload_cache
        assert cache is not None

        # A speculative verify is a decode wearing a prefill's clothes. The scheduler
        # marks the batch "prefill" because the block is an extend over several
        # positions, but it carries block_size rows per request, not a prompt -- and it
        # runs on the critical path of every decode step.
        #
        # The path below fetches EVERY missing expert over PCIe, uncapped, with no CPU
        # overlap. That is the right trade for a prompt, where the fetch amortizes over
        # hundreds of tokens. For a 5-row block it moves up to T*top_k experts per layer
        # with nothing hiding the latency: measured at ~0.4s per block against a 0.06s
        # single-token step, which is the whole reason speculation lost to plain decode
        # here rather than a low acceptance rate.
        #
        # Hybrid decode caps the fetch and overlaps the overflow on the CPU pool, which
        # is what a handful of rows wants.
        if (
            getattr(get_global_ctx().batch, "speculative", False)
            and cache.decode_target == "hybrid"
        ):
            return self._decode_routed(hidden_states, topk_weights, topk_ids)
        # unpinned (LOCKED) layers must take the base materialize path: their copy_missing is the whole-layer pageable branch with position == expert id, which ensure_experts's LRU slot remap would contradict (the GEMM would gather other experts' weights)
        if (
            hidden_states.shape[0] * self.top_k >= self.num_experts
            or cache.is_unpinned_layer(self.layer_id)
        ):
            return super()._prefill_routed(hidden_states, topk_weights, topk_ids)
        cache.ensure_experts(self.layer_id, topk_ids)  # in-place expert-id -> slot
        cache.copy_missing()
        if cache.collect_stats:
            cache.record_decode_stats(self.layer_id)
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=True,
        )


class MoE(nn.Module):
    """Sparse MoE: hash/score router -> offloaded FP4 routed experts + shared expert."""

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        super().__init__()
        self.dim = args.dim
        self.gate = Gate(layer_id, args)
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, args.swiglu_limit)
        self.experts = DSV4OffloadMoELayer(layer_id, args)
        self._comm = DistributedCommunicator() if tp_size() > 1 else None
        # Publish this layer's router so layer L-1 can run it one layer ahead (in-graph
        # L+1 expert prefetch). Registration is unconditional and free -- one dict
        # entry, no tensors -- which keeps it independent of when FREETOKEN_MOE_PREFETCH
        # is read; whether anything actually prefetches is decided by
        # ``experts.prefetcher``, set only when the cache allocated prefetch state.
        _get_moe_prefetcher().register_gate(
            layer_id,
            self.gate,
            n_layers=args.n_layers,
            top_k=args.n_activated_experts,
            n_experts=args.n_routed_experts,
        )
        # Research instrumentation, off unless FREETOKEN_ROUTING_TRACE was set at
        # process start. Resolved once here so ``forward`` costs one ``is None``
        # test in the default configuration -- see freetoken/moe/routing_trace.py.
        self._routing_trace = _get_routing_tracer()
        if self._routing_trace is not None:
            self._routing_trace_layer_id = layer_id
            self._routing_trace.register_gate(
                layer_id,
                self.gate,
                n_layers=args.n_layers,
                top_k=args.n_activated_experts,
                n_experts=args.n_routed_experts,
            )

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        flat_ids = input_ids.flatten()
        # ``want_int32``: the routed GEMM needs the ids as contiguous int32 and the
        # offload decode rewrites them in place, so the cast is not optional -- it is
        # just cheaper inside the router kernel (one extra store) than as its own
        # launch. The int64 ids still come back unchanged for the prefetcher and the
        # routing tracer, which slice and read them.
        weights, indices, ids32 = self.gate(x, flat_ids, want_int32=True)
        if self._routing_trace is not None:
            self._maybe_trace_routing(x, flat_ids, weights, indices)
        if self.experts.prefetcher is not None:
            # ``x`` is the post-ffn_norm hidden this layer's router just consumed --
            # the exact tensor layer L+1's Gate is replayed on, which is what makes
            # the prediction a genuine one-layer-early question. ``flat_ids`` matters
            # for the leading hash routers, which key on the token id.
            self.experts.prefetch_ctx = (x, flat_ids)
        # Shared expert enqueued before routed_forward: hybrid decode blocks on the
        # CPU pool inside routed_forward, so this GEMM must already be on the stream
        # to overlap the CPU overflow compute.
        shared = self.shared_experts(x)
        # routed_forward may mutate the ids in place (offload decode slot remap);
        # ``ids32`` is a fresh buffer the gate wrote this call and nothing else
        # reads, so no clone is needed here -- same guarantee the old
        # ``indices.to(int32)`` copy gave. ``weights`` is already fp32 contiguous
        # (kernel output), so ``.float().contiguous()`` would be two no-ops.
        routed = self.experts.routed_forward(x, weights, ids32)
        out = routed + shared
        # Both halves are partial sums over the split intermediate dim; one collective
        # completes the layer (see DSV4OffloadMoELayer._maybe_all_reduce).
        if self._comm is not None:
            out = self._comm.all_reduce(out)
        return out.view(shape)

    def _maybe_trace_routing(self, x, flat_ids, weights, indices) -> None:
        """Routing-trace tap (only reached with FREETOKEN_ROUTING_TRACE set).

        ``x`` is the post-``ffn_norm`` hidden this layer's router just consumed --
        the exact tensor the lookahead routers of L+1/L+2 are replayed on, which
        is what makes the prediction a genuine "one layer early" question rather
        than a re-derivation of the later layer's own input.

        Decode only: prefill materializes routing over the whole prompt and a
        speculative verify block rides the prefill phase, so both are skipped
        here. The phase is a host value read while the graph is being TRACED, not
        a device value branched on during replay, so the captured graph contains
        exactly the decode-path record ops.
        """
        tracer = self._routing_trace
        assert tracer is not None
        if get_global_ctx().batch.is_prefill:
            return
        if not tracer.traces_layer(self._routing_trace_layer_id):
            return
        tracer.record(self._routing_trace_layer_id, x, flat_ids, weights, indices)
