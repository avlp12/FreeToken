"""Qwen3.8-Flash-Next MTP (multi-token-prediction) draft head -- weight definition only.

**Scope**: this module makes the checkpoint's ``mtp.*`` tensors loadable and gives them a
plausible, documented forward pass so ``bench/mtp_accept_probe.py`` can measure a real
acceptance rate. It is **not** wired into the live serving path: no KV-cache slot is
reserved for this layer, no ``--speculative-*`` flag exists, and the offload MoE cache
does not know about ``mtp.layers.0.mlp.experts``. Turning the probe's number into an
actual speculative-decode loop is separate follow-on work.

Gate: ``FREETOKEN_LOAD_MTP=1`` (see ``weight.py``). Default OFF -- the text-only serving
path is unaffected byte-for-byte when the env var is unset (no ``mtp`` attribute is even
constructed on :class:`~freetoken.models.qwen4_exp.model.Qwen4ExpForCausalLM`).

=====================================================================================
What was actually measured (safetensors headers, ``/root/models/Qwen3.8-Flash-Next-NVFP4``,
no GPU) -- 31 tensors, all BF16, none quantized (``hf_quant_config.json``'s modelopt
``exclude_modules`` lists ``mtp.*`` and ``model.mtp.*`` wholesale)::

    mtp.fc_embedding.weight                                          [2560, 2560]
    mtp.fc_hidden.weight                                             [2560, 2560]
    mtp.pre_fc_norm_embedding.weight                                 [2560]
    mtp.pre_fc_norm_hidden.weight                                    [10240]           <- hc_count*hidden, NOT hidden
    mtp.hyper_connection_mixer.hc_norm.weight                        [10240]
    mtp.hyper_connection_mixer.input_mix_weight_down.weight          [320, 10240]      <- no block_inject_weight => use_combine=False
    mtp.hyper_connection_mixer.input_mix_weight_up.weight            [10240, 320]
    mtp.layers.0.self_attn.{q,k,v,o}_proj.weight                     matches the base model's
    mtp.layers.0.self_attn.{q,k}_norm.weight                            full_attention (QSA) group
    mtp.layers.0.self_attn.indexer.{index_qk_proj,q_layernorm,k_layernorm}  exactly (24 q / 2 kv /
                                                                             head_dim 256, indexer
                                                                             4x128 + 1x128)
    mtp.layers.0.{attn,mlp}_hyper_connection.{hc_norm,input_mix_weight_{down,up},
                  block_inject_weight}.weight                        same shapes as a base decoder layer's GatedResidual(use_combine=True)
    mtp.layers.0.mlp.gate.weight                                     [512, 2560]
    mtp.layers.0.mlp.experts.gate_up_proj                            [512, 1280, 2560]  <- ONE stacked BF16 tensor, not per-expert
    mtp.layers.0.mlp.experts.down_proj                                [512, 2560, 640]   <- (main model's routed experts are NVFP4, per-expert; MTP's are plain BF16, stacked)
    mtp.layers.0.mlp.shared_expert.{gate,up,down}_proj.weight        [640,2560]/[640,2560]/[2560,640]
    mtp.layers.0.mlp.shared_expert_gate.weight                       [1, 2560]

(31st tensor: ``mtp.hyper_connection_mixer.hc_norm.weight`` above.) See the executor's
report for the exact script used to read these headers.

**Inference, not verified against any reference implementation** (none exists -- HF
``transformers`` drops every ``mtp.*`` key on load; see ``weight.py``'s module docstring):
the checkpoint's own design note (``/root/MTP_DESIGN.md``, a prior session's scratch file,
*not* part of this worktree) assumed the hidden state this head consumes is the base
model's *already-collapsed* ``[T, hidden]`` output. That is inconsistent with
``pre_fc_norm_hidden`` measuring ``[hc_count*hidden] = [10240]`` -- a width only a
per-stream norm (identical pattern to ``hc.py``'s ``GroupedPlusOneRMSNorm`` / ``hc_norm``)
would use. This module instead assumes the tap point is the **raw hyper-connection
residual R** from the base model's *last* decoder layer, *before* the base model's own
final ``model.hyper_connection_mixer`` collapse -- the one interpretation consistent with
every one of the 31 measured shapes:

    R_last [T, hc_count*hidden]                          # base model, layer 47 output, pre-collapse
      -> pre_fc_norm_hidden (per-stream RMSNorm, zero-centered, hc_count groups)
      -> mtp.hyper_connection_mixer.mix(...)  -> h_mix [T, hidden]     # MTP's OWN low-rank collapse
      -> fc_hidden(h_mix)                     -> h [T, hidden]
    embed_tokens(next_token_ids)                          # shared table (mtp_use_dedicated_embeddings=False)
      -> pre_fc_norm_embedding (RMSNorm, zero-centered)
      -> fc_embedding(...)                    -> e [T, hidden]
    combined = h + e                                       # "split-add": the fc pair is the
                                                            # 2-matmul equivalent of the single
                                                            # concat-linear other Qwen3-Next/3.5
                                                            # MTP heads use (both fc_* are [hidden,hidden])
    R0 = combined.repeat(1, hc_count)                      # mirrors Qwen4ExpModel.forward's embed repeat
    R1 = mtp.layers[0](R0)                                 # one QSA decoder layer, own 512-expert bank
    hidden, _ = mtp.hyper_connection_mixer.mix(R1)          # SAME weights, reused as the output collapse
    logits = lm_head(hidden)                                # shared head, no mtp.norm tensor exists

The checkpoint ships exactly one ``hyper_connection_mixer`` (no ``block_inject_weight``,
i.e. built with ``use_combine=False`` -- the same shape as the base model's own top-level,
end-of-stack mixer). Reusing it for *both* the input collapse and the output collapse is
the only reading that (a) accounts for every one of the 31 tensors and (b) invents no new
weights. Flagged here and in the executor's report; the advisor's first GPU run of the
probe is the actual check (a much-better-than-baseline accept rate is circumstantial
support, not proof, since a wrong wiring could still land in a plausible range by chance).

**Attention is dense causal, not the live QSA sparse backend.** ``Qwen4ExpAttention``
hard-codes a call to ``get_global_ctx().attn_backend`` (the live engine's paged KV/QSA
backend), which has no slot reserved for this extra layer. Per ``TorchDenseQSAReference``'s
own docstring (``attention.py``), QSA is *exactly* equivalent to dense causal attention
whenever a request sees at most ``index_budget + index_ratio - 1`` tokens -- 2048 + 4 - 1
= 2051 for this checkpoint. The probe's captured sequences (a handful of prompts, well
under that bound) are within this range, so dense attention here is exact, not an
approximation. The indexer's weights are still loaded (state-dict complete, all 31
tensors land somewhere) but unused at forward time -- :class:`_MTPDenseAttention` only
overrides ``forward``; weight shapes/attribute names are inherited unchanged from
:class:`~.attention.Qwen4ExpAttention` so the loader's existing fusion rules
(``weight.py``'s ``_FUSIONS``, matched by suffix) apply to ``mtp.layers.0.self_attn.*``
with no additional code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, GemmaPlusOneRMSNorm, LinearReplicated, OPList
from freetoken.layers.rotary import get_rope
from freetoken.models.qwen3_5_moe.moe import _SharedExpert

from .attention import Qwen4ExpAttention
from .hc import GatedResidual, GroupedPlusOneRMSNorm

if TYPE_CHECKING:
    from freetoken.layers.embedding import ParallelLMHead, VocabParallelEmbedding
    from freetoken.models.config import ModelConfig

# One extra decoder layer past the base model's 48 (config.num_layers). Never registered
# with the live attention backend/KV pool -- see module docstring -- so this id is only a
# debug label, not an engine slot index.
MTP_LAYER_ID_OFFSET = 0


def _dense_causal_attend(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float
) -> torch.Tensor:
    """Reference dense causal attention: ``q``/``k``/``v`` are ``[T, heads, head_dim]``
    (``k``/``v`` may have fewer heads than ``q``; GQA repeat is applied here). fp32 math,
    same pattern as :class:`~.attention.TorchDenseQSAReference._attend`. CPU- and
    CUDA-safe; used because :class:`_MTPDenseAttention` cannot reach the live engine's
    QSA backend (see module docstring)."""
    t, num_q, _ = q.shape
    num_kv = k.shape[1]
    rep = num_q // num_kv
    kk = k.repeat_interleave(rep, dim=1).float()
    vv = v.repeat_interleave(rep, dim=1).float()
    scores = torch.einsum("qhd,khd->hqk", q.float(), kk) * sm_scale
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=q.device))
    scores = scores.masked_fill(~causal, float("-inf"))
    out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vv)
    return out.to(q.dtype)


class _MTPDenseAttention(Qwen4ExpAttention):
    """:class:`~.attention.Qwen4ExpAttention` with dense causal attention instead of the
    live ``get_global_ctx().attn_backend`` call -- see module docstring for why this is
    exact (not approximate) at this probe's context lengths. Same ``__init__`` (same
    weight shapes/state-dict keys) as the parent; only ``forward`` and the rope base
    (this checkpoint's ``mtp.rope_theta``, which may differ from the base model's) change.
    """

    def __init__(self, config: ModelConfig, layer_id: int, rope_theta: float) -> None:
        super().__init__(config, layer_id)
        rotary = config.rotary_config
        self.rotary = get_rope(
            head_dim=self.head_dim,
            rotary_dim=rotary.rotary_dim,
            max_position=rotary.max_position,
            base=rope_theta,
            rope_scaling=tuple(rotary.scaling.items()) if rotary.scaling else None,
        )
        self.sm_scale = config.attn_sm_scale or self.head_dim**-0.5

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        qg, k, v = self.qkv_proj.forward(x).split(self._qkv_split, dim=-1)
        qg = qg.view(-1, self.num_q, self.head_dim * 2)
        q = qg[..., : self.head_dim].contiguous()
        gate = qg[..., self.head_dim :].reshape(-1, self.qo_attn_dim)
        k = k.contiguous().view(-1, self.num_kv, self.head_dim)
        v = v.contiguous().view(-1, self.num_kv, self.head_dim)
        self.q_norm.forward_inplace(q)
        self.k_norm.forward_inplace(k)
        q, k = self.rotary.forward(
            positions, q.view(-1, self.qo_attn_dim), k.view(-1, self.kv_attn_dim)
        )
        q = q.view(-1, self.num_q, self.head_dim)
        k = k.view(-1, self.num_kv, self.head_dim)
        o = _dense_causal_attend(q, k, v, self.sm_scale)
        gated = o.reshape(-1, self.qo_attn_dim) * torch.sigmoid(gate)
        return self.o_proj.forward(gated)


class _MTPExperts(BaseOP):
    """Plain-PyTorch top-k routed experts for the MTP block's own 512-expert bank.

    The checkpoint stores this bank as two whole BF16 tensors (``[num_experts, ...]``,
    stacked -- not per-expert, not quantized), unlike the base model's per-expert NVFP4
    offload banks (``weight.py``'s ``_EXPERT_KEY_RE``/offload cache machinery, which does
    not apply here). Reusing ``freetoken.layers.moe.make_moe_layer`` would route through
    the live offload/KV-cache-aware ``OffloadMoELayer`` -- wrong for an unregistered,
    KV-cache-free probe layer -- so this does a straightforward masked per-expert loop
    instead. It is O(num_experts) python-level iterations per call: correct and simple,
    not fast. Fine for an offline probe evaluating at most a few hundred positions; not
    meant for a hot serving path.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool,
    ) -> None:
        self.num_experts = num_experts
        self.top_k = top_k
        self.renormalize = renormalize
        # [E, 2*intermediate, hidden] and [E, hidden, intermediate]; F.linear-style
        # (out_features, in_features) per expert, matching every other Linear in this
        # codebase, and exactly the checkpoint's measured shapes.
        self.gate_up_proj = torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        self.down_proj = torch.empty(num_experts, hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        probs = F.softmax(router_logits.float(), dim=-1)  # [N, E]
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)  # [N, K]
        if self.renormalize:
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        combine_weight = torch.zeros(
            num_tokens, self.num_experts, device=hidden_states.device, dtype=torch.float32
        )
        combine_weight.scatter_(1, topk_idx, topk_probs)

        x = hidden_states.float()
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            w = combine_weight[:, e]
            mask = w > 0
            if not bool(mask.any()):
                continue
            xe = x[mask]
            gate, up = F.linear(xe, self.gate_up_proj[e].float()).chunk(2, dim=-1)
            ye = F.linear(F.silu(gate) * up, self.down_proj[e].float())
            out[mask] += ye * w[mask].unsqueeze(-1)
        return out.to(hidden_states.dtype)


class _MTPMoE(BaseOP):
    """``routed(x) + sigmoid(shared_expert_gate(x)) * shared_expert(x)`` -- same formula
    as :class:`~.qwen3_5_moe.moe.Qwen3_5MoE`, with :class:`_MTPExperts` standing in for
    the base model's offload-backed routed experts (see :class:`_MTPExperts` docstring).
    ``gate``/``shared_expert``/``shared_expert_gate`` reuse the base model's classes
    unchanged: the shared ``config`` object's ``expert_quant``/``dense_quant`` already
    route ``_SharedExpert`` to its plain-BF16 branch for this checkpoint (both are "none"
    or "nvfp4" for reasons unrelated to MTP; MTP's own tensors are BF16 either way per
    ``hf_quant_config.json``'s blanket ``mtp.*`` exclusion).
    """

    def __init__(self, config: ModelConfig) -> None:
        self.experts = _MTPExperts(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
        )
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _SharedExpert(
            config, config.hidden_size, config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        shared = shared * torch.sigmoid(self.shared_expert_gate.forward(hidden_states))
        routed = self.experts.forward(hidden_states, router_logits)
        return (routed + shared).view(num_tokens, hidden_dim)


class _MTPDecoderLayer(BaseOP):
    """The MTP block's one decoder layer: identical shape to the base model's
    full-attention (QSA) layers, minus PLE (MTP has none -- ``ple_layer_ids`` never
    include it), with dense attention (:class:`_MTPDenseAttention`) and the dense
    :class:`_MTPMoE` in place of the live sparse-attention/offload-MoE backends."""

    def __init__(self, config: ModelConfig, layer_id: int, rope_theta: float) -> None:
        self.self_attn = _MTPDenseAttention(config, layer_id, rope_theta)
        self.mlp = _MTPMoE(config)
        self.attn_hyper_connection = GatedResidual(config)
        self.mlp_hyper_connection = GatedResidual(config)

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        block_input, inject = self.attn_hyper_connection.mix(hidden)
        block_output = self.self_attn.forward(block_input, positions)
        hidden = self.attn_hyper_connection.combine(hidden, block_output, inject)
        block_input, inject = self.mlp_hyper_connection.mix(hidden)
        return self.mlp_hyper_connection.combine(hidden, self.mlp.forward(block_input), inject)


class Qwen4ExpMTPHead(BaseOP):
    """Draft head for depth-1 MTP acceptance probing. See the module docstring for the
    verified tensor inventory and the (documented, unverified-by-reference) forward
    recipe. Only constructed when ``FREETOKEN_LOAD_MTP=1`` (``weight.py``); the state
    dict this produces is exactly the 31 ``mtp.*`` checkpoint tensors, keyed to match
    ``weight.py``'s pass-through renaming with no further translation needed.
    """

    def __init__(self, config: ModelConfig, layer_id: int, rope_theta: float) -> None:
        args = config.qwen4_args
        self.hc_count = args.hc_count
        self.hidden_size = config.hidden_size
        self.fc_embedding = LinearReplicated(config.hidden_size, config.hidden_size, has_bias=False)
        self.fc_hidden = LinearReplicated(config.hidden_size, config.hidden_size, has_bias=False)
        self.pre_fc_norm_embedding = GemmaPlusOneRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = GroupedPlusOneRMSNorm(
            args.ple_state_width, config.rms_norm_eps, self.hc_count
        )
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)
        self.layers = OPList([_MTPDecoderLayer(config, layer_id, rope_theta)])

    def fuse_inputs(
        self,
        next_token_ids: torch.Tensor,
        prev_layer_r: torch.Tensor,
        embed_tokens: VocabParallelEmbedding,
    ) -> torch.Tensor:
        """``[T, hc_count*hidden]`` fused input to ``layers[0]`` -- the input-side half
        of the recipe in the module docstring. Split out from :meth:`forward` because it
        needs no attention backend / engine context, so it is unit-testable on CPU."""
        rn = self.pre_fc_norm_hidden.forward(prev_layer_r)
        h_mix, _ = self.hyper_connection_mixer.mix(rn)
        h = self.fc_hidden.forward(h_mix)
        e = embed_tokens.forward(next_token_ids)
        en = self.pre_fc_norm_embedding.forward(e)
        eh = self.fc_embedding.forward(en)
        return (h + eh).repeat(1, self.hc_count)

    def forward(
        self,
        next_token_ids: torch.Tensor,
        prev_layer_r: torch.Tensor,
        positions: torch.Tensor,
        embed_tokens: VocabParallelEmbedding,
        lm_head: ParallelLMHead,
    ) -> torch.Tensor:
        """Predict the token after ``next_token_ids`` given ``prev_layer_r`` (the base
        model's raw last-layer hyper-connection R at the same positions). Returns
        ``[T, vocab]`` logits from the shared ``lm_head`` -- callers take ``argmax`` for
        the depth-1 top-1 draft. Requires CUDA (rope/gemma-norm kernels have no CPU path;
        see the executor's report)."""
        r0 = self.fuse_inputs(next_token_ids, prev_layer_r, embed_tokens)
        r1 = self.layers.op_list[0].forward(r0, positions)
        hidden, _ = self.hyper_connection_mixer.mix(r1)
        return lm_head.forward(hidden)


__all__ = ["MTP_LAYER_ID_OFFSET", "Qwen4ExpMTPHead"]
