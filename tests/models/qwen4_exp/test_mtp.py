"""FREETOKEN_LOAD_MTP=1 weight loading + Qwen4ExpMTPHead, against a synthetic checkpoint.

No GPU: everything here either runs on plain tensors (F.linear/softmax) or explicitly takes
the CPU fallback path of the reused base-model ops (``GroupedPlusOneRMSNorm``,
``GatedResidual.mix`` on a non-cuda tensor). The parts that only have a CUDA kernel
(``GemmaPlusOneRMSNorm`` -- ``pre_fc_norm_embedding``, ``q_norm``/``k_norm``, the indexer
norms -- and rope) are exercised structurally (constructed, present in the state dict) but
not run; see test_weight_ckpt.py-style real-checkpoint runs for that, or the GPU probe.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers import BaseOP
from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.mtp import Qwen4ExpMTPHead, _MTPExperts
from freetoken.models.qwen4_exp.weight import _rename, iter_weights, load_mtp_enabled
from freetoken.utils import torch_dtype

from .common import toy_hf_config

# Toy full-attention-group geometry (matches toy_hf_config): hidden 128, hc_count 4,
# hc_lowrank 16, q/kv heads 4/1 @ head_dim 64, indexer 2x1 @ head_dim 64, 8 experts top-2
# @ intermediate 64/64. Real-checkpoint shapes are documented in mtp.py's module docstring.
H, HC, LR, HCH = 128, 4, 16, 512
QH, KVH, AHD = 4, 1, 64
IH, IKVH, IHD = 2, 1, 64
E, TOPK, MI, SI = 8, 2, 64, 64


@pytest.fixture(scope="module", autouse=True)
def _tp_info():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


@pytest.fixture
def toy_config():
    return parse_config(toy_hf_config(num_layers=4))


def _bf16(*shape: int) -> torch.Tensor:
    return torch.randn(*shape).to(torch.bfloat16)


def _hc_weights(prefix: str, inject: bool) -> dict[str, torch.Tensor]:
    w = {
        f"{prefix}.hc_norm.weight": _bf16(HCH),
        f"{prefix}.input_mix_weight_down.weight": _bf16(LR, HCH),
        f"{prefix}.input_mix_weight_up.weight": _bf16(HCH, LR),
    }
    if inject:
        w[f"{prefix}.block_inject_weight.weight"] = _bf16(HC, HCH)
    return w


def _raw_mtp_checkpoint() -> dict[str, torch.Tensor]:
    """All 31 ``mtp.*`` tensors, toy-sized but shaped exactly like the real checkpoint
    (same key names, same fusion geometry: qkv split, hc down+block_inject, shared-expert
    gate+up)."""
    raw: dict[str, torch.Tensor] = {}
    raw.update(_hc_weights("mtp.hyper_connection_mixer", inject=False))
    layer = "mtp.layers.0"
    raw.update(_hc_weights(f"{layer}.attn_hyper_connection", inject=True))
    raw.update(_hc_weights(f"{layer}.mlp_hyper_connection", inject=True))
    raw.update({
        f"{layer}.self_attn.q_proj.weight": _bf16(2 * QH * AHD, H),
        f"{layer}.self_attn.k_proj.weight": _bf16(KVH * AHD, H),
        f"{layer}.self_attn.v_proj.weight": _bf16(KVH * AHD, H),
        f"{layer}.self_attn.o_proj.weight": _bf16(H, QH * AHD),
        f"{layer}.self_attn.q_norm.weight": _bf16(AHD),
        f"{layer}.self_attn.k_norm.weight": _bf16(AHD),
        f"{layer}.self_attn.indexer.index_qk_proj.weight": _bf16(IH * IHD + IKVH * IHD, H),
        f"{layer}.self_attn.indexer.q_layernorm.weight": _bf16(IHD),
        f"{layer}.self_attn.indexer.k_layernorm.weight": _bf16(IHD),
        f"{layer}.mlp.gate.weight": _bf16(E, H),
        f"{layer}.mlp.experts.gate_up_proj": _bf16(E, 2 * MI, H),
        f"{layer}.mlp.experts.down_proj": _bf16(E, H, MI),
        f"{layer}.mlp.shared_expert.gate_proj.weight": _bf16(SI, H),
        f"{layer}.mlp.shared_expert.up_proj.weight": _bf16(SI, H),
        f"{layer}.mlp.shared_expert.down_proj.weight": _bf16(H, SI),
        f"{layer}.mlp.shared_expert_gate.weight": _bf16(1, H),
        "mtp.fc_embedding.weight": _bf16(H, H),
        "mtp.fc_hidden.weight": _bf16(H, H),
        "mtp.pre_fc_norm_embedding.weight": _bf16(H),
        "mtp.pre_fc_norm_hidden.weight": _bf16(HCH),
    })
    assert len(raw) == 31, f"fixture drifted from the measured 31-tensor inventory: {len(raw)}"
    return raw


@pytest.fixture
def mtp_checkpoint(tmp_path) -> str:
    save_file(_raw_mtp_checkpoint(), str(tmp_path / "model-bf16-00000.safetensors"))
    return str(tmp_path)


class _FakeRoot(BaseOP):
    """Stands in for Qwen4ExpForCausalLM: the one attribute that matters for state-dict
    prefixing is the name ``mtp``."""

    def __init__(self, head: Qwen4ExpMTPHead) -> None:
        self.mtp = head

    def forward(self):  # pragma: no cover - BaseOP requires it, unused here
        raise NotImplementedError


def test_load_mtp_enabled_reads_the_env_var(monkeypatch):
    monkeypatch.delenv("FREETOKEN_LOAD_MTP", raising=False)
    assert load_mtp_enabled() is False
    monkeypatch.setenv("FREETOKEN_LOAD_MTP", "1")
    assert load_mtp_enabled() is True
    # anything other than exactly "1" stays off (no accidental truthy strings)
    monkeypatch.setenv("FREETOKEN_LOAD_MTP", "true")
    assert load_mtp_enabled() is False


def test_rename_drops_mtp_by_default_and_passes_through_when_enabled(monkeypatch):
    monkeypatch.delenv("FREETOKEN_LOAD_MTP", raising=False)
    assert _rename("mtp.fc_embedding.weight") is None
    assert _rename("mtp.layers.0.mlp.experts.gate_up_proj") is None
    monkeypatch.setenv("FREETOKEN_LOAD_MTP", "1")
    assert _rename("mtp.fc_embedding.weight") == "mtp.fc_embedding.weight"
    assert _rename("mtp.layers.0.mlp.experts.gate_up_proj") == "mtp.layers.0.mlp.experts.gate_up_proj"


def test_visual_is_dropped_regardless_of_the_mtp_gate(monkeypatch):
    for env in (None, "1"):
        if env is None:
            monkeypatch.delenv("FREETOKEN_LOAD_MTP", raising=False)
        else:
            monkeypatch.setenv("FREETOKEN_LOAD_MTP", env)
        assert _rename("model.visual.blocks.0.attn.qkv.weight") is None
        assert _rename("visual.merger.norm.weight") is None


def test_config_carries_mtp_rope_theta_with_fallback():
    # toy_hf_config ships no `mtp` section: falls back to the base model's rope_theta.
    cfg = parse_config(toy_hf_config(num_layers=4))
    assert cfg.qwen4_args.mtp_rope_theta == pytest.approx(10000.0)

    # an explicit mtp section (as the real NVFP4 checkpoint ships) wins.
    cfg2 = parse_config(toy_hf_config(num_layers=4, mtp={"rope_theta": 10_000_000}))
    assert cfg2.qwen4_args.mtp_rope_theta == pytest.approx(10_000_000)


def test_iter_weights_yields_nothing_mtp_when_disabled(mtp_checkpoint, monkeypatch):
    monkeypatch.delenv("FREETOKEN_LOAD_MTP", raising=False)
    loaded = dict(
        iter_weights(mtp_checkpoint, torch.device("cpu"), include_moe_experts=True, include_non_moe=True)
    )
    assert loaded == {}


def test_mtp_head_loads_every_tensor_with_no_leftovers(mtp_checkpoint, monkeypatch, toy_config):
    """The end-to-end contract: FREETOKEN_LOAD_MTP=1 + iter_weights + Qwen4ExpMTPHead's
    state dict line up exactly -- same names, shapes and dtypes, nothing missing or
    extra. This is the load-bearing test for "MTP tensors become loadable"."""
    monkeypatch.setenv("FREETOKEN_LOAD_MTP", "1")
    loaded = dict(
        iter_weights(mtp_checkpoint, torch.device("cpu"), include_moe_experts=True, include_non_moe=True)
    )
    # 31 raw tensors minus 5 two-part fusions (qkv; attn_hc + mlp_hc down/block_inject;
    # shared_expert gate/up) = 26 keys actually landing in the model.
    assert len(loaded) == 26, sorted(loaded)

    # Real construction always happens inside `torch_dtype(config.dtype)` (engine.py); the
    # checkpoint is BF16, so match that here or every leaf tensor is allocated fp32 by
    # torch.empty's default and load_state_dict's dtype assertion trips on every key.
    with torch_dtype(torch.bfloat16):
        head = Qwen4ExpMTPHead(
            toy_config, layer_id=toy_config.num_layers, rope_theta=toy_config.qwen4_args.mtp_rope_theta
        )
    root = _FakeRoot(head)
    root.load_state_dict(dict(loaded))  # raises on any missing/unexpected key or shape/dtype mismatch


def test_mtp_experts_routes_to_exactly_top_k_and_matches_a_manual_reference():
    torch.manual_seed(0)
    num_experts, top_k, hidden, inter = 6, 2, 5, 4
    experts = _MTPExperts(
        num_experts=num_experts, top_k=top_k, hidden_size=hidden, intermediate_size=inter,
        renormalize=True,
    )
    experts.gate_up_proj = torch.randn(num_experts, 2 * inter, hidden)
    experts.down_proj = torch.randn(num_experts, hidden, inter)

    tokens = 7
    hidden_states = torch.randn(tokens, hidden)
    # Router logits engineered so token t's top-2 experts are exactly {t % E, (t+1) % E}.
    router_logits = torch.full((tokens, num_experts), -10.0)
    for t in range(tokens):
        router_logits[t, t % num_experts] = 5.0
        router_logits[t, (t + 1) % num_experts] = 4.0

    out = experts.forward(hidden_states, router_logits)
    assert out.shape == (tokens, hidden)
    assert out.dtype == hidden_states.dtype

    probs = F.softmax(router_logits.float(), dim=-1)
    top2, idx = probs.topk(2, dim=-1)
    top2 = top2 / top2.sum(-1, keepdim=True)
    expected = torch.zeros(tokens, hidden)
    for t in range(tokens):
        for k in range(2):
            e = int(idx[t, k])
            gate, up = F.linear(hidden_states[t].float(), experts.gate_up_proj[e]).chunk(2, dim=-1)
            ye = F.linear(F.silu(gate) * up, experts.down_proj[e])
            expected[t] += ye * top2[t, k]
    torch.testing.assert_close(out.float(), expected, atol=1e-5, rtol=1e-4)


def test_mtp_experts_all_zero_router_logits_still_normalizes():
    """Degenerate router (every expert tied): renormalize must not divide by zero or
    silently drop mass -- each selected expert should land at weight 1/top_k."""
    experts = _MTPExperts(num_experts=4, top_k=2, hidden_size=3, intermediate_size=3, renormalize=True)
    experts.gate_up_proj = torch.zeros(4, 6, 3)
    experts.down_proj = torch.zeros(4, 3, 3)
    router_logits = torch.zeros(2, 4)
    out = experts.forward(torch.randn(2, 3), router_logits)
    assert torch.isfinite(out).all()


def test_hidden_side_fusion_is_cpu_safe_up_to_fc_hidden(toy_config):
    """The half of Qwen4ExpMTPHead.fuse_inputs that only touches GroupedPlusOneRMSNorm
    (CPU fallback) and GatedResidual.mix on a non-cuda tensor (also CPU fallback) --
    i.e. everything except the GemmaPlusOneRMSNorm-based embedding branch, which has no
    CPU kernel (see this file's module docstring)."""
    head = Qwen4ExpMTPHead(
        toy_config, layer_id=toy_config.num_layers, rope_theta=toy_config.qwen4_args.mtp_rope_theta
    )
    for tensor in head.pre_fc_norm_hidden.state_dict().values():
        tensor.normal_(0, 0.05)
    for tensor in head.hyper_connection_mixer.state_dict().values():
        tensor.normal_(0, 0.05)
    for tensor in head.fc_hidden.state_dict().values():
        tensor.normal_(0, 0.05)

    prev_r = torch.randn(5, HCH)
    rn = head.pre_fc_norm_hidden.forward(prev_r)
    assert rn.shape == prev_r.shape
    h_mix, s = head.hyper_connection_mixer.mix(rn)
    assert h_mix.shape == (5, H)
    assert s is None  # use_combine=False
    h = head.fc_hidden.forward(h_mix)
    assert h.shape == (5, H)
    assert torch.isfinite(h).all()
