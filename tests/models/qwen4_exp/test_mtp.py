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


# ======================================================================================
# GPU crash regression: Qwen4ExpMTPHead.forward's lm_head must be optional, and the
# probe's ctx-free bypass must be provably equivalent to the real, batch-aware
# ParallelLMHead.forward for how the probe actually calls it. See mtp_accept_probe.py's
# module docstring (Methodology, step 3) and _lm_head_logits for the narrative.
# ======================================================================================


def test_forward_lm_head_none_returns_hidden_and_skips_lm_head(toy_config):
    """Qwen4ExpMTPHead.forward(..., lm_head=None) must return the pre-lm_head hidden
    tensor without touching lm_head at all -- stubbing out the CUDA-only stages
    (fuse_inputs, the decoder layer) isolates just this branch on CPU."""
    with torch_dtype(torch.float32):
        head = Qwen4ExpMTPHead(
            toy_config, layer_id=toy_config.num_layers, rope_theta=toy_config.qwen4_args.mtp_rope_theta
        )
    for tensor in head.hyper_connection_mixer.state_dict().values():
        tensor.normal_(0, 0.05)

    fake_r0 = torch.randn(3, HCH)
    fake_r1 = torch.randn(3, HCH)
    head.fuse_inputs = lambda *a, **k: fake_r0
    head.layers.op_list[0].forward = lambda r0, positions: fake_r1
    expected_hidden, _ = head.hyper_connection_mixer.mix(fake_r1)

    dummy_ids = torch.zeros(3, dtype=torch.long)
    dummy_pos = torch.arange(3)

    hidden = head.forward(dummy_ids, fake_r0, dummy_pos, embed_tokens=None, lm_head=None)
    torch.testing.assert_close(hidden, expected_hidden)

    class _FakeLMHead:
        def __init__(self):
            self.calls = []

        def forward(self, x):
            self.calls.append(x)
            return "SENTINEL_LOGITS"

    fake_lm_head = _FakeLMHead()
    out = head.forward(dummy_ids, fake_r0, dummy_pos, embed_tokens=None, lm_head=fake_lm_head)
    assert out == "SENTINEL_LOGITS"
    assert len(fake_lm_head.calls) == 1
    torch.testing.assert_close(fake_lm_head.calls[0], expected_hidden)


def test_probe_lm_head_bypass_matches_the_real_forward_for_a_decode_batch():
    """The safety claim behind mtp_accept_probe.py's _lm_head_logits: for a decode-phase
    batch (one row per request, no prefill last-token gather to apply -- exactly the
    shape the probe always calls it with), the ctx-free bypass is byte-identical to the
    real, batch-aware ParallelLMHead.forward."""
    from freetoken.core import Batch
    from freetoken.layers.embedding import ParallelLMHead

    from .common import fresh_ctx

    probe = _load_probe_module()

    torch.manual_seed(0)
    vocab, hidden = 11, 6
    with torch_dtype(torch.float32):
        head = ParallelLMHead(num_embeddings=vocab, embedding_dim=hidden, tie_word_embeddings=False)
    head.weight.normal_()
    x = torch.randn(3, hidden)

    batch = Batch(reqs=[None, None, None], phase="decode")
    ctx = fresh_ctx()
    with ctx.forward_batch(batch):
        real = head.forward(x)

    bypass = probe._lm_head_logits(head, x)
    torch.testing.assert_close(bypass, real)


def test_probe_lm_head_bypass_would_diverge_on_a_mixed_prefill_batch():
    """Negative control: the bypass is only valid because the probe never feeds it a
    genuine multi-position-per-request prefill batch. If it did, the real
    ParallelLMHead.forward's last-token gather would select a strict subset of rows;
    this proves that difference is real (not hand-waved) and pins down exactly which
    rows -- so nobody "simplifies" the probe to call this on a raw prefill batch."""
    from types import SimpleNamespace

    from freetoken.core import Batch
    from freetoken.layers.embedding import ParallelLMHead

    from .common import fresh_ctx

    probe = _load_probe_module()

    torch.manual_seed(0)
    vocab, hidden = 5, 4
    with torch_dtype(torch.float32):
        head = ParallelLMHead(num_embeddings=vocab, embedding_dim=hidden, tie_word_embeddings=False)
    head.weight.normal_()
    x = torch.randn(4, hidden)  # e.g. 2 requests' prefill rows, 2 positions each

    batch = Batch(reqs=[None, None], phase="prefill")
    batch.attn_metadata = SimpleNamespace(get_last_indices=lambda bs: torch.tensor([1, 3]))
    ctx = fresh_ctx()
    with ctx.forward_batch(batch):
        real = head.forward(x)

    bypass = probe._lm_head_logits(head, x)
    assert real.shape[0] == 2
    assert bypass.shape[0] == 4
    torch.testing.assert_close(real, bypass[[1, 3]])
    assert not torch.allclose(real, bypass[:2])


def _load_probe_module():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "benchmarks" / "mtp_accept_probe.py"
    spec = importlib.util.spec_from_file_location("mtp_accept_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mtp_accept_probe"] = module
    spec.loader.exec_module(module)
    return module


# ======================================================================================
# Diagnostic sweep added after the first GPU run's pooled p_hat=0.163 landed far below an
# external comparable-model figure (83.4%, ik_llama.cpp PR #1698 depth-1 MTP): distinguish
# "wiring bug" from "genuinely weak head" by scoring every captured position against
# full_ids[i+1]/[i+2]/[i+3] and against several readings of the captured hidden state,
# instead of trusting the single (raw R, offset +2) cell the first run reported.
# ======================================================================================


def test_hidden_candidates_shapes_and_relationships(toy_config):
    """CPU-testable in full (unlike Qwen4ExpMTPHead.forward): GatedResidual.mix falls back
    to plain torch on a non-cuda tensor, so this exercises the real hc.py code, not a
    stub, for the 'collapsed_then_repeat' candidate."""
    probe = _load_probe_module()

    from freetoken.models.qwen4_exp.hc import GatedResidual

    torch.manual_seed(0)
    mixer = GatedResidual(toy_config, use_combine=False)
    for tensor in mixer.state_dict().values():
        tensor.normal_(0, 0.05)

    class _FakeBaseModel:
        def __init__(self, mixer):
            self.model = type("M", (), {"hyper_connection_mixer": mixer})()

    model = _FakeBaseModel(mixer)
    t = 5
    r_selected = torch.randn(t, HCH)

    cands = probe.hidden_candidates(model, r_selected, HC, H)
    assert set(cands) == {"raw_blocked", "stream_hidden_swapped", "collapsed_then_repeat"}

    # raw_blocked is exactly the input, unmodified.
    torch.testing.assert_close(cands["raw_blocked"], r_selected)

    # stream_hidden_swapped is a genuine reshuffle of the SAME values (same shape, same
    # total), not a no-op and not data loss.
    swapped = cands["stream_hidden_swapped"]
    assert swapped.shape == r_selected.shape
    torch.testing.assert_close(swapped.sum(), r_selected.sum())
    assert not torch.allclose(swapped, r_selected)

    # collapsed_then_repeat: repeat(1, hc_count) means every one of the hc_count blocks
    # along the last dim must be identical to each other.
    collapsed = cands["collapsed_then_repeat"]
    assert collapsed.shape == (t, HCH)
    blocks = collapsed.view(t, HC, H)
    for g in range(1, HC):
        torch.testing.assert_close(blocks[:, 0], blocks[:, g])
    # and it must actually be model.model.hyper_connection_mixer.mix(r_selected)[0]
    # repeated, not some other transform.
    expected, _ = mixer.mix(r_selected)
    torch.testing.assert_close(blocks[:, 0], expected)


def test_run_prompt_offset_and_hidden_candidate_sweep():
    """Mocked-engine regression for the sweep bookkeeping added on top of the original
    position-alignment logic (already covered by this project's earlier ad hoc mock, now
    folded in here): every (position, hidden_candidate, offset) cell that should exist
    does, out-of-range offsets near the end of the trace are skipped (not padded with
    wrong data), and -- using a fake MTP head that is deliberately "correct" only two
    steps ahead -- offset+2 is the one column that lands at 100% while +1 and +3 land at
    0%, exactly the "one offset jumps, the others don't" signature the sweep exists to
    surface, reproduced here as a known-answer case."""
    probe = _load_probe_module()

    HIDDEN = 4
    full_ids = [10, 11, 12, 20, 21, 22, 23, 24]  # prompt=[10,11,12], generated=[20..24]
    prompt_len = 3

    class FakeStatus:
        input_ids = full_ids[:prompt_len]
        output_ids = full_ids[prompt_len:]

    class FakeLayer:
        def __init__(self):
            self.forward = self._orig_forward

        def _orig_forward(self, hidden, batch):
            return hidden

    class FakeMixer:
        def mix(self, r):
            return r[:, : HIDDEN // 2], None  # arbitrary deterministic [T, hidden_size] collapse

    class FakeInnerModel:
        def __init__(self, layer, mixer):
            self.layers = type("L", (), {"op_list": [layer]})()
            self.embed_tokens = object()
            self.hyper_connection_mixer = mixer

    VOCAB = 40

    class FakeMTP:
        hc_count = 2
        hidden_size = HIDDEN // 2

        def forward(self, next_ids, prev_r, positions, embed_tokens, lm_head=None):
            assert lm_head is None  # run_prompt must call with lm_head=None, see mtp.py
            # "correct" exactly two steps ahead of next_ids, independent of prev_r on
            # purpose: this test is about the sweep's bookkeeping, not about
            # hidden_candidates()'s own numerics (covered separately above). Returns
            # "hidden" that is really already the desired logits -- FakeLMHead below is
            # the identity, so _lm_head_logits passes it through unchanged.
            preds = [int(nid) + 1 for nid in next_ids.tolist()]
            hidden = torch.full((len(preds), VOCAB), -10.0)
            for k, p in enumerate(preds):
                hidden[k, p] = 100.0
            return hidden

    class FakeLMHead:
        tied_embedding = None
        weight = torch.eye(VOCAB)
        bias = None

    class FakeModel:
        def __init__(self, layer, mixer):
            self.model = FakeInnerModel(layer, mixer)
            self.mtp = FakeMTP()
            self.lm_head = FakeLMHead()

    class FakeEngine:
        def __init__(self, model):
            self.model = model

    class FakeLLM:
        def __init__(self, model):
            self.engine = FakeEngine(model)
            self.status_map = {0: FakeStatus()}

        def generate(self, prompts, sampling_params):
            layer = self.engine.model.model.layers.op_list[0]
            prefill = torch.arange(3 * HIDDEN, dtype=torch.float32).reshape(3, HIDDEN)
            layer.forward(prefill, batch=None)
            for step in range(5):
                row = torch.full((1, HIDDEN), float(100 + step))
                layer.forward(row, batch=None)
            return [{"text": "", "token_ids": FakeStatus.output_ids}]

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
            return "<fake prompt>"

    layer = FakeLayer()
    model = FakeModel(layer, FakeMixer())
    llm = FakeLLM(model)
    tok = FakeTokenizer()

    rows = probe.run_prompt(llm, tok, "prose", "fake", "irrelevant", max_tokens=64, warmup=0)

    candidates_seen = {r.hidden_candidate for r in rows}
    assert candidates_seen == {"raw_blocked", "stream_hidden_swapped", "collapsed_then_repeat"}
    assert {r.offset for r in rows} == set(probe.OFFSETS)
    assert len(rows) == 3 * 12  # 3 candidates x (5 + 4 + 3) valid (position, offset) pairs

    # Out-of-range ground truth is skipped, never scored with wrong data.
    assert not any(r.position == 6 and r.offset in (2, 3) for r in rows)
    assert not any(r.position == 5 and r.offset == 3 for r in rows)

    for cand in candidates_seen:
        cand_rows = [r for r in rows if r.hidden_candidate == cand]
        off1 = [r for r in cand_rows if r.offset == 1]
        off2 = [r for r in cand_rows if r.offset == 2]
        off3 = [r for r in cand_rows if r.offset == 3]
        assert len(off1) == 5 and not any(r.match for r in off1)
        assert len(off2) == 4 and all(r.match for r in off2)
        assert len(off3) == 3 and not any(r.match for r in off3)
