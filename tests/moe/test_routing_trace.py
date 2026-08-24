"""Micro-validation for the env-gated decode routing trace.

Two halves, because the tracer reads its environment variable exactly once at
import time:

* the PARENT process runs with ``FREETOKEN_ROUTING_TRACE`` unset and asserts the
  instrumentation is completely inert -- no tracer object, and a constructed
  DSV4 ``MoE`` carries no trace state at all;
* a CHILD process re-execs this file with the variable set, builds three real
  DSV4 ``Gate`` routers with random weights on the GPU, drives the trace
  machinery in EAGER mode exactly the way ``MoE.forward`` does, then parses the
  resulting file with ``/root/trace_reader.py`` and checks the record layout:
  actual ids match the layer's own router, ``pred_l1``/``pred_l2`` match the
  NEXT routers run on the SAME hidden, the trailing layers keep the -1 sentinel,
  and the fp16 weight AND score slots round-trip against an independent
  ``Gate(..., return_scores=True)`` call. It also drives a second, separate
  small hash-router trace (own ``RoutingTracer`` instance) to cover the "hash
  layers have no meaningful selection score" case the reader documents.

The child needs a GPU but uses only a few MB (dim=128, 16 experts).

CUDA-graph mode is deliberately NOT covered here: capturing the real decode
graph needs the loaded 400B checkpoint. That path is validated by booting the
server with ``FREETOKEN_ROUTING_TRACE`` set and running this reader over the
result (see the acceptance criteria in the assignment).

Run directly (``python tests/moe/test_routing_trace.py``) or under pytest.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRACE_READER_DIR = "/root"

DIM = 128
N_EXPERTS = 16
TOP_K = 4
N_LAYERS = 8
VOCAB = 64
STEPS = 16


def _ensure_tp():
    from freetoken.distributed.info import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(0, 1)


def _args():
    from freetoken.models.deepseek_v4.args import DeepseekV4Args

    return DeepseekV4Args(
        vocab_size=VOCAB,
        dim=DIM,
        n_layers=N_LAYERS,
        n_hash_layers=0,
        n_routed_experts=N_EXPERTS,
        n_activated_experts=TOP_K,
    )


# ----------------------------------------------------------------------------
# Parent half: the variable is unset, nothing may be instrumented.
# ----------------------------------------------------------------------------


def test_unset_env_leaves_no_hooks():
    assert "FREETOKEN_ROUTING_TRACE" not in os.environ or not os.environ[
        "FREETOKEN_ROUTING_TRACE"
    ], "run this test without FREETOKEN_ROUTING_TRACE set"
    from freetoken.moe import routing_trace

    assert routing_trace.get_tracer() is None

    from freetoken.models.deepseek_v4.moe import MoE

    _ensure_tp()
    with torch.device("meta"):
        moe = MoE(0, _args())
    # The only added state in the default configuration is the None sentinel the
    # forward guard tests; no layer id, no gate registration, no buffers.
    assert moe._routing_trace is None
    assert not hasattr(moe, "_routing_trace_layer_id")


def test_header_geometry_is_self_describing():
    from freetoken.moe import routing_trace as rt

    t = rt.RoutingTracer("/dev/null-unused")
    t.n_layers, t.top_k, t.n_experts = 43, 6, 256
    # 43 layers x 6 slots x 6 ids x 2 bytes = 3096 B of payload per decode step
    # (4 id/weight slots + 2 score slots added in format version 2).
    assert rt.FORMAT_VERSION == 2
    assert rt.N_SLOTS == 6
    assert t.record_bytes == 8 + 3096
    t.close()


def test_old_format_trace_still_loads():
    """A version-1 file (written before the score slots existed) must still
    parse: the reader keys its layout off the header's own ``version`` field,
    not off ``routing_trace``'s current constants."""
    sys.path.insert(0, TRACE_READER_DIR)
    import struct

    import numpy as np

    from trace_reader import load_trace

    n_layers, top_k, n_experts = 5, 3, 16
    n_slots_v1 = 4
    header = struct.Struct("<8s6I")
    record_header = struct.Struct("<IHH")
    record_bytes = record_header.size + n_layers * n_slots_v1 * top_k * 2

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "v1.bin")
        with open(path, "wb") as fh:
            fh.write(header.pack(b"FTROUTE1", 1, n_layers, top_k, n_experts, record_bytes, 0))
            payload = np.zeros((n_layers, n_slots_v1, top_k), dtype=np.int16)
            payload[:, 0, :] = 1  # actual ids
            fh.write(record_header.pack(0, 1, 0))
            fh.write(payload.tobytes())

        tr = load_trace(path)
        assert tr["version"] == 1
        assert tr["n_steps"] == 1
        assert tr["pred_l1_scores"] is None, "v1 files carry no score slots"
        assert tr["pred_l2_scores"] is None
        np.testing.assert_array_equal(tr["actual_ids"][0], np.ones((n_layers, top_k)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_enabled_records_match_the_routers():
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "trace")
        env = dict(os.environ)
        env["FREETOKEN_ROUTING_TRACE"] = prefix
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.join(REPO_ROOT, "python"), env.get("PYTHONPATH", "")]
        )
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", prefix],
            env=env,
            capture_output=True,
            text=True,
        )
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        assert proc.returncode == 0, "child validation failed"


# ----------------------------------------------------------------------------
# Child half: the variable is set, exercise the real record path in eager mode.
# ----------------------------------------------------------------------------


def _child(prefix: str) -> int:
    sys.path.insert(0, TRACE_READER_DIR)
    import numpy as np

    from freetoken.models.deepseek_v4.moe import Gate
    from freetoken.moe import routing_trace as rt
    from trace_reader import load_trace, summarize

    tracer = rt.get_tracer()
    assert tracer is not None, "FREETOKEN_ROUTING_TRACE was set but no tracer exists"
    assert tracer.path == prefix + ".bin"

    device = torch.device("cuda")
    args = _args()
    torch.manual_seed(0)

    # Three routers standing in for three consecutive MoE layers. Same module the
    # model uses, so the recorded prediction goes through the real scoring path
    # (sqrtsoftplus -> +bias -> top-k -> unbiased gather -> renorm -> route_scale).
    gates = []
    for layer_id in range(N_LAYERS):
        g = Gate(layer_id, args).to(device)
        with torch.no_grad():
            g.weight.copy_(torch.randn(N_EXPERTS, DIM, device=device).bfloat16())
            g.bias.copy_(torch.randn(N_EXPERTS, device=device) * 0.1)
        gates.append(g)
        tracer.register_gate(
            layer_id, g, n_layers=N_LAYERS, top_k=TOP_K, n_experts=N_EXPERTS
        )
    assert tracer.n_layers == N_LAYERS and tracer.top_k == TOP_K

    # A 5th router beyond n_layers (a drafter's) must not be registered.
    extra = Gate(N_LAYERS, args).to(device)
    tracer.register_gate(
        N_LAYERS, extra, n_layers=N_LAYERS, top_k=TOP_K, n_experts=N_EXPERTS
    )
    assert not tracer.traces_layer(N_LAYERS), "layer ids beyond n_layers must be ignored"

    tracer.arm()
    assert tracer.armed

    expect = []  # per step: (actual, pred1, pred2, weights, score1, score2) as numpy
    for _ in range(STEPS):
        token = torch.randint(0, VOCAB, (1,), device=device, dtype=torch.int64)
        # A stand-in for the residual stream: each layer's router sees a slightly
        # DIFFERENT hidden, the way it does in the real stack. That is what makes
        # this a real test -- feeding every layer the same tensor would make
        # pred_l1[L] equal actual[L+1] by construction and hide a mis-wiring.
        base = torch.randn(1, DIM, device=device)
        hiddens = [
            (base + 0.35 * torch.randn(1, DIM, device=device)).bfloat16()
            for _ in range(N_LAYERS)
        ]
        step_actual, step_p1, step_p2, step_w = [], [], [], []
        step_s1, step_s2 = [], []
        for layer_id in range(N_LAYERS):
            hidden = hiddens[layer_id]
            # Exactly what MoE.forward does around its own router call.
            weights, indices = gates[layer_id](hidden, token)
            tracer.record(layer_id, hidden, token, weights, indices)
            step_actual.append(indices[0].cpu().numpy().astype(np.int64))
            step_w.append(weights[0].half().cpu().numpy())
            for ahead, id_sink, score_sink in (
                (1, step_p1, step_s1),
                (2, step_p2, step_s2),
            ):
                if layer_id + ahead < N_LAYERS:
                    # Independent recompute: a SEPARATE call to the SAME gate the
                    # tracer's lookahead hook used, not a read-back of what the
                    # tracer wrote -- this is what would catch record() reading
                    # the wrong slot or the wrong tensor.
                    _, pi, psc = gates[layer_id + ahead](hidden, token, return_scores=True)
                    id_sink.append(pi[0].cpu().numpy().astype(np.int64))
                    score_sink.append(psc[0].half().cpu().numpy())
                else:
                    id_sink.append(np.full(TOP_K, -1, dtype=np.int64))
                    score_sink.append(np.full(TOP_K, np.nan, dtype=np.float16))
        expect.append(
            (
                np.stack(step_actual),
                np.stack(step_p1),
                np.stack(step_p2),
                np.stack(step_w),
                np.stack(step_s1),
                np.stack(step_s2),
            )
        )
        tracer.after_step(batch_size=1)
    tracer.close()

    tr = load_trace(tracer.path)
    assert tr["version"] == 2
    assert tr["n_steps"] == STEPS, f"expected {STEPS} records, got {tr['n_steps']}"
    assert tr["n_layers"] == N_LAYERS and tr["top_k"] == TOP_K
    assert tr["n_experts"] == N_EXPERTS
    assert list(tr["steps"]) == list(range(STEPS))
    assert tr["pred_l1_scores"] is not None and tr["pred_l2_scores"] is not None

    for s, (a, p1, p2, w, sc1, sc2) in enumerate(expect):
        np.testing.assert_array_equal(tr["actual_ids"][s].astype(np.int64), a)
        np.testing.assert_array_equal(tr["pred_l1_ids"][s].astype(np.int64), p1)
        np.testing.assert_array_equal(tr["pred_l2_ids"][s].astype(np.int64), p2)
        np.testing.assert_allclose(tr["weights"][s].astype(np.float32), w.astype(np.float32))
        got_s1 = tr["pred_l1_scores"][s].astype(np.float32)
        got_s2 = tr["pred_l2_scores"][s].astype(np.float32)
        # Present rows compare exactly (both sides round through the same fp16
        # bit pattern); absent rows are NaN on both sides and np.isnan-matched
        # separately since NaN != NaN under assert_allclose.
        present1, present2 = ~np.isnan(sc1), ~np.isnan(sc2)
        np.testing.assert_allclose(got_s1[present1], sc1.astype(np.float32)[present1])
        np.testing.assert_allclose(got_s2[present2], sc2.astype(np.float32)[present2])
        assert np.isnan(got_s1[~present1]).all(), "pred_l1_score missing its NaN sentinel"
        assert np.isnan(got_s2[~present2]).all(), "pred_l2_score missing its NaN sentinel"
        # Pre-renorm scores are non-negative by construction (softplus/sigmoid/
        # softmax are all >= 0), independent of the e-score bias's sign.
        assert (got_s1[present1] >= 0).all() and (got_s2[present2] >= 0).all()

    # The predictions must be the NEXT layers' routers, not this layer's: with
    # random weights an exact match on every layer would mean the wiring collapsed.
    a = tr["actual_ids"].astype(np.int64)
    p1 = tr["pred_l1_ids"].astype(np.int64)
    same = (np.sort(p1[:, : N_LAYERS - 1], -1) == np.sort(a[:, : N_LAYERS - 1], -1)).all(-1)
    assert not same.all(), "pred_l1 equals the layer's own routing -- wiring bug"

    # And the prediction must come from layer L's OWN hidden: since each layer here
    # sees a perturbed hidden, pred_l1[L] == actual[L+1] on every row would mean the
    # lookahead router had been fed layer L+1's input instead.
    p1_vs_next = (
        np.sort(p1[:, : N_LAYERS - 1], -1) == np.sort(a[:, 1:], -1)
    ).all(-1)
    assert not p1_vs_next.all(), "pred_l1[L] == actual[L+1] everywhere -- wrong hidden tapped"

    print("\n--- trace_reader summary over the micro-validation trace ---")
    rc = summarize(tr, per_layer=True)
    print("--- end summary ---")
    assert rc == 0, "trace_reader reported a sanity PROBLEM on a known-good trace"

    # Fresh device-buffer sentinels: the last two layers never write a prediction.
    assert (tr["pred_l1_ids"][:, N_LAYERS - 1 :] == -1).all()
    assert (tr["pred_l2_ids"][:, N_LAYERS - 2 :] == -1).all()
    # -1 int16 reads back as NaN in float16 -- the score slots share the id
    # slots' sentinel by construction (same buffer, same -1 fill), not by a
    # separate check in record().
    assert np.isnan(tr["pred_l1_scores"][:, N_LAYERS - 1 :]).all()
    assert np.isnan(tr["pred_l2_scores"][:, N_LAYERS - 2 :]).all()
    # Scores are pre-renorm (see the module docstring): `weights` is forced to
    # sum to `route_scale` every row (sqrtsoftplus is the default score_func,
    # so that renorm branch runs), but the recorded score is the numerator
    # BEFORE that division, so its row sums vary instead of clustering on
    # `route_scale`. Assert both halves directly rather than just trusting the
    # docstring's claim.
    route_scale = _args().route_scale
    w_row_sum = tr["weights"][:, : N_LAYERS - 1].astype(np.float32).sum(axis=-1)
    np.testing.assert_allclose(w_row_sum, route_scale, atol=0.02)
    s_row_sum = tr["pred_l1_scores"][:, : N_LAYERS - 1].astype(np.float32).sum(axis=-1)
    assert not np.allclose(s_row_sum, route_scale, atol=0.02), (
        "pred_l1_score row sums cluster on route_scale like `weights` does -- "
        "looks like the renormalized weight was recorded instead of the "
        "pre-renorm score"
    )
    assert tracer.dropped == 0, f"{tracer.dropped} records dropped"
    print("micro-validation OK: layout, lookahead wiring and sentinels all verified")

    _child_hash_layers(prefix + "_hash")
    return 0


def _child_hash_layers(prefix: str) -> None:
    """Cover the hash-router case the reader documents but the main loop above
    does not exercise (``_args()`` there uses ``n_hash_layers=0``): a hash
    layer's indices come from a token-id lookup table, not from any score, so
    its recorded "score" is just ``original_scores`` gathered at whatever
    expert the hash picked -- not a selection confidence, and not expected to
    be sorted. Uses its OWN ``RoutingTracer`` instance (like
    ``test_header_geometry_is_self_describing`` does) rather than the
    env-installed singleton, so it does not disturb the main trace above.
    """
    import numpy as np

    from freetoken.models.deepseek_v4.args import DeepseekV4Args
    from freetoken.models.deepseek_v4.moe import Gate
    from freetoken.moe import routing_trace as rt
    from trace_reader import load_trace

    dim, n_experts, top_k, n_layers, vocab, n_hash, steps = 32, 8, 2, 4, 16, 2, 6
    device = torch.device("cuda")
    args = DeepseekV4Args(
        vocab_size=vocab,
        dim=dim,
        n_layers=n_layers,
        n_hash_layers=n_hash,
        n_routed_experts=n_experts,
        n_activated_experts=top_k,
    )
    torch.manual_seed(1)
    gates = []
    for layer_id in range(n_layers):
        g = Gate(layer_id, args).to(device)
        with torch.no_grad():
            g.weight.copy_(torch.randn(n_experts, dim, device=device).bfloat16())
            if g.hash:
                # Deterministic token -> expert-set table, distinct per layer.
                g.tid2eid.copy_(
                    torch.randint(0, n_experts, (vocab, top_k), device=device, dtype=torch.int64)
                )
            else:
                g.bias.copy_(torch.randn(n_experts, device=device) * 0.1)
        gates.append(g)

    tracer = rt.RoutingTracer(prefix)
    for layer_id in range(n_layers):
        tracer.register_gate(layer_id, gates[layer_id], n_layers=n_layers, top_k=top_k, n_experts=n_experts)
    tracer.arm()

    # Layer 1 is a hash router (n_hash_layers=2): its predicted (from layer 0's
    # record) ids come from tid2eid[token], not from any score comparison.
    expect_ids, expect_scores = [], []
    for _ in range(steps):
        token = torch.randint(0, vocab, (1,), device=device, dtype=torch.int64)
        base = torch.randn(1, dim, device=device)
        hiddens = [(base + 0.35 * torch.randn(1, dim, device=device)).bfloat16() for _ in range(n_layers)]
        for layer_id in range(n_layers):
            hidden = hiddens[layer_id]
            weights, indices = gates[layer_id](hidden, token)
            tracer.record(layer_id, hidden, token, weights, indices)
        # Independent recompute of layer 0's lookahead-at-layer-1 prediction: a
        # SEPARATE call to gates[1], the hash gate, on layer 0's hidden/token.
        _, pi, psc = gates[1](hiddens[0], token, return_scores=True)
        expect_ids.append(pi[0].cpu().numpy().astype(np.int64))
        expect_scores.append(psc[0].half().cpu().numpy())
        tracer.after_step(batch_size=1)
    tracer.close()

    tr = load_trace(tracer.path)
    assert tr["version"] == 2
    got_ids = tr["pred_l1_ids"][:, 0, :].astype(np.int64)
    got_scores = tr["pred_l1_scores"][:, 0, :].astype(np.float32)
    np.testing.assert_array_equal(got_ids, np.stack(expect_ids))
    np.testing.assert_allclose(got_scores, np.stack(expect_scores).astype(np.float32))
    # The predicted score must still be finite and non-negative even though it
    # played no role in the hash pick (it is `original_scores` gathered at
    # whatever expert tid2eid[token] happened to name) -- see the module
    # docstring's "hash layers have no meaningful selection score" note.
    assert np.isfinite(got_scores).all(), "hash-layer gathered score must still be finite"
    assert (got_scores >= 0).all(), "hash-layer gathered score must still be non-negative"

    print("hash-layer sub-check OK: gather-only score matches an independent recompute, "
          "stays finite/non-negative, no wiring error")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        sys.exit(_child(sys.argv[2]))
    test_unset_env_leaves_no_hooks()
    print("ok: unset env leaves no hooks")
    test_header_geometry_is_self_describing()
    print("ok: header geometry (43 layers, top-6 -> 3096 B/step, format v2)")
    test_old_format_trace_still_loads()
    print("ok: a version-1 (pre-score) trace still loads")
    test_enabled_records_match_the_routers()
    print("ok: enabled trace matches the routers")
