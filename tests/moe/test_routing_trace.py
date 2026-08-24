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
  and the fp16 weight slots round-trip.

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
    # 43 layers x 4 slots x 6 ids x 2 bytes = 2064 B of payload per decode step.
    assert t.record_bytes == 8 + 2064
    t.close()


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

    expect = []  # per step: (actual, pred1, pred2, weights) as numpy
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
        for layer_id in range(N_LAYERS):
            hidden = hiddens[layer_id]
            # Exactly what MoE.forward does around its own router call.
            weights, indices = gates[layer_id](hidden, token)
            tracer.record(layer_id, hidden, token, weights, indices)
            step_actual.append(indices[0].cpu().numpy().astype(np.int64))
            step_w.append(weights[0].half().cpu().numpy())
            for ahead, sink in ((1, step_p1), (2, step_p2)):
                if layer_id + ahead < N_LAYERS:
                    _, pi = gates[layer_id + ahead](hidden, token)
                    sink.append(pi[0].cpu().numpy().astype(np.int64))
                else:
                    sink.append(np.full(TOP_K, -1, dtype=np.int64))
        expect.append(
            (np.stack(step_actual), np.stack(step_p1), np.stack(step_p2), np.stack(step_w))
        )
        tracer.after_step(batch_size=1)
    tracer.close()

    tr = load_trace(tracer.path)
    assert tr["n_steps"] == STEPS, f"expected {STEPS} records, got {tr['n_steps']}"
    assert tr["n_layers"] == N_LAYERS and tr["top_k"] == TOP_K
    assert tr["n_experts"] == N_EXPERTS
    assert list(tr["steps"]) == list(range(STEPS))

    for s, (a, p1, p2, w) in enumerate(expect):
        np.testing.assert_array_equal(tr["actual_ids"][s].astype(np.int64), a)
        np.testing.assert_array_equal(tr["pred_l1_ids"][s].astype(np.int64), p1)
        np.testing.assert_array_equal(tr["pred_l2_ids"][s].astype(np.int64), p2)
        np.testing.assert_allclose(tr["weights"][s].astype(np.float32), w.astype(np.float32))

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
    assert tracer.dropped == 0, f"{tracer.dropped} records dropped"
    print("micro-validation OK: layout, lookahead wiring and sentinels all verified")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        sys.exit(_child(sys.argv[2]))
    test_unset_env_leaves_no_hooks()
    print("ok: unset env leaves no hooks")
    test_header_geometry_is_self_describing()
    print("ok: header geometry (43 layers, top-6 -> 2064 B/step)")
    test_enabled_records_match_the_routers()
    print("ok: enabled trace matches the routers")
