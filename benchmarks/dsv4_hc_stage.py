"""Standalone micro-harness for DSV4's hyper-connection stage.

Replays a DSV4-Flash-shaped chain -- 43 blocks x 2 hyper-connection sites plus the
output head's collapse -- with the sublayers stubbed out, so the only thing measured is
the hyper-connection tail itself: the mix gemv, ``inv_rms``, the Sinkhorn, the two
combines and the sublayer input RMSNorms.

Two compositions run over the same weights:

  reference   what ``FREETOKEN_UNFUSED_HC=1`` restores -- ``hc_post_combine`` ->
              ``x.float()`` -> ``inv_rms`` -> ``F.linear`` -> ``* rsqrt`` ->
              ``hc_split_sinkhorn`` -> ``hc_pre_combine`` -> ``rms_norm``.
  fused       ``kernel/triton/dsv4/hc_fused.hc_stage``, one launch per site, with the
              pending re-expand of the previous sublayer folded in.

Reports, per token: kernel launches (counted with the CUDA profiler over an eager pass),
CUDA-graph replay cost over 200 replays, and the drift between the two chains' final
hidden state after 43 layers of accumulation.

    python benchmarks/dsv4_hc_stage.py [--layers 43] [--tokens 1] [--replays 200]

Weights are ~1.6 MB per site, so a 43-layer run holds ~140 MB on the GPU.
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F
from torch.autograd.profiler_util import DeviceType

from freetoken.kernel.triton.dsv4.hc import hc_post_combine, hc_pre_combine
from freetoken.kernel.triton.dsv4.hc_fused import hc_head_stage, hc_stage
from freetoken.kernel.triton.dsv4.norm import inv_rms, rms_norm
from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn

HC = 4
DIM = 4096
HCD = HC * DIM
MIX = (2 + HC) * HC
EPS = 1e-6
ITERS = 20
DEV = "cuda"


class Site:
    """One hyper-connection site's parameters (attn or ffn half of a block)."""

    def __init__(self, mix: int, gen):
        self.fn = torch.randn(mix, HCD, generator=gen, device=DEV, dtype=torch.float32) * 0.02
        self.scale = (torch.rand(3, generator=gen, device=DEV) + 0.5).contiguous()
        self.base = torch.randn(mix, generator=gen, device=DEV) * 0.1
        self.w = torch.randn(DIM, generator=gen, device=DEV, dtype=torch.bfloat16)


def build(layers: int, seed: int = 0):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    sites = [Site(MIX, gen) for _ in range(2 * layers)]
    head = Site(HC, gen)
    return sites, head


# --------------------------------------------------------------------------- reference


def _ref_pre(stream, s, tokens):
    """inv_rms -> mix gemv -> scale -> Sinkhorn -> collapse -> the sublayer's RMSNorm."""
    mixes = F.linear(stream.float(), s.fn) * inv_rms(stream, EPS)
    pre, post, comb = hc_split_sinkhorn(mixes, s.scale, s.base, HC, ITERS, EPS)
    y = hc_pre_combine(stream.view(tokens, HC, DIM), pre, stream.dtype)
    return rms_norm(y, s.w, EPS), post, comb


def ref_chain(x, sites, head, tokens):
    stream = x
    for s in sites:
        residual = stream
        y, post, comb = _ref_pre(stream, s, tokens)
        a = y  # sublayer stub: attention / MoE are not what this harness measures
        stream = hc_post_combine(a, residual.view(tokens, HC, DIM), post, comb).view(tokens, HCD)
    mixes = F.linear(stream.float(), head.fn) * inv_rms(stream, EPS)
    pre = torch.sigmoid(mixes * head.scale[:1] + head.base) + EPS
    return rms_norm(hc_pre_combine(stream.view(tokens, HC, DIM), pre, stream.dtype), head.w, EPS)


# ------------------------------------------------------------------------------- fused


def fused_chain(x, sites, head, tokens):
    stream, pending = x, None
    for s in sites:
        stream, y, post, comb = hc_stage(
            stream, pending, s.fn, s.scale, s.base, hc_mult=HC, sinkhorn_iters=ITERS,
            hc_eps=EPS, norm_eps=EPS, norm_weight=s.w, tokens=tokens, dim=DIM,
        )
        pending, stream = (y, stream, post, comb), None  # sublayer stub: a = y
    return hc_head_stage(
        stream, pending, head.fn, head.scale[:1].contiguous(), head.base, hc_mult=HC,
        hc_eps=EPS, norm_eps=EPS, norm_weight=head.w, tokens=tokens, dim=DIM,
    )


# ------------------------------------------------------------------------------ metrics


def count_launches(call) -> int:
    """Kernel launches in one eager pass, the way the CUPTI decode profile counts them."""
    call()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    return sum(
        e.count for e in prof.key_averages()
        if e.device_type == DeviceType.CUDA and e.self_device_time_total > 0
    )


def graph_ms(call, replays: int) -> float:
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    torch.cuda.synchronize()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    # The GPU may be shared with a live server, so the minimum over rounds is the only
    # statistic that means anything -- the mean measures the neighbours.
    best = float("inf")
    for _ in range(5):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(replays):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / replays)
    del g
    torch.cuda.synchronize()
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--replays", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1

    sites, head = build(args.layers, args.seed)
    x = torch.randn(args.tokens, HCD, device=DEV, dtype=torch.bfloat16)
    n = args.tokens

    ref = ref_chain(x, sites, head, n)
    fus = fused_chain(x, sites, head, n)
    # Control: nudge one input element PER TOKEN by one bf16 step and re-run the
    # *reference*. A stack of hyper-connections is a chaotic map, so this is the
    # yardstick the fused-vs-reference drift has to be read against -- anything at or
    # below it is the chain's own conditioning, not a defect in the kernel. It has to be
    # per token because tokens do not mix: perturbing only row 0 would leave every other
    # row identical and understate the floor.
    xp = x.clone()
    col = xp[:, 0]
    col.copy_((col.view(torch.int16) + 1).view(torch.bfloat16))
    ctl = ref_chain(xp, sites, head, n)
    torch.cuda.synchronize()

    def _drift(a, b):
        d = (a.float() - b.float()).abs()
        rel = (d.pow(2).sum().sqrt() / b.float().pow(2).sum().sqrt()).item()
        return rel, (a == b).float().mean().item(), d.max().item()

    rel, eq, dmax = _drift(fus, ref)
    c_rel, c_eq, c_max = _drift(ctl, ref)

    n_ref = count_launches(lambda: ref_chain(x, sites, head, n))
    n_fus = count_launches(lambda: fused_chain(x, sites, head, n))
    t_ref = graph_ms(lambda: ref_chain(x, sites, head, n), args.replays)
    t_fus = graph_ms(lambda: fused_chain(x, sites, head, n), args.replays)

    print(f"DSV4 hyper-connection chain: {args.layers} layers x 2 sites + head, "
          f"{n} token(s), {args.replays} graph replays")
    print()
    print(f"{'':14} {'launches':>10} {'ms/token':>10}")
    print(f"{'reference':14} {n_ref:10d} {t_ref:10.4f}")
    print(f"{'fused':14} {n_fus:10d} {t_fus:10.4f}")
    print(f"{'delta':14} {n_fus - n_ref:10d} {t_fus - t_ref:10.4f}")
    print(f"{'':14} {'':>10} {t_ref / max(t_fus, 1e-9):9.2f}x")
    print()
    print(f"drift after {2 * args.layers} sites (vs the reference chain)")
    print(f"{'':22} {'rel L2':>10} {'bit-equal':>10} {'max|d|':>10}")
    print(f"{'fused':22} {rel:10.3e} {eq * 100:9.2f}% {dmax:10.3e}")
    print(f"{'control: input +1ulp':22} {c_rel:10.3e} {c_eq * 100:9.2f}% {c_max:10.3e}")
    print("  (the control is the same reference chain with ONE input element moved by")
    print("   one bf16 step -- the chain's own conditioning, i.e. the noise floor)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
