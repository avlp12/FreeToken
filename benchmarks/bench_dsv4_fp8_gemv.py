"""DSV4 decode FP8 projections: fused single-launch GEMV vs the unfused three-launch path.

Also compares ``wo_a``'s bf16 grouped einsum against the FP8 grouped GEMV that
``FREETOKEN_WO_A_FP8=1`` selects.

Two measurement hazards this script exists to avoid:

  * **CPU launch bound.** An eager Triton launch costs ~10-20 us of *host* time, more
    than any of these kernels take on the GPU, so a naive ``do_bench`` loop measures
    Python. Everything here is timed inside a CUDA graph -- which is also how the decode
    step actually runs.
  * **L2 resident.** An RTX 5090 has 128 MB of L2; replaying one 33 MB weight makes the
    GEMV cache-resident and reports ~3.9 TB/s. Decode streams ~4.4 GB/token, none of it
    resident, so each graph rotates over a ~640 MB bank of distinct weights.

Usage::

    python benchmarks/bench_dsv4_fp8_gemv.py            # shipped-vs-unfused table
    python benchmarks/bench_dsv4_fp8_gemv.py --sweep    # re-tune _FUSED_DECODE_CFG
    python benchmarks/bench_dsv4_fp8_gemv.py --wo-a     # wo_a bf16 vs fp8
"""

from __future__ import annotations

import argparse
import itertools

import torch

from freetoken.kernel.triton.dsv4.fp8_linear import (
    _decode_cfg,
    _fused_cfg,
    block_fp8_linear,
    fused_fp8_gemv,
    grouped_block_fp8_linear,
)
from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view

DEV = "cuda"
WORKING_SET = 640 << 20

# Every distinct (N, K) the DSV4-Flash decode step runs, and how often per token
# (43 layers; the indexer only exists on the 21 layers with compress_ratio == 4).
DECODE_SHAPES = [
    (1024, 4096, "wq_a", 43),
    (32768, 1024, "wq_b", 43),
    (512, 4096, "wkv", 43),
    (4096, 8192, "wo_b", 43),
    (2048, 4096, "shared w1/w3", 86),
    (4096, 2048, "shared w2", 43),
    (8192, 1024, "indexer wq_b", 21),
]
LAYERS = 43
O_GROUPS, O_LORA_RANK, WO_A_K = 8, 1024, 4096


def graph_bench(bufs, call, iters=30, rounds=5):
    """Median GPU us per call, over a graph that walks the whole weight bank once."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for w in bufs[:3]:
            call(w)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for w in bufs:
            call(w)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / (iters * len(bufs)))
    ts.sort()
    del g
    torch.cuda.empty_cache()
    return ts[len(ts) // 2]


def weight_bank(N, K):
    """A rotating bank of distinct FP8 weights, ~WORKING_SET bytes in total."""
    reps = max(8, min(320, -(-WORKING_SET // (N * K))))
    big = torch.randint(0, 120, (reps * N, K), dtype=torch.uint8, device=DEV)
    ws = [big[i * N:(i + 1) * N].view(torch.float8_e4m3fn) for i in range(reps)]
    sb = torch.randint(120, 134, (N // 128, K // 128), dtype=torch.uint8, device=DEV)
    x = torch.randn(1, K, device=DEV, dtype=torch.bfloat16)
    return ws, sb, x, big


def compare():
    print("| projection | N | K | /token | MB | unfused (3 launches) | fused (1) | speedup |")
    print("|---|---|---|---|---|---|---|---|")
    tot_old = tot_new = 0.0
    for N, K, name, cnt in DECODE_SHAPES:
        ws, sb, x, big = weight_bank(N, K)
        old = graph_bench(ws, lambda w: block_fp8_linear(x, w, sb, fused=False))
        new = graph_bench(ws, lambda w: block_fp8_linear(x, w, sb, fused=True))
        mb = N * K / 1e6
        print(f"| {name} | {N} | {K} | {cnt} | {mb:.1f} | {old:.2f} us "
              f"({mb/old*1e3:.0f} GB/s) | {new:.2f} us ({mb/new*1e3:.0f} GB/s) "
              f"| {old/new:.2f}x |")
        tot_old += old * cnt
        tot_new += new * cnt
        del ws, big
        torch.cuda.empty_cache()
    n = sum(c for *_, c in DECODE_SHAPES)
    print(f"\nper token: {n} projections")
    print(f"  unfused  {3*n:4d} launches  {tot_old/1e3:.3f} ms")
    print(f"  fused    {1*n:4d} launches  {tot_new/1e3:.3f} ms   "
          f"(saved {(tot_old-tot_new)/1e3:.3f} ms, {2*n} launches)")


def sweep():
    print("_FUSED_DECODE_CFG candidates (best first per shape)\n")
    best = {}
    for N, K, name, cnt in DECODE_SHAPES + [
        (O_GROUPS * O_LORA_RANK, WO_A_K, "wo_a (grouped)", LAYERS)
    ]:
        grouped = name.startswith("wo_a")
        ws, sb, x, big = weight_bank(N, K)
        if grouped:
            x = torch.randn(O_GROUPS, K, device=DEV, dtype=torch.bfloat16)
        old = None if grouped else graph_bench(
            ws, lambda w: block_fp8_linear(x, w, sb, fused=False))
        rows = []
        for bn, sk, nw, ns in itertools.product(
            (2, 4, 8, 16, 32), (1, 2, 4, 8, 16, 32, 64), (1, 2), (1, 3)
        ):
            gr = O_LORA_RANK if grouped else N
            if (N % bn or gr % bn or sk > K // 128 or (K // sk) % 128
                    or bn * 128 < nw * 32 * 4):
                continue
            try:
                rows.append((graph_bench(ws, lambda w, c=(bn, sk, nw, ns): fused_fp8_gemv(
                    x, e4m3_kernel_view(w), sb, torch.bfloat16,
                    group_rows=O_LORA_RANK if grouped else None, cfg=c)), bn, sk, nw, ns))
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {(bn, sk, nw, ns)}: {exc!r:.80}")
        rows.sort()
        gb = N * K / 1e9
        print(f"=== {name}  N={N} K={K} ===")
        if old:
            print(f"    unfused                 {old:7.2f} us  {gb/old*1e6:6.0f} GB/s")
        for t, bn, sk, nw, ns in rows[:5]:
            sp = f"  {old/t:.2f}x" if old else ""
            print(f"    bn={bn:<3} sk={sk:<3} w={nw} st={ns}  {t:7.2f} us  "
                  f"{gb/t*1e6:6.0f} GB/s{sp}")
        best[(N, K)] = rows[0]
        del ws, big
        torch.cuda.empty_cache()
    print("\n_FUSED_DECODE_CFG = {")
    for (N, K), (t, bn, sk, nw, ns) in best.items():
        print(f"    ({N}, {K}): ({bn}, {sk}, {nw}, {ns}),")
    print("}")


def wo_a():
    x = torch.randn(1, 1, O_GROUPS, WO_A_K, device=DEV, dtype=torch.bfloat16)
    n_bf16 = max(2, WORKING_SET // (O_GROUPS * O_LORA_RANK * WO_A_K * 2))
    wb = [torch.randn(O_GROUPS, O_LORA_RANK, WO_A_K, device=DEV, dtype=torch.bfloat16)
          for _ in range(n_bf16)]
    t_bf16 = graph_bench(wb, lambda w: torch.einsum("bsgd,grd->bsgr", x, w).flatten(2))
    del wb
    torch.cuda.empty_cache()
    ws, sb, _x, big = weight_bank(O_GROUPS * O_LORA_RANK, WO_A_K)
    t_fp8 = graph_bench(ws, lambda w: grouped_block_fp8_linear(x, w, sb, O_LORA_RANK))
    mb = O_GROUPS * O_LORA_RANK * WO_A_K / 1e6
    print(f"wo_a bf16 einsum  {t_bf16:7.2f} us/layer  {2*mb:.1f} MB  "
          f"{2*mb/t_bf16*1e3:6.0f} GB/s  -> {t_bf16*LAYERS/1e3:.3f} ms/token")
    print(f"wo_a fp8  grouped {t_fp8:7.2f} us/layer  {mb:.1f} MB  "
          f"{mb/t_fp8*1e3:6.0f} GB/s  -> {t_fp8*LAYERS/1e3:.3f} ms/token")
    print(f"saving {(t_bf16-t_fp8)*LAYERS/1e3:.3f} ms/token ({t_bf16/t_fp8:.2f}x)")
    del ws, big


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="re-tune _FUSED_DECODE_CFG")
    ap.add_argument("--wo-a", action="store_true", help="wo_a bf16 einsum vs fp8 grouped")
    a = ap.parse_args()
    if a.sweep:
        sweep()
    elif a.wo_a:
        wo_a()
    else:
        compare()
        print()
        for (N, K, name, _c) in DECODE_SHAPES:
            print(f"  {name:<14} unfused cfg {_decode_cfg(N, K)}  "
                  f"fused cfg {_fused_cfg(N, K)}")
