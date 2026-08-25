"""Prefill routed-expert GEMM: batched GEMV vs decode-once + grouped bf16 GEMM.

Production geometry of the DeepSeek-V4 UD-Q2_K_XL expert banks:

    hidden H = 4096, moe_intermediate I = 2048, routed experts E = 256, top_k = 6
    gate_up  [E, 2I = 4096, pitch 1568]  IQ2_XS  (16 blocks/row, native 1184 B)
    down     [E, H  = 4096, pitch 784]   IQ3_XXS ( 8 blocks/row, native  784 B)

Reports, per chunk size, the wall time of one layer's two GEMMs on each path,
with the dequant-GEMM path split into dequant / GEMM / gather+scatter so the
crossover against the GEMV is attributable rather than just observed.

    python benchmarks/bench_prefill_dequant_gemm.py
    python benchmarks/bench_prefill_dequant_gemm.py --chunks 8192 --tiles 16

Measured on an RTX 5090 (sm120, CUDA 13.3, torch 2.11), ms per layer:

    chunk   vec N=8   tile 8   tile 16   tile 32   speedup @ tile 16
      512      19.4     33.2      29.6      27.8              0.65x
      768      26.7        -      29.9         -              0.90x
     1024      34.8        -      31.0         -              1.12x
     2048      64.7     36.3      30.2      29.1              2.14x
     8192     248.1     41.7      39.7      40.4              6.25x

The dequant is a FIXED ~14.3 ms -- it decodes all 256 experts regardless of how
many rows are routed -- which is the entire reason there is a crossover at all.
It sits at ~900 tokens, so ``FREETOKEN_PREFILL_DEQUANT_MIN_TOKENS`` defaults to
1024. Tile 32 wins by 1 ms at 2048 tokens and loses by 0.7 ms at 8192 for double
the scratch, so the default tile is 16.
"""

from __future__ import annotations

import argparse
import time

import torch

QK_K = 256
GGML_IQ2_XS = 17
GGML_IQ3_XXS = 18

H = 4096
I = 2048
E = 256
TOP_K = 6
GU_PITCH = 1568
DN_PITCH = 784


def make_bank(qtype, num_experts, nrows, ncols, pitch, seed):
    from freetoken.kernel.gguf import ggml_type_block_bytes

    blk = ggml_type_block_bytes(qtype)
    nb = ncols // QK_K
    native = nb * blk
    g = torch.Generator().manual_seed(seed)
    bank = torch.zeros(num_experts, nrows, pitch, dtype=torch.uint8)
    bank[:, :, :native] = torch.randint(
        0, 256, (num_experts, nrows, native), generator=g, dtype=torch.uint8
    )
    d = (0.01 + 0.04 * torch.rand(num_experts, nrows, nb, generator=g)).to(torch.float16)
    db = d.view(torch.uint8).reshape(num_experts, nrows, nb, 2)
    for b in range(nb):
        bank[:, :, b * blk : b * blk + 2] = db[:, :, b]
    return bank.cuda()


def timed(fn, iters=3, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, nargs="+", default=[512, 2048, 8192])
    ap.add_argument("--tiles", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    from freetoken.kernel.gguf import ggml_moe_a8_vec_batched
    from freetoken.kernel.triton.dsv4.fused_moe import fused_swiglu
    from freetoken.moe import prefill_dequant_gemm as dq
    from freetoken.moe.fused_q2_k_ud import _expert_group_perm

    print(f"gate_up [{E}, {2*I}, {GU_PITCH}] IQ2_XS   down [{E}, {H}, {DN_PITCH}] IQ3_XXS")
    gu = make_bank(GGML_IQ2_XS, E, 2 * I, H, GU_PITCH, 1)
    dn = make_bank(GGML_IQ3_XXS, E, H, I, DN_PITCH, 2)
    print(f"banks resident: {(gu.numel() + dn.numel()) / 2**30:.2f} GiB")

    for T in args.chunks:
        g = torch.Generator(device="cuda").manual_seed(9)
        x = (torch.randn(T, H, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
        ids = torch.randint(0, E, (T, TOP_K), generator=g, device="cuda", dtype=torch.int32)
        R = T * TOP_K
        print(f"\n=== chunk {T} tokens ({R} routed rows) ===")

        # ---- (i) the shipped batched-N=8 vec path ----
        def vec():
            perm = _expert_group_perm(ids, E, 8)
            gup = ggml_moe_a8_vec_batched(
                x, gu, ids, perm, TOP_K, GGML_IQ2_XS, 2 * I, T, GU_PITCH, 8
            )
            inter = fused_swiglu(gup, 7.0)
            del gup
            out = ggml_moe_a8_vec_batched(
                inter, dn, ids, perm, 1, GGML_IQ3_XXS, H, R, DN_PITCH, 8
            )
            del inter
            return out

        ms_vec = timed(vec, iters=args.iters)
        print(f"  vec batched N=8            {ms_vec:9.1f} ms")

        # ---- (ii) the dequant-GEMM path, per tile ----
        for tile in args.tiles:
            def run():
                plan = dq.RoutePlan(ids, E, tile)
                a = x.index_select(0, plan.order // TOP_K)
                gup = dq.grouped_expert_gemm(a, gu, GGML_IQ2_XS, plan)
                del a
                inter = fused_swiglu(gup, 7.0)
                del gup
                s = dq.grouped_expert_gemm(inter, dn, GGML_IQ3_XXS, plan)
                del inter
                out = torch.empty_like(s)
                out.index_copy_(0, plan.order, s)
                return out

            ms = timed(run, iters=args.iters)

            # component split: same work, isolated.
            plan = dq.RoutePlan(ids, E, tile)
            tiles = plan.tiles()

            def dequant_only():
                from freetoken.kernel.gguf import ggml_dequantize

                for e0, e1, _r0, _r1 in tiles:
                    for bank, qt, ncols in ((gu, GGML_IQ2_XS, H), (dn, GGML_IQ3_XXS, I)):
                        nb = dq.native_row_bytes(qt, ncols)
                        nr = bank.shape[1]
                        t = bank[e0:e1, :, :nb].contiguous()
                        w = ggml_dequantize(
                            t.view(-1, nb), qt, (e1 - e0) * nr, ncols, torch.bfloat16
                        )
                        del t, w

            ms_deq = timed(dequant_only, iters=args.iters)

            def perm_only():
                p = dq.RoutePlan(ids, E, tile)
                a = x.index_select(0, p.order // TOP_K)
                s = torch.empty((R, H), dtype=torch.bfloat16, device="cuda")
                out = torch.empty_like(s)
                out.index_copy_(0, p.order, s)
                del a, s
                return out

            ms_perm = timed(perm_only, iters=args.iters)
            ms_gemm = ms - ms_deq - ms_perm
            speed = ms_vec / ms
            print(
                f"  dequant-GEMM tile={tile:<3}      {ms:9.1f} ms   "
                f"(dequant {ms_deq:7.1f} | gemm {ms_gemm:7.1f} | perm {ms_perm:6.1f})   "
                f"{speed:5.2f}x"
            )


if __name__ == "__main__":
    main()
