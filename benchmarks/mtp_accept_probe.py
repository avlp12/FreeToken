#!/usr/bin/env python3
"""Depth-1 MTP acceptance-rate probe for Qwen3.8-Flash-Next -- the ONE number this
speculative-decode question turns on. No speculative loop here; see "Scope" below.

=====================================================================================
Why this number, and why it decides the feature (put here per the assignment, not
buried in a commit message)
=====================================================================================

The advisor's measured per-position unit cost for a depth-1 speculative *verify* step on
this model/hardware (RTX 5090, offload MoE) is 1.42 units (a normal decode step = 1 unit;
verifying a 2-position block costs more than a plain 1-token decode because of the extra
QSA/offload-MoE traffic for the second position, even though both positions run in one
batched forward). Call the true (unknown, to be measured) per-position acceptance
probability ``p`` -- the fraction of positions where the MTP draft head's greedy top-1
agrees with what the target model actually decodes next.

For a depth-1 draft, one verify step covers ``1 + p`` expected accepted output positions
(the free bonus token when the draft matches, else you fall back to the target's own
token). So the expected speedup over plain decoding is::

    speedup(p) = (1 + p) / 1.42

    p = 0.60 -> 1.13x   (worth shipping)
    p = 0.50 -> 1.06x   (marginal)
    p = 0.40 -> 0.99x   (breaks even at 0.42; below that, MTP is a net loss)

So: **p < 0.42 means depth-1 MTP is not worth building**, full stop, regardless of how
elegant the draft head is. This script exists to produce that one number (per workload --
this project has a documented history of a low-entropy workload inflating a decode number
by 7%, so prose and code are measured and reported SEPARATELY, never pooled into one
figure that could hide a workload-dependent flip).

=====================================================================================
Scope (read before extending this file)
=====================================================================================

- **No speculative decoding loop.** This does not draft, verify, accept/reject, or
  change what tokens get emitted. The target model runs completely normal greedy
  (temperature=0) decoding, unmodified and unaffected by anything in this file except a
  read-only hook that copies (never mutates) one intermediate tensor per forward call.
- **Retrospective scoring, not online drafting.** For depth-1 MTP, "did the draft match"
  and "did decode already commit the ground-truth token" are answerable from ONE normal
  decode trace: after generation finishes, feed the *entire* captured sequence of
  (base-model last-layer hidden state, actually-decoded next token) pairs through the MTP
  head IN ONE causal batch (mathematically identical to feeding it step-by-step with a
  growing KV cache -- causal attention does not care which), and check whether its
  prediction for each position matches the token the target *already* produced two steps
  ahead in its own trace. No drafting, branching, or rejection sampling needed to get this
  number.
- **Do not start `ft serve` / touch GPUs from this repo's automation.** This script uses
  the offline in-process ``freetoken.llm.LLM`` API (see ``tests/e2e/test_aime.py`` for the
  same pattern against a real checkpoint) -- no HTTP server, no subprocess. It must be
  invoked directly by a human/advisor on a machine with a free GPU; nothing in this
  repository should call it automatically.
- Needs ``FREETOKEN_LOAD_MTP=1`` (this script sets it itself, unconditionally, before
  importing ``freetoken`` -- see ``freetoken.models.qwen4_exp.weight.load_mtp_enabled``
  and ``.mtp.Qwen4ExpMTPHead``). Without it there is no ``model.mtp`` to score against and
  this script raises immediately with a clear message rather than silently reporting 0%.

=====================================================================================
Methodology
=====================================================================================

1. For each prompt, apply the checkpoint's chat template (thinking disabled -- a
   low-entropy `<think>` block would inflate the number the way this project has seen
   before) and run ONE normal greedy generation via ``LLM.generate``.
2. A read-only forward hook on the base model's *last* decoder layer
   (``model.model.layers.op_list[-1]``) clones that layer's output -- the raw
   hyper-connection residual R, *before* the base model's own final
   ``hyper_connection_mixer`` collapse -- for every forward call (prefill and every decode
   step), in order. Concatenated, this reconstructs R_i for every absolute position i the
   target actually processed. This is the exact hidden state ``Qwen4ExpMTPHead`` is
   built to consume; see that module's docstring for why.
3. ``full_ids = input_ids + output_ids`` (the complete, already-decoded sequence) gives
   the ground truth for free: position i's forward pass used R_i to pick ``full_ids[i+1]``
   (that is just normal greedy decoding, already done in step 1). Feeding
   ``(full_ids[i+1], R_i)`` through ``model.mtp`` for every position at once (one causal
   batch, positions ``0..len(R)-1``) predicts a token for position i+2; compare that
   against the target's OWN ``full_ids[i+2]`` from the SAME trace. Match rate over i is
   the estimator for p (greedy, temperature 0 -- this is not the general "sampling accept
   rate", only the depth-1-verify-under-greedy number the break-even math above uses).
4. The first ``--warmup-tokens`` *generated* positions of each prompt are dropped from
   the tally (KV/cache/routing warm-up noise, e.g. the first couple of tokens after a
   long templated prompt tend to be unrepresentative). Prompt/prefill positions are never
   scored at all -- only the generated continuation.
5. Reported per workload (prose, code) and pooled: sample count n, match rate p_hat,
   stddev, and a 95% Wilson confidence interval (safe at small n, unlike the normal
   approximation) -- never just a bare percentage.

Run (advisor, on a free GPU; this script never runs itself):

    FREETOKEN_LOAD_MTP=1 /root/ftenv/bin/python benchmarks/mtp_accept_probe.py \\
        --model /root/models/Qwen3.8-Flash-Next-NVFP4 --max-tokens 256 --warmup-tokens 4

Structural-only, no GPU (safe to run anywhere, including this executor's sandbox):

    /root/ftenv/bin/python benchmarks/mtp_accept_probe.py --help
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Sequence

# Break-even math from the module docstring: (1 + p) / VERIFY_UNIT_COST = 1 at p = BREAK_EVEN_P.
VERIFY_UNIT_COST = 1.42
BREAK_EVEN_P = VERIFY_UNIT_COST - 1.0  # 0.42

DEFAULT_MODEL = "/root/models/Qwen3.8-Flash-Next-NVFP4"

# Natural-prose prompts: open-ended, high-entropy continuations (instructions, explanation,
# narrative, translation) -- deliberately NOT code, NOT structured/templated output, so this
# workload does not accidentally share code's low-entropy token patterns (identifiers,
# indentation, closing brackets) that this project has previously seen inflate accept-style
# numbers by ~7%.
PROSE_PROMPTS: list[tuple[str, str]] = [
    ("prose_howare", "Hello! How has your day been so far? Tell me a little about it."),
    (
        "prose_explain",
        "Explain, in a few plain-English sentences, why the sky is blue. Assume I know "
        "no physics.",
    ),
    (
        "prose_story",
        "Write a short, whimsical paragraph about a cat who secretly runs a bakery at "
        "night.",
    ),
    (
        "prose_translate",
        "Translate the following into natural English: \"이 프로젝트는 "
        "추론 디코딩의 수락률을 측정한다.\"",
    ),
    (
        "prose_advice",
        "I'm nervous about a job interview tomorrow. Can you give me a few pieces of "
        "genuinely useful advice, in plain conversational language?",
    ),
    (
        "prose_summarize",
        "Summarize, in your own words and in two or three sentences, what a large "
        "language model is and how it is typically trained.",
    ),
]

# Code prompts: generation dominated by syntax, indentation, closing brackets and other
# low-entropy tokens -- reported separately because that low-entropy structure is exactly
# what previously inflated a decode-style number on this project.
CODE_PROMPTS: list[tuple[str, str]] = [
    (
        "code_fib",
        "Write a Python function `fib(n)` that returns the nth Fibonacci number "
        "iteratively. No explanation, just the code.",
    ),
    (
        "code_sort",
        "Write a Python implementation of quicksort as a function `quicksort(arr)` that "
        "returns a new sorted list. No explanation, just the code.",
    ),
    (
        "code_class",
        "Write a small Python class `Stack` with `push`, `pop`, and `peek` methods backed "
        "by a plain list. No explanation, just the code.",
    ),
    (
        "code_json",
        "Write a Python function that takes a nested dict and returns a flattened dict "
        "with dot-separated keys. No explanation, just the code.",
    ),
    (
        "code_regex",
        "Write a Python one-liner using `re` that extracts all email addresses from a "
        "string into a list. No explanation, just the code.",
    ),
    (
        "code_cpp",
        "Write a C++ function `int gcd(int a, int b)` computing the greatest common "
        "divisor iteratively. No explanation, just the code.",
    ),
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="checkpoint dir (NVFP4 build)")
    p.add_argument("--max-tokens", type=int, default=256, help="generation budget per prompt")
    p.add_argument(
        "--warmup-tokens", type=int, default=4,
        help="generated positions per prompt excluded from the tally (not prompt/prefill "
        "positions, which are never scored -- see module docstring)",
    )
    p.add_argument(
        "--workload", choices=["prose", "code", "both"], default="both",
        help="restrict to one prompt set (default: both, reported separately either way)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="also write the full report to this path")
    # Engine construction knobs, mirroring tests/e2e/test_aime.py's build_llm for this same
    # checkpoint family. Left unset (None) by default so freetoken's own model-family
    # defaults apply; this checkpoint (512 experts/layer, NVFP4, one RTX 5090) most likely
    # needs an offload MoE backend sized to fit -- pass --moe-cache-size the same way you
    # would to `ft serve` for this checkpoint if the engine does not auto-select one.
    p.add_argument("--attention-backend", default="auto")
    p.add_argument("--max-running-req", type=int, default=1)
    p.add_argument("--max-extend-tokens", type=int, default=8192)
    p.add_argument("--moe-backend", default=None)
    p.add_argument("--moe-cache-size", type=int, default=None)
    p.add_argument("--moe-cache-policy", default="lru")
    p.add_argument("--memory-ratio", type=float, default=0.9)
    p.add_argument("--max-seq-len-override", type=int, default=None)
    return p.parse_args(argv)


# ======================================================================================
# Stats
# ======================================================================================


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "(unknown: git rev-parse HEAD failed)"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95%-default Wilson score interval for a binomial proportion -- stable at the small
    n a handful of prompts gives you, unlike the normal approximation this project's past
    single-number reports have been (rightly) distrusted for omitting entirely."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return ((center - half) / denom, (center + half) / denom)


@dataclass
class StepResult:
    workload: str
    prompt_name: str
    position: int  # absolute position i in the target's own sequence (0-indexed)
    match: bool


def summarize(results: Sequence[StepResult]) -> dict[str, Any]:
    n = len(results)
    k = sum(1 for r in results if r.match)
    if n == 0:
        return {"n": 0, "accepted": 0, "p_hat": None, "stddev": None, "ci95": None}
    phat = k / n
    stddev = math.sqrt(phat * (1 - phat)) if n > 1 else 0.0
    lo, hi = wilson_ci(k, n)
    return {
        "n": n,
        "accepted": k,
        "p_hat": phat,
        "stddev": stddev,
        "ci95": [lo, hi],
        "speedup_at_p_hat": (1.0 + phat) / VERIFY_UNIT_COST,
    }


# ======================================================================================
# The capture hook (read-only: clones the base model's last-layer output, mutates nothing)
# ======================================================================================


class _CaptureLastLayer:
    """Installs on ``model.model.layers.op_list[-1]``: records a detached clone of every
    return value of that layer's ``forward`` (both prefill and each decode step, in call
    order), then returns the ORIGINAL result unchanged. Restores the original ``forward``
    on exit -- this is a read-only observer, it changes nothing about generation."""

    def __init__(self, model: Any) -> None:
        self._layer = model.model.layers.op_list[-1]
        self._orig = self._layer.forward
        self.chunks: list[Any] = []

    def __enter__(self) -> "_CaptureLastLayer":
        def wrapper(hidden, batch):
            out = self._orig(hidden, batch)
            self.chunks.append(out.detach().clone())
            return out

        self._layer.forward = wrapper
        return self

    def __exit__(self, *exc: Any) -> None:
        self._layer.forward = self._orig


# ======================================================================================
# Engine + per-prompt scoring
# ======================================================================================


def build_llm(args: argparse.Namespace):
    import torch
    from freetoken.llm import LLM

    kwargs: dict[str, Any] = dict(
        model_path=args.model,
        dtype=torch.bfloat16,
        attention_backend=args.attention_backend,
        max_running_req=args.max_running_req,
        max_extend_tokens=args.max_extend_tokens,
        # Correctness requirement for this probe, not a speed choice: a CUDA-graph-replayed
        # decode step never re-enters Python, so the capture hook above would silently miss
        # every decode step after the first. Always run eager. See _assert_eager below for
        # the corresponding runtime check.
        cuda_graph_bs=[],
        cuda_graph_max_bs=0,
    )
    if args.moe_backend is not None:
        kwargs["moe_backend"] = args.moe_backend
    if args.moe_cache_size is not None:
        kwargs["moe_cache_size"] = args.moe_cache_size
        kwargs["moe_cache_policy"] = args.moe_cache_policy
    kwargs["memory_ratio"] = args.memory_ratio
    if args.max_seq_len_override is not None:
        kwargs["max_seq_len_override"] = args.max_seq_len_override
    return LLM(**kwargs)


def _assert_eager(llm: Any) -> None:
    """Loud, specific failure instead of a silently-too-small sample: if CUDA graphs
    somehow got enabled anyway, most decode steps would bypass the capture hook and this
    would under-count almost every position past the first per prompt."""
    graph_runner = getattr(llm.engine, "graph_runner", None)
    max_graph_bs = getattr(graph_runner, "max_graph_bs", 0) if graph_runner is not None else 0
    if max_graph_bs:
        raise RuntimeError(
            f"CUDA graphs are enabled (max_graph_bs={max_graph_bs}) despite "
            "cuda_graph_max_bs=0/cuda_graph_bs=[]; the capture hook would silently miss "
            "most decode steps. Refusing to produce a number that could be wrong."
        )


def run_prompt(
    llm: Any, tokenizer: Any, workload: str, name: str, prompt: str, *, max_tokens: int, warmup: int
) -> list[StepResult]:
    import torch
    from freetoken.core import SamplingParams

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model = llm.engine.model
    with _CaptureLastLayer(model) as cap:
        llm.generate([text], SamplingParams(temperature=0.0, max_tokens=max_tokens))
    status = llm.status_map[0]
    full_ids = list(status.input_ids) + list(status.output_ids)
    if not cap.chunks:
        raise RuntimeError(f"{name}: capture hook recorded nothing -- did generation run at all?")
    r_all = torch.cat(cap.chunks, dim=0)

    # A normal (non-EOS-truncated) run captures len(full_ids) positions; hitting the
    # max_tokens budget can end one short (the loop stops before the final decode step
    # that would have produced one more token). Either is fine below. Anything much
    # smaller is the CUDA-graph-bypass failure mode _assert_eager is meant to catch
    # earlier -- this is a second, per-prompt tripwire in case that ever regresses.
    if r_all.shape[0] < len(full_ids) - 2:
        raise RuntimeError(
            f"{name}: captured {r_all.shape[0]} positions but the trace has "
            f"{len(full_ids)} tokens -- the capture hook likely missed decode steps "
            "(see _assert_eager)."
        )

    prompt_len = len(status.input_ids)
    n_eval = min(r_all.shape[0], len(full_ids) - 2)  # need full_ids[i+1] and [i+2] to exist
    first_generated_i = prompt_len - 1  # position whose forward predicts the FIRST generated token
    start = max(0, first_generated_i) + max(0, warmup)
    if start >= n_eval:
        return []  # prompt too short after warmup to say anything; caller just gets 0 rows

    idx = list(range(start, n_eval))
    device = r_all.device
    next_ids = torch.tensor([full_ids[i + 1] for i in idx], dtype=torch.long, device=device)
    positions = torch.tensor(idx, dtype=torch.long, device=device)
    prev_r = r_all[idx]

    with torch.inference_mode():
        logits = model.mtp.forward(next_ids, prev_r, positions, model.model.embed_tokens, model.lm_head)
    preds = logits.argmax(dim=-1).tolist()

    return [
        StepResult(workload=workload, prompt_name=name, position=i, match=(pred == full_ids[i + 2]))
        for i, pred in zip(idx, preds)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # Must be set before `import freetoken` triggers model-module import machinery;
    # weight.py/model.py read this at model-construction time via load_mtp_enabled().
    os.environ["FREETOKEN_LOAD_MTP"] = "1"

    import torch
    from transformers import AutoTokenizer

    torch.manual_seed(args.seed)

    llm = build_llm(args)
    if not hasattr(llm.engine.model, "mtp"):
        raise RuntimeError(
            "model.mtp was not constructed even with FREETOKEN_LOAD_MTP=1 -- "
            "see freetoken.models.qwen4_exp.model.Qwen4ExpForCausalLM.__init__"
        )
    _assert_eager(llm)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompt_sets: list[tuple[str, list[tuple[str, str]]]] = []
    if args.workload in ("prose", "both"):
        prompt_sets.append(("prose", PROSE_PROMPTS))
    if args.workload in ("code", "both"):
        prompt_sets.append(("code", CODE_PROMPTS))

    all_results: list[StepResult] = []
    per_prompt: dict[str, int] = {}
    for workload, prompts in prompt_sets:
        for name, prompt in prompts:
            rows = run_prompt(
                llm, tokenizer, workload, name, prompt,
                max_tokens=args.max_tokens, warmup=args.warmup_tokens,
            )
            all_results.extend(rows)
            per_prompt[name] = len(rows)

    by_workload = {
        workload: summarize([r for r in all_results if r.workload == workload])
        for workload, _ in prompt_sets
    }
    pooled = summarize(all_results)

    def verdict(stats: dict[str, Any]) -> str | None:
        if stats["p_hat"] is None:
            return None
        lo, hi = stats["ci95"]
        if lo > BREAK_EVEN_P:
            return f"p_hat CI clears break-even ({BREAK_EVEN_P}) -- worth building"
        if hi < BREAK_EVEN_P:
            return f"p_hat CI is entirely below break-even ({BREAK_EVEN_P}) -- not worth building"
        return f"p_hat CI straddles break-even ({BREAK_EVEN_P}) -- inconclusive, more samples needed"

    report = {
        "conditions": {
            "checkpoint": args.model,
            "commit": os.environ.get("FREETOKEN_PROBE_COMMIT") or _git_commit(),
            "temperature": 0.0,
            "max_tokens_per_prompt": args.max_tokens,
            "warmup_tokens_per_prompt": args.warmup_tokens,
            "prompt_count": {w: len(p) for w, p in prompt_sets},
            "verify_unit_cost": VERIFY_UNIT_COST,
            "break_even_p": BREAK_EVEN_P,
        },
        "by_workload": by_workload,
        "pooled": pooled,
        "verdict_by_workload": {w: verdict(s) for w, s in by_workload.items()},
        "verdict_pooled": verdict(pooled),
        "per_prompt_sample_counts": per_prompt,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
