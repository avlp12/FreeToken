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

External calibration point (why a low first result was NOT taken at face value): a
depth-1 MTP measurement on a comparable checkpoint (Qwen3.5-27B, ik_llama.cpp PR #1698,
2x3090) reported an **83.4%** accept rate. This project's own checkpoint's own draft head
should not land at 1/5 of that by being merely "weak" -- a weak head degrades gracefully
towards the base rate, it does not selectively land 5x below an architecturally similar
model's number while staying far above chance (~1/vocab). That gap is what motivated the
sweep below: distinguish "the wiring is wrong" from "this head is genuinely weak" before
concluding either.

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
  prediction for each position matches tokens the target *already* produced in its own
  trace (see the offset sweep below for "which token after" is no longer assumed).
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
   target actually processed.
3. ``full_ids = input_ids + output_ids`` (the complete, already-decoded sequence) gives
   the ground truth for free: position i's forward pass used R_i to pick ``full_ids[i+1]``
   (that is just normal greedy decoding, already done in step 1). Feeding
   ``(full_ids[i+1], R_i)`` through ``model.mtp`` predicts a token; step 3a/3b below cross
   it against several candidate ground truths and several candidate readings of R_i,
   because the first full run's result (pooled p_hat=0.163, see the executor's report) was
   far enough below an external comparable-model measurement (83.4%, see above) that a
   wiring bug is at least as likely as a genuinely weak head, and both hypotheses are
   cheap to check from data already captured in step 2 -- no regeneration needed for 3a,
   one more probe run for 3b (same generation, since it re-derives everything from the
   same greedy trace and is deterministic at temperature 0).

   3a. **Offset sweep.** The original run compared the prediction at position i only
       against ``full_ids[i+2]``. This reports ``full_ids[i+1]``, ``[i+2]`` and ``[i+3]``
       side by side, always (not behind a flag) -- this is the diagnostic, not an optional
       extra. Decision rule: if exactly one offset jumps to 0.6-0.85 while the others stay
       near the original number, the ground-truth index was off by one (or two) and the
       fix is a one-line offset change. If all three stay near the original number, the
       comparison target was not the bug -- move to 3b.
   3b. **Hidden-state candidate sweep.** ``mtp.pre_fc_norm_hidden.weight`` is ``[10240]``
       = hc_count(4) x hidden(2560) -- confirmed once, from the checkpoint's safetensors
       header, not re-derived here. This script's ``Qwen4ExpMTPHead`` assumes that width
       is fed the base model's raw, pre-collapse 4-stream R (see
       ``freetoken/models/qwen4_exp/mtp.py``'s module docstring for the full argument),
       but that is an inference with no reference implementation to check it against, so
       it is exactly the kind of assumption this sweep exists to stress-test rather than
       trust. ``hidden_candidates()`` below builds every alternative reading of R_i this
       script can construct purely from the SAME captured tensor (no new capture, no
       assumptions taken on faith):
         - ``raw_blocked``: R_i exactly as captured (current pipeline). By reading
           ``GroupedPlusOneRMSNorm``'s implementation (``hc.py``:
           ``xf.unflatten(-1, (num_groups, -1))``), the 10240-wide vector's groups are
           OUTER in that unflatten -- i.e. contiguous 2560-wide blocks, matching how
           ``Qwen4ExpModel.forward`` builds a fresh R via ``embed.repeat(1, hc_count)``
           (whole-hidden blocks, not interleaved). ``raw_blocked`` is exactly that layout,
           traced from the weight-consuming code, not assumed.
         - ``stream_hidden_swapped``: the opposite grouping order for the SAME numbers
           (re-split as if hidden were the outer/major axis and stream the inner/minor
           one) -- the "what if the [T, hc_count, hidden] vs [T, hidden, hc_count] axis
           order is backwards" hypothesis, made concrete instead of hand-waved.
         - ``collapsed_then_repeat``: apply the BASE model's own top-level
           ``hyper_connection_mixer.mix`` (which runs ITS OWN hc_norm as the first step)
           to collapse R_i to ``[T, hidden]``, then ``repeat(1, hc_count)`` it back out --
           mirroring ``Qwen4ExpModel.forward``'s own embedding-repeat pattern, and testing
           the "before/after the base model's own final norm+collapse" question directly:
           ``raw_blocked`` is the "before" reading, this is the "after" one.
       Reported per (hidden candidate) x (offset) x (workload), so a fix -- if the sweep
       finds one -- is visible as a single standout cell, not inferred from one number.

4. The first ``--warmup-tokens`` *generated* positions of each prompt are dropped from
   the tally (KV/cache/routing warm-up noise, e.g. the first couple of tokens after a
   long templated prompt tend to be unrepresentative). Prompt/prefill positions are never
   scored at all -- only the generated continuation. Warmup and start/end bounds are
   identical across every sweep cell (same ``idx`` positions scored everywhere) so cells
   are directly comparable.
5. Reported per (hidden candidate, offset), per workload (prose, code) and pooled: sample
   count n, match rate p_hat, stddev, and a 95% Wilson confidence interval (safe at small
   n, unlike the normal approximation) -- never just a bare percentage. A ``headline``
   section mirrors the pre-sweep report format (the ``raw_blocked`` / offset=+2 cell,
   i.e. exactly what the first run reported) for continuity.

Step 3 calls ``model.mtp.forward(..., lm_head=None)`` -- NOT with ``lm_head=model.lm_head``
-- and instead applies ``_lm_head_logits`` below. Reason: this script scores the MTP head
*after* the live engine forward call that generated each prompt has already returned, so
there is no ``get_global_ctx().batch`` for ``ParallelLMHead.forward`` to read (it asserts
``"No active batch in context"`` otherwise -- this is exactly the crash a first GPU run of
an earlier version of this script hit). ``_lm_head_logits`` reproduces the one line of
``ParallelLMHead.forward`` that is not batch-dependent; see its docstring for the read of
``freetoken/layers/embedding.py`` that justifies this is exact (not approximate) for how
this script calls it.

=====================================================================================
Reading the sweep's verdict (put here, not left to eyeballing a table)
=====================================================================================

Across every (hidden candidate, offset) pooled cell, this script reports the single
largest pooled p_hat found and a plain-language read of it:
  - >= 0.60: a wiring bug most likely explains the low baseline number -- some cell
    landed in the range an offset-by-one or hidden-state mixup would produce; go fix that
    specific cell's difference from ``raw_blocked``/offset+2 before touching the model.
  - >= 0.30 and < 0.60: ambiguous. Higher than chance and higher than the baseline, but
    not the clean jump an obvious wiring bug produces -- worth another look (e.g. the
    natural next candidate not covered here: tapping a layer other than the last one) but
    not conclusive either way.
  - < 0.30: the wiring hypotheses this sweep can cheaply test do not explain the gap --
    this is evidence (not proof; no reference implementation exists to fully rule
    everything out) that this checkpoint's MTP head is genuinely weak on this workload,
    which is a legitimate answer to the original question, not a failure of this probe.

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

# Ground-truth offsets scored against the SAME prediction at position i (full_ids[i+offset]).
# +2 is the original (and architecturally intended) target -- MTP(full_ids[i+1], R_i) should
# predict full_ids[i+2]. +1 and +3 are free diagnostic columns: an accidental off-by-one
# ground-truth index would show up as one of these jumping instead of +2.
OFFSETS: tuple[int, ...] = (1, 2, 3)
BASELINE_CANDIDATE = "raw_blocked"
BASELINE_OFFSET = 2

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
    hidden_candidate: str
    offset: int
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
# Hidden-state candidates (all derived from the ONE captured tensor -- see module
# docstring, step 3b, for what each one tests and why)
# ======================================================================================


def hidden_candidates(model: Any, r_selected: Any, hc_count: int, hidden_size: int) -> dict[str, Any]:
    """``{name: tensor[T, hc_count*hidden]}`` -- every alternative reading of the captured
    last-layer R this script can construct without a new capture pass. ``r_selected`` is
    already indexed down to the T rows being scored (all candidates are per-row/token
    independent -- GatedResidual.mix and the norms never mix across rows -- so selecting
    before or after computing a candidate gives identical results; done before, here, to
    keep every candidate's compute proportional to T instead of the full sequence)."""
    t = r_selected.shape[0]
    candidates: dict[str, Any] = {}

    # (1) Current pipeline: R exactly as captured. Layout traced, not assumed -- see
    # module docstring step 3b.
    candidates["raw_blocked"] = r_selected

    # (2) Opposite grouping-order hypothesis for the identical numbers.
    candidates["stream_hidden_swapped"] = (
        r_selected.view(t, hc_count, hidden_size)
        .transpose(1, 2)
        .reshape(t, hc_count * hidden_size)
        .contiguous()
    )

    # (3) "After the base model's own final norm+collapse" hypothesis: collapse via the
    # BASE model's own top-level hyper_connection_mixer (own weights, own hc_norm as
    # mix()'s first step -- distinct from mtp.hyper_connection_mixer), then re-expand via
    # repeat(1, hc_count), mirroring Qwen4ExpModel.forward's own embed-repeat pattern.
    collapsed, _ = model.model.hyper_connection_mixer.mix(r_selected)
    candidates["collapsed_then_repeat"] = collapsed.repeat(1, hc_count)

    return candidates


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


def _lm_head_logits(lm_head: Any, hidden: Any) -> Any:
    """``logits = hidden @ lm_head.weight.T (+ bias)`` -- the batch-context-free subset of
    ``ParallelLMHead.forward`` (``freetoken/layers/embedding.py``), for callers (this
    script) that run outside a live engine forward call and so have no
    ``get_global_ctx().batch`` for the real ``forward`` to read.

    Read of that method (as of this file's commit -- re-check if it changes):

        ctx = get_global_ctx(); batch = ctx.batch
        bs = batch.size
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
        module = self.tied_embedding or self
        logits = F.linear(x, module.weight, self.bias)
        if self.tp_size == 1:
            return logits
        ... (tp_size > 1 all-gather) ...

    Two things read the batch, neither applies here:
      - the ``is_prefill`` gather keeps only each request's LAST prefill position out of a
        multi-position-per-request batch. This script only ever calls the head with
        ``hidden`` already reduced to exactly the rows it wants scored (one per evaluated
        position) -- there is nothing to gather down further. (Proved wrong to skip in
        general, not just asserted: see tests/models/qwen4_exp/test_mtp.py's
        ``test_probe_lm_head_bypass_would_diverge_on_a_mixed_prefill_batch``, which
        constructs exactly that batch shape and shows the two DO diverge there -- this
        script just never produces that shape.)
      - the ``tp_size > 1`` all-gather. ``qwen4_exp/weight.py``'s ``iter_weights`` asserts
        TP=1 for this model family, so ``tp_size == 1`` always and that branch never runs.

    What is left is exactly the ``F.linear`` line, reproduced directly below so it runs
    with no batch context at all. See
    ``test_mtp.py::test_probe_lm_head_bypass_matches_the_real_forward_for_a_decode_batch``
    for a byte-identical-output proof against the real ``forward`` under a decode batch
    (the shape this script's usage actually looks like).
    """
    import torch.nn.functional as F

    module = lm_head.tied_embedding or lm_head
    return F.linear(hidden, module.weight, lm_head.bias)


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
    # Only full_ids[i+1] (fed into MTP as the token embedding) is a hard requirement here;
    # whether full_ids[i+offset] exists for each OFFSETS entry is checked per (i, offset)
    # below, so smaller offsets are not truncated just because +3 would run off the end.
    n_eval = min(r_all.shape[0], len(full_ids) - 1)
    first_generated_i = prompt_len - 1  # position whose forward predicts the FIRST generated token
    start = max(0, first_generated_i) + max(0, warmup)
    if start >= n_eval:
        return []  # prompt too short after warmup to say anything; caller just gets 0 rows

    idx = list(range(start, n_eval))
    device = r_all.device
    next_ids = torch.tensor([full_ids[i + 1] for i in idx], dtype=torch.long, device=device)
    positions = torch.tensor(idx, dtype=torch.long, device=device)
    r_selected = r_all[idx]

    hc_count = model.mtp.hc_count
    hidden_size = model.mtp.hidden_size
    rows: list[StepResult] = []
    with torch.inference_mode():
        for cand_name, prev_r in hidden_candidates(model, r_selected, hc_count, hidden_size).items():
            # lm_head=None: this runs after llm.generate() has already returned, outside
            # any live engine forward call, so there is no get_global_ctx().batch for
            # ParallelLMHead.forward to read. See _lm_head_logits and this file's module
            # docstring (Methodology, step 3) for why applying it ourselves is exact here.
            hidden = model.mtp.forward(next_ids, prev_r, positions, model.model.embed_tokens, lm_head=None)
            logits = _lm_head_logits(model.lm_head, hidden)
            preds = logits.argmax(dim=-1).tolist()
            for i, pred in zip(idx, preds):
                for offset in OFFSETS:
                    target_pos = i + offset
                    if target_pos >= len(full_ids):
                        continue
                    rows.append(
                        StepResult(
                            workload=workload, prompt_name=name, position=i,
                            hidden_candidate=cand_name, offset=offset,
                            match=(pred == full_ids[target_pos]),
                        )
                    )
    return rows


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
            # per_prompt counts the baseline cell only, so this stays comparable to the
            # pre-sweep report (every cell has the same position count per prompt anyway).
            per_prompt[name] = sum(
                1 for r in rows
                if r.hidden_candidate == BASELINE_CANDIDATE and r.offset == BASELINE_OFFSET
            )

    def verdict(stats: dict[str, Any]) -> str | None:
        if stats["p_hat"] is None:
            return None
        lo, hi = stats["ci95"]
        if lo > BREAK_EVEN_P:
            return f"p_hat CI clears break-even ({BREAK_EVEN_P}) -- worth building"
        if hi < BREAK_EVEN_P:
            return f"p_hat CI is entirely below break-even ({BREAK_EVEN_P}) -- not worth building"
        return f"p_hat CI straddles break-even ({BREAK_EVEN_P}) -- inconclusive, more samples needed"

    candidate_names = list(dict.fromkeys(r.hidden_candidate for r in all_results))
    sweep: dict[str, dict[str, Any]] = {}
    max_pooled_p_hat = 0.0
    max_pooled_cell = None
    for cand in candidate_names:
        for offset in OFFSETS:
            cell_key = f"{cand}|offset+{offset}"
            subset = [r for r in all_results if r.hidden_candidate == cand and r.offset == offset]
            by_workload = {
                workload: summarize([r for r in subset if r.workload == workload])
                for workload, _ in prompt_sets
            }
            pooled = summarize(subset)
            sweep[cell_key] = {
                "hidden_candidate": cand,
                "offset": offset,
                "by_workload": by_workload,
                "pooled": pooled,
                "verdict_by_workload": {w: verdict(s) for w, s in by_workload.items()},
                "verdict_pooled": verdict(pooled),
            }
            if pooled["p_hat"] is not None and pooled["p_hat"] > max_pooled_p_hat:
                max_pooled_p_hat = pooled["p_hat"]
                max_pooled_cell = cell_key

    if max_pooled_p_hat >= 0.60:
        sweep_reading = (
            f"max pooled p_hat={max_pooled_p_hat:.3f} at '{max_pooled_cell}' -- a wiring bug "
            "most likely explains the low baseline; go fix that cell's difference from "
            f"{BASELINE_CANDIDATE}|offset+{BASELINE_OFFSET} before touching the model."
        )
    elif max_pooled_p_hat >= 0.30:
        sweep_reading = (
            f"max pooled p_hat={max_pooled_p_hat:.3f} at '{max_pooled_cell}' -- ambiguous: "
            "above chance and above baseline, but not the clean jump an obvious wiring bug "
            "produces. Worth another look (e.g. a hidden-state tap from a layer other than "
            "the last one, not covered by this sweep) but not conclusive either way."
        )
    else:
        sweep_reading = (
            f"max pooled p_hat={max_pooled_p_hat:.3f} at '{max_pooled_cell}' -- the wiring "
            "hypotheses this sweep can cheaply test do not explain the gap to the external "
            "83.4% comparable-model figure. This is evidence (not proof) that this "
            "checkpoint's MTP head is genuinely weak on this workload."
        )

    baseline_key = f"{BASELINE_CANDIDATE}|offset+{BASELINE_OFFSET}"

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
            "offsets_scored": list(OFFSETS),
            "hidden_candidates_scored": candidate_names,
        },
        "headline": sweep.get(baseline_key),  # same cell the pre-sweep report used
        "sweep": sweep,
        "sweep_reading": sweep_reading,
        "max_pooled_p_hat": max_pooled_p_hat,
        "max_pooled_cell": max_pooled_cell,
        "per_prompt_sample_counts": per_prompt,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
