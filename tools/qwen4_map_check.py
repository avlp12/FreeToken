#!/usr/bin/env python3
"""Dry-run coverage check of the qwen4_exp weight map against a checkpoint index.

For every key in model.safetensors.index.json, classify via
freetoken.models.qwen4_exp.weight.classify_key and verify:
  * zero UNKNOWN keys (pass condition),
  * every fusion group is complete (all parts present per layer).

Usage: python tools/qwen4_map_check.py [/root/models/Qwen3.8-Flash-Next-FP8]
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from freetoken.models.qwen4_exp.weight import _FUSIONS, classify_key  # noqa: E402


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/root/models/Qwen3.8-Flash-Next-FP8"
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as fh:
        weight_map = json.load(fh)["weight_map"]

    counts: Counter[str] = Counter()
    unknown: list[str] = []
    # fused target key -> set of part-suffix indices seen
    fusion_seen: dict[str, set[int]] = defaultdict(set)
    fusion_expected: dict[str, int] = {}

    for raw in sorted(weight_map):
        category, detail = classify_key(raw)
        counts[category] += 1
        if category == "unknown":
            unknown.append(raw)
        elif category == "dense":
            for fused_suffix, parts in _FUSIONS.items():
                for idx, part in enumerate(parts):
                    if detail.endswith(part):
                        target = detail[: -len(part)] + fused_suffix
                        fusion_seen[target].add(idx)
                        fusion_expected[target] = len(parts)

    incomplete = {
        target: sorted(set(range(fusion_expected[target])) - seen)
        for target, seen in fusion_seen.items()
        if len(seen) != fusion_expected[target]
    }

    print(f"total keys      : {len(weight_map)}")
    for cat in ("dense", "expert_bank", "ngram_module", "dropped", "unknown"):
        print(f"  {cat:14s}: {counts.get(cat, 0)}")
    print(f"fusion groups   : {len(fusion_seen)} (incomplete: {len(incomplete)})")
    for target, missing in sorted(incomplete.items())[:10]:
        print(f"  INCOMPLETE {target}: missing part indices {missing}")
    print(f"unknown keys    : {len(unknown)}")
    for k in unknown[:20]:
        print(f"  UNKNOWN {k}")

    ok = not unknown and not incomplete
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
