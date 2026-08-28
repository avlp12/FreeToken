#!/usr/bin/env python3
"""Plan an FTW conversion for Qwen3.8-Flash-Next / qwen4_exp without writing weights.

Reads ``config.json`` + ``model.safetensors.index.json`` and safetensors *headers*
only (no 187 GiB data path). Safe to run on the conversion host.

    /root/ftenv/bin/python -m freetoken.checkpoint.ftw_qwen_dryrun \
        --model /root/models/Qwen3.8-Flash-Next-FP8 \
        --out /root/models/Qwen38FN-FP8-FTW-v1
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

# Allow `python path/to/ftw_qwen_dryrun.py` without installing the package.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from .qwen_layout import (  # type: ignore
        DEST_DENSE,
        DEST_EXPERT,
        DEST_NGRAM,
        DEST_SKIP,
        classify_tensor,
        disk_free_bytes,
        estimate_fp8_block_banks,
        estimate_ngram_table,
        extra_index_files,
        extra_safetensor_files,
        ftw_padded_bytes,
        is_qwen4_exp_config,
        is_wrapper_config,
        iter_shard_tensor_metas,
        load_hf_config_dict,
        load_json,
        load_weight_map,
        looks_like_ngram_shard,
        read_safetensors_header,
        tensor_nbytes_from_meta,
        unwrap_text_config,
    )
except ImportError:  # script path: python .../ftw_qwen_dryrun.py
    from qwen_layout import (  # type: ignore
        DEST_DENSE,
        DEST_EXPERT,
        DEST_NGRAM,
        DEST_SKIP,
        classify_tensor,
        disk_free_bytes,
        estimate_fp8_block_banks,
        estimate_ngram_table,
        extra_index_files,
        extra_safetensor_files,
        ftw_padded_bytes,
        is_qwen4_exp_config,
        is_wrapper_config,
        iter_shard_tensor_metas,
        load_hf_config_dict,
        load_json,
        load_weight_map,
        looks_like_ngram_shard,
        read_safetensors_header,
        tensor_nbytes_from_meta,
        unwrap_text_config,
    )

_GIB = 1 << 30
_MIB = 1 << 20


def _gib(n: int) -> str:
    return f" {n / _GIB:8.3f} GiB"


def _plan(model_path: str, out_dir: str, shard_limit: int) -> dict:
    cfg = load_hf_config_dict(model_path)
    text = unwrap_text_config(cfg)
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    index_meta = load_json(index_path).get("metadata", {}) if os.path.isfile(index_path) else {}
    weight_map = load_weight_map(model_path)
    extra_files = extra_safetensor_files(model_path, weight_map)
    extra_indexes = extra_index_files(model_path)

    # Classify every indexed tensor by *name* first; fill sizes from headers.
    dest_of = {name: classify_tensor(name) for name in weight_map}
    shards = sorted(set(weight_map.values()))
    sizes: dict[str, int] = {}
    shapes: dict[str, list] = {}
    dtypes: dict[str, str] = {}
    header_errors: list[str] = []
    try:
        for name, _shard, meta, nbytes in iter_shard_tensor_metas(model_path, shards):
            sizes[name] = nbytes
            shapes[name] = list(meta.get("shape") or [])
            dtypes[name] = str(meta.get("dtype") or "")
    except (OSError, ValueError) as exc:
        header_errors.append(str(exc))

    # Extra (unindexed) safetensors: treat ngram-named shards as ngram, else classify keys.
    extra_tensors: list[tuple[str, str, int, str]] = []
    for fname in extra_files:
        try:
            hdr = read_safetensors_header(os.path.join(model_path, fname))
        except (OSError, ValueError) as exc:
            header_errors.append(f"{fname}: {exc}")
            continue
        force_ngram = looks_like_ngram_shard(fname)
        for name, meta in hdr.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            nbytes = tensor_nbytes_from_meta(meta)
            dest = DEST_NGRAM if force_ngram else classify_tensor(name)
            extra_tensors.append((name, dest, nbytes, fname))
            dest_of[name] = dest
            sizes[name] = nbytes
            shapes[name] = list(meta.get("shape") or [])
            dtypes[name] = str(meta.get("dtype") or "")

    by_dest: dict[str, list[str]] = defaultdict(list)
    for name, dest in dest_of.items():
        by_dest[dest].append(name)

    payload = {d: sum(sizes.get(n, 0) for n in names) for d, names in by_dest.items()}
    padded = {d: sum(ftw_padded_bytes(sizes.get(n, 0)) for n in names) for d, names in by_dest.items()}

    layers = int(text.get("num_hidden_layers") or 0)
    experts = int(text.get("num_experts") or 0)
    hidden = int(text.get("hidden_size") or 0)
    moe_i = int(text.get("moe_intermediate_size") or 0)
    banks = estimate_fp8_block_banks(
        num_layers=layers, num_experts=experts, hidden_size=hidden, moe_intermediate=moe_i,
    ) if layers and experts and hidden and moe_i else None
    ngram_cfg = estimate_ngram_table(text)

    # FTW estimate: dense + ngram as original tensors; experts as packed banks (not raw).
    dense_pad = padded.get(DEST_DENSE, 0)
    ngram_pad = padded.get(DEST_NGRAM, 0)
    # If the index hid the 51 GiB table, fall back to the config estimate so the
    # disk verdict still sees the bytes that convert.py will pass through.
    if ngram_pad == 0 and ngram_cfg:
        ngram_pad = ftw_padded_bytes(ngram_cfg["weight"]) + ftw_padded_bytes(ngram_cfg["scale"])
        ngram_from = "config-estimate (no ngram keys in index)"
    else:
        ngram_from = "index/headers"
    expert_pad = banks["padded"] if banks else padded.get(DEST_EXPERT, 0)
    ftw_total = dense_pad + ngram_pad + expert_pad
    n_shards = max(1, (ftw_total + shard_limit - 1) // shard_limit) if ftw_total else 0

    free = disk_free_bytes(out_dir)
    # Source stays on disk (convert reads it); dest is additional. Need dest + ~1 shard
    # of headroom for the writer's current file.
    need = ftw_total + min(shard_limit, ftw_total)
    if free >= need + (8 << 30):
        verdict = "OK"
    elif free >= need:
        verdict = "TIGHT"
    else:
        verdict = "FAIL"

    return {
        "cfg": cfg,
        "text": text,
        "index_meta": index_meta,
        "weight_map": weight_map,
        "extra_files": extra_files,
        "extra_indexes": extra_indexes,
        "extra_tensors": extra_tensors,
        "dest_of": dest_of,
        "by_dest": by_dest,
        "sizes": sizes,
        "shapes": shapes,
        "dtypes": dtypes,
        "payload": payload,
        "padded": padded,
        "banks": banks,
        "ngram_cfg": ngram_cfg,
        "ngram_from": ngram_from,
        "dense_pad": dense_pad,
        "ngram_pad": ngram_pad,
        "expert_pad": expert_pad,
        "ftw_total": ftw_total,
        "n_shards": n_shards,
        "shard_limit": shard_limit,
        "free": free,
        "need": need,
        "verdict": verdict,
        "header_errors": header_errors,
        "out_dir": out_dir,
        "model_path": model_path,
    }


def _print_plan(p: dict) -> None:
    cfg, text = p["cfg"], p["text"]
    print("=== FTW convert plan (dry-run, headers only) ===")
    print(f"source : {p['model_path']}")
    print(f"out    : {p['out_dir']}")
    print(f"wrapper: {is_wrapper_config(cfg)}  model_type={cfg.get('model_type')!r}  "
          f"arch={cfg.get('architectures')}")
    print(f"text   : model_type={text.get('model_type')!r}  layers={text.get('num_hidden_layers')}  "
          f"experts={text.get('num_experts')}  hidden={text.get('hidden_size')}  "
          f"moe_i={text.get('moe_intermediate_size')}")
    print(f"qwen4  : {is_qwen4_exp_config(cfg)}  "
          f"quant={((cfg.get('quantization_config') or {}).get('quant_method'))}  "
          f"block={((cfg.get('quantization_config') or {}).get('weight_block_size'))}")
    src_total = int((p["index_meta"] or {}).get("total_size") or 0)
    print(f"index  : {len(p['weight_map'])} keys  metadata.total_size={_gib(src_total)}  "
          f"extra_shards={len(p['extra_files'])}  extra_indexes={p['extra_indexes']}")
    if p["extra_files"]:
        print("  extra safetensors:")
        for fn in p["extra_files"][:16]:
            print(f"    {fn}")
        if len(p["extra_files"]) > 16:
            print(f"    ... +{len(p['extra_files']) - 16} more")

    print("\n--- destination (by raw HF key) ---")
    for dest in (DEST_SKIP, DEST_EXPERT, DEST_NGRAM, DEST_DENSE):
        names = p["by_dest"].get(dest, [])
        print(f"  {dest:8s}  {len(names):7d} tensors  payload={_gib(p['payload'].get(dest, 0))}  "
              f"ftw_pad={_gib(p['padded'].get(dest, 0))}")

    print("\n--- skip prefixes (must NOT enter FTW) ---")
    skip_pref = Counter()
    for name in p["by_dest"].get(DEST_SKIP, []):
        if name.startswith("mtp."):
            skip_pref["mtp.*"] += 1
        elif name.startswith("model.visual.") or name.startswith("visual."):
            skip_pref["model.visual.*"] += 1
        else:
            skip_pref["other"] += 1
    for k, n in skip_pref.most_common():
        print(f"  {k:20s} {n}")

    print("\n--- n-gram keys (kind=ngram, original names, not concat) ---")
    print(f"  source: {p['ngram_from']}")
    ngram_names = sorted(p["by_dest"].get(DEST_NGRAM, []))
    if ngram_names:
        for name in ngram_names[:32]:
            print(f"  {name:72s} {_gib(p['sizes'].get(name, 0))}  "
                  f"{p['dtypes'].get(name, '?'):8s} {p['shapes'].get(name)}")
        if len(ngram_names) > 32:
            print(f"  ... +{len(ngram_names) - 32} more ngram tensors")
        odd = [n for n in ngram_names if p["sizes"].get(n, 0) < 100 * _MIB]
        if odd:
            print("  small/scale ngram siblings:")
            for name in odd:
                print(f"    {name:72s} {_gib(p['sizes'].get(name, 0))}  "
                      f"{p['dtypes'].get(name, '?'):8s} {p['shapes'].get(name)}")
    else:
        print("  (none classified from keys)")
        if p["ngram_cfg"]:
            ng = p["ngram_cfg"]
            print(f"  config fallback: vocab={ng['vocab']} hidden={ng['hidden']} "
                  f"parts={ng['parts']} payload={_gib(ng['payload'])}")

    print("\n--- dense keys >= 10 MiB (kind=weight) ---")
    dense_big = [
        n for n in p["by_dest"].get(DEST_DENSE, []) if p["sizes"].get(n, 0) >= 10 * _MIB
    ]
    dense_big.sort(key=lambda n: -p["sizes"].get(n, 0))
    for name in dense_big[:24]:
        print(f"  {name:72s} {_gib(p['sizes'][name])}  "
              f"{p['dtypes'].get(name, '?'):8s} {p['shapes'].get(name)}")
    if not dense_big:
        print("  (none)")

    print("\n--- non-expert prefix tree (first 5 components, dest != expert) ---")
    pref = Counter()
    for dest in (DEST_SKIP, DEST_NGRAM, DEST_DENSE):
        for name in p["by_dest"].get(dest, []):
            parts = name.split(".")
            pref[".".join(parts[:5])] += 1
    for k, n in pref.most_common(40):
        print(f"  {n:6d}  {k}")

    print("\n--- fp8_block expert banks (layer_sink streamed) ---")
    if p["banks"]:
        b = p["banks"]
        print(f"  entries: {b['n_entries']}  payload={_gib(b['payload'])}  "
              f"ftw_pad={_gib(b['padded'])}")
        for name, nbytes in b["per_bank"].items():
            print(f"    {name:16s} {_gib(nbytes)}")
    else:
        print("  (missing dims; cannot estimate)")

    print("\n--- FTW output estimate ---")
    print(f"  dense  (weight)        {_gib(p['dense_pad'])}")
    print(f"  ngram  (kind=ngram)    {_gib(p['ngram_pad'])}   [{p['ngram_from']}]")
    print(f"  experts(fp8_block)     {_gib(p['expert_pad'])}")
    print(f"  TOTAL                  {_gib(p['ftw_total'])}")
    print(f"  shards @ {p['shard_limit'] / _GIB:.1f} GiB   ~{p['n_shards']}")

    print("\n--- disk budget ---")
    print(f"  free at out fs : {_gib(p['free'])}")
    print(f"  need (out+head): {_gib(p['need'])}")
    print(f"  WSL note       : host reported 353 GiB free; this probe is the out filesystem")
    print(f"  verdict        : {p['verdict']}")

    if p["header_errors"]:
        print("\n--- header errors ---")
        for e in p["header_errors"]:
            print(f"  {e}")

    print("\n--- convert command (advisor) ---")
    print("  ft checkpoint --model {} --out {} --moe-backend offload --dtype bfloat16".format(
        p["model_path"], p["out_dir"],
    ))
    print("  (no --expert-gguf: Qwen ships native fp8 safetensors)")


def _selfcheck() -> None:
    cases = [
        ("mtp.layers.0.mlp.experts.0.gate_proj.weight", DEST_SKIP),
        # Vision is carried now (convert.py's dedicated pass-through renames it to
        # vision_tower.*), not dropped -- see qwen_layout.classify_tensor's docstring.
        ("model.visual.blocks.0.attn.qkv.weight", DEST_DENSE),
        ("model.language_model.layers.0.mlp.experts.3.gate_proj.weight", DEST_EXPERT),
        ("model.layers.0.mlp.experts.3.down_proj.weight_scale_inv", DEST_EXPERT),
        ("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", DEST_NGRAM),
        ("model.language_model.layers.2.ple.embed.weight", DEST_NGRAM),
        ("model.language_model.ngram_embed.weight", DEST_NGRAM),
        ("model.language_model.layers.2.ple.conv1d.weight", DEST_DENSE),
        ("model.language_model.embed_tokens.weight", DEST_DENSE),
        ("lm_head.weight", DEST_DENSE),
    ]
    bad = [(n, classify_tensor(n), e) for n, e in cases if classify_tensor(n) != e]
    if bad:
        raise SystemExit(f"qwen_layout selfcheck failed: {bad}")
    print("qwen_layout selfcheck: ok")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/root/models/Qwen3.8-Flash-Next-FP8")
    ap.add_argument("--out", default="/root/models/Qwen38FN-FP8-FTW-v1")
    ap.add_argument("--shard-gib", type=float, default=8.0)
    ns = ap.parse_args(argv)
    _selfcheck()
    shard_limit = int(ns.shard_gib * (1 << 30))
    shard_limit -= shard_limit % 4096
    plan = _plan(ns.model, ns.out, shard_limit)
    _print_plan(plan)
    return 0 if plan["verdict"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
