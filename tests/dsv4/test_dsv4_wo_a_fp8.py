"""FREETOKEN_WO_A_FP8 wiring, and the FTW checkpoint's independence from it.

The flag is read once at import because it decides two things that happen at different
times: the dtype of the ``wo_a`` parameter the model allocates, and what the load path
hands ``load_state_dict``. If those two ever disagree the load fails (or, worse, silently
mis-shapes), so the first half of this file pins that they agree in BOTH settings.

The second half pins the harder version of the same problem. An FTW stores whatever
``iter_weights`` yielded at CONVERSION time, but the model is built from the environment
at SERVE time, so anything the env decides has to be decided at LOAD. ``wo_a`` therefore
travels in its env-independent STORAGE form (the checkpoint's own FP8 + e8m0 scale) and
``adapt_weights`` resolves the runtime form -- which is what lets ONE FTW serve both
settings. Also pinned: an FTW converted before that change (bf16 ``wo_a``, no scale)
still loads with the flag off, and with the flag on is refused with an actionable
message rather than a bare missing-key error.

The flag is import-time, so each case runs in a subprocess.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu


def _child_env(flag: str) -> dict:
    import freetoken

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        os.path.dirname(os.path.dirname(freetoken.__file__))
        + os.pathsep + env.get("PYTHONPATH", "")
    )
    env["FREETOKEN_WO_A_FP8"] = flag
    return env


_SENTINEL = "@@FTJSON@@"


def _child(src: str, flag: str) -> dict:
    """Run ``src`` in a fresh interpreter (the flag is import-time) and parse the JSON it
    writes after ``_SENTINEL`` -- loaders print progress bars, so stdout is not clean."""
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(src)], env=_child_env(flag),
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    assert _SENTINEL in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    return json.loads(r.stdout.split(_SENTINEL, 1)[1])


_STORAGE_AND_RUNTIME = """
    import json, sys, torch
    from freetoken.models.deepseek_v4.args import DeepseekV4Args
    from freetoken.models.deepseek_v4.attention import Attention
    from freetoken.models.deepseek_v4 import layers as L, weight as W

    args = DeepseekV4Args(max_batch_size=1, compress_ratios=(0,) * 8)
    attn = Attention(0, args, compress_ratio=0)
    params = {n: [list(p.shape), str(p.dtype)]
              for n, p in attn.named_parameters() if "wo_a" in n}

    rows = args.o_groups * args.o_lora_rank
    k = args.n_heads * args.head_dim // args.o_groups
    w = torch.arange(rows * k, dtype=torch.float32).remainder(200).sub(100)
    w = w.reshape(rows, k).to(torch.float8_e4m3fn)
    s = torch.arange(rows // 128 * (k // 128), dtype=torch.uint8)
    s = s.reshape(rows // 128, k // 128).remainder(20).add(120)
    s = s.view(torch.float8_e8m0fnu)

    storage = {n: [list(t.shape), str(t.dtype), int(t.view(torch.uint8).sum())]
               for n, t in W._wo_a("attn.wo_a", w, s)}
    runtime = {n: [list(t.shape), str(t.dtype), int(t.view(torch.uint8).sum())]
               for n, t in W.adapt_weights(W._wo_a("attn.wo_a", w, s), "<synthetic>")}
    deq = W._dequant_fp8_block(w, s)
    sys.stdout.write("@@FTJSON@@" + json.dumps(
        {"flag": L.WO_A_FP8, "params": params, "storage": storage, "runtime": runtime,
         "w_sum": int(w.view(torch.uint8).sum()),
         "s_sum": int(s.view(torch.uint8).sum()),
         "deq_sum": int(deq.view(torch.uint8).sum())}))
"""


@pytest.mark.parametrize("flag,expect", [("", False), ("1", True)])
def test_storage_form_is_flag_independent(flag, expect):
    """``iter_weights`` (via ``_wo_a``) must yield the checkpoint's own FP8 + e8m0 pair
    whatever the flag says -- that is what makes one FTW serve both settings, and it is
    a byte-for-byte pass-through, so no re-quantization enters anywhere."""
    got = _child(_STORAGE_AND_RUNTIME, flag)
    assert got["flag"] is expect
    storage = got["storage"]
    assert set(storage) == {"attn.wo_a", "attn.wo_a_scale"}
    assert storage["attn.wo_a"][1] == "torch.float8_e4m3fn"
    assert storage["attn.wo_a_scale"][1] == "torch.float8_e8m0fnu"
    assert storage["attn.wo_a"][2] == got["w_sum"]
    assert storage["attn.wo_a_scale"][2] == got["s_sum"]


@pytest.mark.parametrize("flag,expect", [("", False), ("1", True)])
def test_param_dtypes_match_the_adapted_stream(flag, expect):
    got = _child(_STORAGE_AND_RUNTIME, flag)
    assert got["flag"] is expect
    params, runtime = got["params"], got["runtime"]
    assert {n for n in params} == {n.split("attn.")[-1] for n in runtime}
    for n, (shape, dtype) in params.items():
        r_shape, r_dtype, _sum = runtime[f"attn.{n}"]
        assert [shape, dtype] == [r_shape, r_dtype], (n, shape, dtype, r_shape, r_dtype)
    if expect:
        # FP8 runtime form: the adapter is a pass-through, byte for byte.
        assert runtime["attn.wo_a"][2] == got["w_sum"]
        assert runtime["attn.wo_a_scale"][2] == got["s_sum"]
        assert params["wo_a"][1] == "torch.float8_e4m3fn"
        assert params["wo_a_scale"][1] == "torch.float8_e8m0fnu"
    else:
        # bf16 runtime form: the adapter folded the pair, and produced exactly what the
        # old load-time dequant produced.
        assert "wo_a_scale" not in params and "attn.wo_a_scale" not in runtime
        assert params["wo_a"][1] == "torch.bfloat16"
        assert runtime["attn.wo_a"][2] == got["deq_sum"]


def test_flipping_the_flag_after_import_is_refused(monkeypatch):
    from freetoken.models.deepseek_v4 import layers as L

    monkeypatch.setenv("FREETOKEN_WO_A_FP8", "0" if L.WO_A_FP8 else "1")
    with pytest.raises(RuntimeError, match="changed after import"):
        L.wo_a_fp8()


# ======================================================================================
# FTW round-trip: one checkpoint, both settings.
# ======================================================================================
ROWS, K, BLOCK, LAYERS = 256, 128, 128, 2


def _mini_ftw(tmp_path, *, legacy: bool):
    """A minimal DSV4 checkpoint dir holding an FTW dense region, written exactly the way
    ``convert_checkpoint`` writes it (``FTWWriter.add_tensor(..., kind="weight")``).

    ``legacy=True`` reproduces an FTW converted BEFORE wo_a moved to its storage form:
    ``wo_a`` already expanded to bf16, with no scale beside it.
    """
    from freetoken.checkpoint.ftw import FTWWriter
    from freetoken.models.deepseek_v4.weight import _dequant_fp8_block

    d = tmp_path / ("legacy_ftw" if legacy else "ftw")
    (d / "inference").mkdir(parents=True)
    # Enough config for cached_load_hf_config + parse_config to resolve the model spec,
    # which is all the load path needs in order to find the adapt_weights hook.
    (d / "config.json").write_text(json.dumps({
        "architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4",
        "max_position_embeddings": 4096,
    }))
    (d / "inference" / "config.json").write_text(json.dumps({
        "dim": 4096, "n_layers": LAYERS, "n_heads": 64, "head_dim": 512,
        "o_groups": 8, "o_lora_rank": 1024, "vocab_size": 129280,
        "compress_ratios": [0] * LAYERS,
    }))

    g = torch.Generator().manual_seed(11)
    truth = {}
    w = FTWWriter(str(d), shard_limit=1 << 20)
    for layer in range(LAYERS):
        a = f"layers.{layer}.attn"
        wt = (torch.randn(ROWS, K, generator=g) * 0.05).to(FP8)
        sc = torch.randint(112, 142, (ROWS // BLOCK, K // BLOCK), generator=g,
                           dtype=torch.uint8).view(E8M0)
        truth[f"{a}.wo_a"] = (wt, sc)
        if legacy:
            w.add_tensor(f"{a}.wo_a", _dequant_fp8_block(wt, sc), kind="weight")
        else:
            w.add_tensor(f"{a}.wo_a", wt, kind="weight")
            w.add_tensor(f"{a}.wo_a_scale", sc, kind="weight")
        # An ordinary FP8 projection, for contrast: wo_b already round-trips this way,
        # and the storage form for wo_a is the same treatment.
        w.add_tensor(f"{a}.wo_b.weight", (torch.randn(K, K, generator=g) * 0.05).to(FP8),
                     kind="weight")
        w.add_tensor(f"{a}.wo_b.scale",
                     torch.randint(112, 142, (1, 1), generator=g,
                                   dtype=torch.uint8).view(E8M0), kind="weight")
        w.add_tensor(f"{a}.kv_norm.weight", torch.ones(K), kind="weight")
    w.finalize({"counts": {"weight": 0, "experts_bank": 0}})
    return d, truth


_LOAD_FTW = """
    import json, sys, torch
    from freetoken.models.weight import load_weight
    from freetoken.models.deepseek_v4 import layers as L

    out, err = {}, None
    try:
        for name, t in load_weight(PATH, torch.device("cpu"), include_moe_experts=False,
                                   adapt=ADAPT):
            out[name] = [list(t.shape), str(t.dtype), int(t.view(torch.uint8).sum())]
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    sys.stdout.write("@@FTJSON@@" + json.dumps(
        {"flag": L.WO_A_FP8, "loaded": out, "error": err}))
"""


def _load(path, flag: str, *, adapt: bool = True) -> dict:
    src = _LOAD_FTW.replace("PATH", repr(str(path))).replace("ADAPT", repr(adapt))
    return _child(src, flag)


@pytest.mark.parametrize("flag,fp8", [("", False), ("1", True)])
def test_one_ftw_serves_both_settings(tmp_path, flag, fp8):
    """The whole point: the SAME FTW loads correctly with the flag on and off."""
    d, truth = _mini_ftw(tmp_path, legacy=False)
    got = _load(d, flag)
    assert got["error"] is None, got["error"]
    assert got["flag"] is fp8
    loaded = got["loaded"]
    # The unrelated tensors are untouched either way.
    assert loaded["layers.0.attn.wo_b.weight"][1] == "torch.float8_e4m3fn"
    assert loaded["layers.0.attn.wo_b.scale"][1] == "torch.float8_e8m0fnu"
    assert loaded["layers.0.attn.kv_norm.weight"][1] == "torch.float32"
    for layer in range(LAYERS):
        a = f"layers.{layer}.attn"
        wt, sc = truth[f"{a}.wo_a"]
        if fp8:
            assert loaded[f"{a}.wo_a"][1] == "torch.float8_e4m3fn"
            assert loaded[f"{a}.wo_a"][2] == int(wt.view(torch.uint8).sum())
            assert loaded[f"{a}.wo_a_scale"][1] == "torch.float8_e8m0fnu"
            assert loaded[f"{a}.wo_a_scale"][2] == int(sc.view(torch.uint8).sum())
        else:
            from freetoken.models.deepseek_v4.weight import _dequant_fp8_block

            assert f"{a}.wo_a_scale" not in loaded
            assert loaded[f"{a}.wo_a"][1] == "torch.bfloat16"
            assert loaded[f"{a}.wo_a"][0] == [ROWS, K]
            # Bit-identical to the dequant the bf16 path used to get at conversion time.
            ref = _dequant_fp8_block(wt, sc)
            assert loaded[f"{a}.wo_a"][2] == int(ref.view(torch.uint8).sum())


@pytest.mark.parametrize("flag", ["", "1"])
def test_conversion_reads_the_storage_form_whatever_the_env_says(tmp_path, flag):
    """``convert_checkpoint`` calls ``load_weight(..., adapt=False)``; that must hand back
    the FP8 pair in both settings, or the flag gets baked into the FTW again."""
    d, truth = _mini_ftw(tmp_path, legacy=False)
    got = _load(d, flag, adapt=False)
    assert got["error"] is None, got["error"]
    for layer in range(LAYERS):
        a = f"layers.{layer}.attn"
        wt, sc = truth[f"{a}.wo_a"]
        assert got["loaded"][f"{a}.wo_a"][1] == "torch.float8_e4m3fn"
        assert got["loaded"][f"{a}.wo_a"][2] == int(wt.view(torch.uint8).sum())
        assert got["loaded"][f"{a}.wo_a_scale"][2] == int(sc.view(torch.uint8).sum())


def test_legacy_ftw_still_loads_with_the_flag_off(tmp_path):
    d, truth = _mini_ftw(tmp_path, legacy=True)
    got = _load(d, "")
    assert got["error"] is None, got["error"]
    from freetoken.models.deepseek_v4.weight import _dequant_fp8_block

    for layer in range(LAYERS):
        a = f"layers.{layer}.attn"
        wt, sc = truth[f"{a}.wo_a"]
        assert f"{a}.wo_a_scale" not in got["loaded"]
        assert got["loaded"][f"{a}.wo_a"][1] == "torch.bfloat16"
        assert got["loaded"][f"{a}.wo_a"][2] == int(
            _dequant_fp8_block(wt, sc).view(torch.uint8).sum())


def test_legacy_ftw_with_the_flag_on_says_how_to_fix_it(tmp_path):
    d, _truth = _mini_ftw(tmp_path, legacy=True)
    got = _load(d, "1")
    assert got["error"] is not None, "a legacy FTW must not load silently under the flag"
    msg = got["error"]
    assert msg.startswith("RuntimeError")
    # Actionable, not a bare missing-key error out of load_state_dict.
    assert "ft checkpoint" in msg, msg
    assert "FREETOKEN_WO_A_FP8" in msg, msg
    assert "wo_a" in msg and str(d) in msg, msg
    assert "Missing weight for" not in msg, msg
