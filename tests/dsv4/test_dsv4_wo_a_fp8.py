"""FREETOKEN_WO_A_FP8 wiring.

The flag is read once at import because it decides two things that happen at different
times: the dtype of the ``wo_a`` parameter the model allocates, and which tensors
``iter_weights`` yields for it. If those two ever disagree the load fails (or, worse,
silently mis-shapes), so this pins that they agree in BOTH settings -- and that the FP8
path passes the checkpoint's weight and e8m0 scale through untouched, which is what
makes it the same quantization scheme ``wo_b`` already uses.

The flag is import-time, so the ON case runs in a subprocess.
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch


def _run(flag: str) -> dict:
    import freetoken

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        os.path.dirname(os.path.dirname(freetoken.__file__))
        + os.pathsep + env.get("PYTHONPATH", "")
    )
    env["FREETOKEN_WO_A_FP8"] = flag
    src = textwrap.dedent(
        """
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
        yielded = {n: [list(t.shape), str(t.dtype), int(t.view(torch.uint8).sum())]
                   for n, t in W._wo_a("attn.wo_a", w, s)}
        json.dump({"flag": L.WO_A_FP8, "params": params, "yielded": yielded,
                   "w_sum": int(w.view(torch.uint8).sum()),
                   "s_sum": int(s.view(torch.uint8).sum())}, sys.stdout)
        """
    )
    r = subprocess.run([sys.executable, "-c", src], env=env, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    import json

    return json.loads(r.stdout)


@pytest.mark.parametrize("flag,expect", [("", False), ("1", True)])
def test_param_dtypes_match_what_the_loader_yields(flag, expect):
    got = _run(flag)
    assert got["flag"] is expect
    params, yielded = got["params"], got["yielded"]
    assert {n for n in params} == {n.split("attn.")[-1] for n in yielded}
    for n, (shape, dtype) in params.items():
        y_shape, y_dtype, _sum = yielded[f"attn.{n}"]
        assert [shape, dtype] == [y_shape, y_dtype], (n, shape, dtype, y_shape, y_dtype)
    if expect:
        # FP8: the checkpoint tensors pass through byte-for-byte -- no re-quantization,
        # so the numerics differ from bf16 only in WHERE the dequant happens.
        assert yielded["attn.wo_a"][2] == got["w_sum"]
        assert yielded["attn.wo_a_scale"][2] == got["s_sum"]
        assert params["wo_a"][1] == "torch.float8_e4m3fn"
        assert params["wo_a_scale"][1] == "torch.float8_e8m0fnu"
    else:
        assert "wo_a_scale" not in params
        assert params["wo_a"][1] == "torch.bfloat16"


def test_flipping_the_flag_after_import_is_refused(monkeypatch):
    from freetoken.models.deepseek_v4 import layers as L

    monkeypatch.setenv("FREETOKEN_WO_A_FP8", "0" if L.WO_A_FP8 else "1")
    with pytest.raises(RuntimeError, match="changed after import"):
        L.wo_a_fp8()
