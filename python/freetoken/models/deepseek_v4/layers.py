"""Shared DSV4 primitives: weight-only linears, RMSNorm, and the static window /
compressed top-k index builders."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.kernel.triton.dsv4.fp8_linear import block_fp8_linear
from freetoken.kernel.triton.dsv4.norm import rms_norm

FP8 = torch.float8_e4m3fn


def _env_wo_a_fp8() -> bool:
    return os.environ.get("FREETOKEN_WO_A_FP8", "").lower() in ("1", "true", "yes", "on")


#: ``FREETOKEN_WO_A_FP8=1`` keeps the attention output projection's ``wo_a`` in the
#: checkpoint's block-scaled FP8 instead of dequantizing it to bf16 at load. ``wo_a`` is
#: the single largest bf16 read left in the decode step (67 MB/layer x 43 = 2.9 GB/token,
#: already at 94% of achievable bandwidth), so halving its bytes is the only way to make
#: it cheaper. It CHANGES NUMERICS -- the reference runs a bf16 einsum here, and the FP8
#: path additionally quantizes the activation -- so it is OFF by default and must be
#: quality-gated before it is turned on.
#:
#: Read ONCE at import: the flag decides both the parameter dtypes the model allocates and
#: which tensors the weight loader yields, and those two run at different times. A flip in
#: between would silently mis-shape the load, so :func:`wo_a_fp8` refuses it.
WO_A_FP8 = _env_wo_a_fp8()


def wo_a_fp8() -> bool:
    """Is the FP8 ``wo_a`` path enabled for this process? (see :data:`WO_A_FP8`)"""
    if _env_wo_a_fp8() != WO_A_FP8:
        raise RuntimeError(
            "FREETOKEN_WO_A_FP8 changed after import: it selects both the wo_a parameter "
            "dtype and the tensors the loader yields -- set it before the process starts"
        )
    return WO_A_FP8


class Linear(nn.Module):
    """BF16 / FP32 / block-scaled-FP8 linear (weight-only).

    ``kind="fp8"``: weight float8_e4m3fn + 128x128 e8m0 ``scale`` (dequant in the
    Triton GEMM). ``kind="fp32"``/``"bf16"``: plain ``F.linear``.
    """

    def __init__(self, in_features, out_features, bias=False, kind="fp8"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kind = kind
        if kind == "fp8":
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=FP8), requires_grad=False)
            self.scale = nn.Parameter(
                torch.empty(out_features // 128, in_features // 128, dtype=torch.float8_e8m0fnu),
                requires_grad=False,
            )
        else:
            dtype = torch.float32 if kind == "fp32" else torch.bfloat16
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype), requires_grad=False)
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "fp8":
            return block_fp8_linear(x, self.weight, self.scale, self.bias)
        return F.linear(x, self.weight.to(x.dtype), self.bias)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor):
        return rms_norm(x, self.weight, self.eps)


def get_window_topk_idxs(window_size, bsz, seqlen, start_pos):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat([torch.arange(start_pos + 1, window_size), torch.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


def get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)
