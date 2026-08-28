from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated, OPList

if TYPE_CHECKING:
    from freetoken.models.qwen4_exp.config import VisionConfig

# Qwen4-Exp's SigLIP/NaViT-style vision tower, verified tensor-for-tensor (name and shape)
# against the shipped `model.visual.*` weights and against HF transformers' `qwen3_vl`
# vision modules (`Qwen3VLVisionModel` et al. in
# transformers/models/qwen3_vl/modeling_qwen3_vl.py) -- transformers has no `qwen4_exp`
# model at all (text or vision; verified: no `models/qwen4_exp*` directory, no
# `auto_map`/remote code shipped with the checkpoint), but this vision tower's weight
# names (`blocks.N.attn.qkv`, `blocks.N.mlp.linear_fc1/fc2`, `merger.linear_fc1/fc2`,
# `pos_embed.weight`, ...) and config fields (`num_position_embeddings`,
# `deepstack_visual_indexes`, ...) are byte-for-byte the Qwen3-VL vision config/state-dict
# schema, so `Qwen3VLVisionModel` loaded with the real checkpoint weights serves as a
# genuine reference implementation (see tests/models/test_qwen4_exp_vision.py).
#
# Differences from the Gemma4 template this module was scaffolded from
# (python/freetoken/models/gemma4/vision.py) -- every one checked against the reference
# rather than assumed:
#   - Position embedding: Qwen4Exp uses BOTH a learned absolute `pos_embed` (bilinearly
#     interpolated from a 48x48 grid to each image's actual patch grid) AND 2-D rotary
#     position embeddings in attention. Gemma4 uses ONLY the learned table (no rotary).
#   - Norm: plain LayerNorm (mean-centered, with bias, eps=1e-6 hardcoded -- Qwen3VLVision
#     config carries no eps field at all), not Gemma4's bias-free RMSNorm.
#   - Block structure: standard 2-norm pre-LN (`x = x + attn(norm1(x)); x = x +
#     mlp(norm2(x))`), not Gemma4's 4-norm post-sublayer sandwich.
#   - Attention: fused QKV/proj Linears carry bias=True (Gemma4: bias=False); plain MHA,
#     no per-head q/k/v RMSNorm at all (Gemma4 has q_norm/k_norm/v_norm); attention scale
#     is the usual `head_dim ** -0.5` (Gemma4 hardcodes `scale=1.0`).
#   - MLP: plain up-project -> gelu_pytorch_tanh -> down-project with bias (Gemma4: gated
#     SwiGLU-style gate/up/down, bias-free).
#   - Patch embed: Conv3d[Cout, Cin, T=2, P, P] over channel-major flattened patches
#     (temporal_patch_size=2 -- doubled/paired frames per patch); with kernel==stride and
#     no padding this is exactly a Linear over the flattened patch, which is how it's
#     implemented here (weight reshaped [Cout, Cin*T*P*P] at load time). Gemma4 has no
#     temporal axis at all.
#   - Merger: the merger's own final Linear projects straight to `out_hidden_size` (==
#     the text hidden size, 2560), so there is no separate multimodal-embedder stage the
#     way Gemma4 needs `Gemma4MultimodalEmbedder`. The merger's activation is *exact*
#     (erf) GELU (`nn.GELU()`), distinct from the blocks' tanh-approximate GELU -- easy to
#     get wrong since both are "GELU".
#   - No `deepstack_visual_indexes` injection: the checkpoint's list is empty for this
#     model, so the encoder output goes straight to the (single) merger -- verified by
#     asserting the parsed config's list is empty; a non-empty list would need the
#     multi-scale injection path Qwen3-VL has and this module intentionally does not
#     implement.
#   - No "standardize" (post-pool mean/scale) step and no `sqrt(hidden)` output scaling --
#     both Gemma4-specific; absent from the checkpoint's tensor set and from Qwen3-VL.


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_qwen_vision_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """``x``: ``[B, P, N, head_dim]``; ``cos``/``sin``: ``[B, P, head_dim]``.

    Mirrors HF's ``apply_rotary_pos_emb_vision``: upcast to fp32, rotate the *whole*
    head_dim with a single ``rotate_half`` (paired at ``i``/``i + head_dim/2``) -- unlike
    Gemma4's ``_apply_multidim_rope``, which splits head_dim into independent x/y halves
    each rotated on their own.
    """
    orig_dtype = x.dtype
    xf = x.float()
    c = cos.unsqueeze(2).float()
    s = sin.unsqueeze(2).float()
    out = xf * c + _rotate_half(xf) * s
    return out.to(orig_dtype)


class _Qwen4VisionRotary:
    """On-the-fly 2-D rope cos/sin tables (no learnable params).

    Mirrors ``Qwen3VLVisionRotaryEmbedding``: per-axis frequencies over ``head_dim // 4``
    each (h then w, ``head_dim // 2`` combined) are duplicated to fill ``head_dim`` before
    ``rotate_half`` is applied. ``theta`` defaults to 10000.0 -- Qwen3VLVisionConfig
    carries no ``rope_theta`` field; the HF module hardcodes the default.
    """

    def __init__(self, vc: VisionConfig):
        self.head_dim = vc.head_dim
        self.theta = vc.rope_theta
        self._dim = vc.head_dim // 2
        self._inv_freq: torch.Tensor | None = None

    def _inv(self, device: torch.device) -> torch.Tensor:
        if self._inv_freq is None or self._inv_freq.device != device:
            self._inv_freq = 1.0 / (
                self.theta
                ** (torch.arange(0, self._dim, 2, dtype=torch.float32, device=device) / self._dim)
            )
        return self._inv_freq

    def cos_sin(self, position_ids: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        inv = self._inv(position_ids.device)  # [head_dim / 4]
        pos = position_ids.float()  # [B, P, 2] (row, col)
        freqs = (pos.unsqueeze(-1) * inv).flatten(-2)  # [B, P, head_dim / 2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, P, head_dim]
        return emb.cos().to(dtype), emb.sin().to(dtype)


class _LayerNorm(BaseOP):
    """Plain LayerNorm (mean/var, elementwise affine). FreeToken's shared ``layers/norm.py``
    only has RMSNorm variants (every other model in this codebase is RMSNorm-normed); Qwen's
    vision tower needs real LayerNorm, so it is defined locally here rather than added to the
    shared module. ``eps=1e-6`` is hardcoded to match HF (no config field carries it)."""

    def __init__(self, size: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = torch.empty(size)
        self.bias = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class _LearnedPosEmbed(BaseOP):
    """Wraps the learned ``[num_position_embeddings, hidden]`` table as a one-tensor BaseOP
    so its state_dict key is ``pos_embed.weight`` (matching the checkpoint) rather than a
    bare tensor attribute."""

    def __init__(self, num_positions: int, hidden_size: int):
        self.weight = torch.empty(num_positions, hidden_size)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.weight)


class Qwen4VisionMLP(BaseOP):
    """Plain up/down MLP with bias (not Gemma4's gated SwiGLU-style gate/up/down)."""

    def __init__(self, vc: VisionConfig):
        self.linear_fc1 = LinearReplicated(vc.hidden_size, vc.intermediate_size, has_bias=True)
        self.linear_fc2 = LinearReplicated(vc.intermediate_size, vc.hidden_size, has_bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act = F.gelu(self.linear_fc1.forward(x), approximate="tanh")
        return self.linear_fc2.forward(act)


class Qwen4VisionAttention(BaseOP):
    """Bidirectional plain multi-head attention (no GQA, no per-head q/k/v norm) with 2-D
    RoPE. HF uses the usual ``head_dim ** -0.5`` scaling (Gemma4 hardcodes ``scale=1.0``).
    """

    def __init__(self, vc: VisionConfig):
        self.head_dim = vc.head_dim
        self.num_heads = vc.num_heads
        dim = self.num_heads * self.head_dim
        self.qkv = LinearReplicated(vc.hidden_size, 3 * dim, has_bias=True)
        self.proj = LinearReplicated(dim, vc.hidden_size, has_bias=True)
        self._scale = self.head_dim**-0.5

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        B, P, _ = x.shape
        qkv = self.qkv.forward(x).view(B, P, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each [B, P, num_heads, head_dim]

        q = _apply_qwen_vision_rope(q, cos, sin).transpose(1, 2)
        k = _apply_qwen_vision_rope(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)

        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self._scale)
        o = o.transpose(1, 2).reshape(B, P, self.num_heads * self.head_dim)
        return self.proj.forward(o)


class Qwen4VisionEncoderLayer(BaseOP):
    """Standard 2-norm pre-LN block (``x = x + attn(norm1(x)); x = x + mlp(norm2(x))``) --
    not Gemma4's 4-norm post-sublayer sandwich."""

    def __init__(self, vc: VisionConfig):
        self.norm1 = _LayerNorm(vc.hidden_size)
        self.norm2 = _LayerNorm(vc.hidden_size)
        self.attn = Qwen4VisionAttention(vc)
        self.mlp = Qwen4VisionMLP(vc)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn.forward(self.norm1.forward(x), cos, sin, attn_mask)
        x = x + self.mlp.forward(self.norm2.forward(x))
        return x


class Qwen4VisionPatchEmbedder(BaseOP):
    """Conv3d(kernel==stride, no padding) degenerates to a Linear over the flattened
    ``in_channels * temporal_patch_size * patch_size**2`` patch vector; the checkpoint's
    5-D Conv3d weight is reshaped to 2-D at load time (see ``load_visual_state_dict``)."""

    def __init__(self, vc: VisionConfig):
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
        self.proj = LinearReplicated(patch_dim, vc.hidden_size, has_bias=True)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.proj.forward(pixel_values.to(self.proj.weight.dtype))


class Qwen4VisionMerger(BaseOP):
    """2x2 spatial merge -> 2-layer MLP projecting straight to the text hidden size.

    Unlike Gemma4 (whose vision tower pools to its OWN hidden size and needs a separate
    ``Gemma4MultimodalEmbedder`` afterward), this merger's ``linear_fc2`` already outputs
    ``out_hidden_size`` (== the LM hidden size, 2560) -- ``encode_images`` calls this
    directly with nothing downstream. Note: the merger's activation is *exact* GELU
    (``nn.GELU()``), not the blocks' tanh-approximate ``gelu_pytorch_tanh``.
    """

    def __init__(self, vc: VisionConfig):
        merge_dim = vc.hidden_size * vc.spatial_merge_size**2
        self.norm = _LayerNorm(vc.hidden_size)
        self.linear_fc1 = LinearReplicated(merge_dim, merge_dim, has_bias=True)
        self.linear_fc2 = LinearReplicated(merge_dim, vc.out_hidden_size, has_bias=True)
        self._merge_dim = merge_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm.forward(x)
        x = x.reshape(-1, self._merge_dim)
        return self.linear_fc2.forward(F.gelu(self.linear_fc1.forward(x)))


class Qwen4VisionModel(BaseOP):
    """Pixels -> soft tokens, already projected to the text hidden size.

    ``pixel_values``: ``[num_images, num_patches, in_channels*temporal_patch_size*patch_size**2]``
    -- flattened Conv3d-equivalent patches, channel-major (matching ``patch_embed.proj``'s
    weight layout). ``position_ids``: ``[num_images, num_patches, 2]`` raw ``(row, col)``
    patch-grid coordinates (0-indexed, pre-merge resolution; ``(-1, -1)`` marks padding) --
    same ``(-1, -1)``-padded batched-image convention as Gemma4's ``encode_images``, so both
    the per-image attention isolation (batch dim = image) and the padding mask reuse
    Gemma4's trick. Unlike Gemma4, patches need not arrive in any particular order: the
    2x2 merge groups are gathered explicitly from ``position_ids`` (see ``_merge_group``),
    not assumed contiguous -- verified by the permutation self-consistency test.
    """

    def __init__(self, vc: VisionConfig):
        if vc.deepstack_visual_indexes:
            raise NotImplementedError(
                "deepstack visual injection is not implemented; this checkpoint's "
                f"deepstack_visual_indexes={list(vc.deepstack_visual_indexes)} is non-empty"
            )
        self.patch_embed = Qwen4VisionPatchEmbedder(vc)
        self.pos_embed = _LearnedPosEmbed(vc.num_position_embeddings, vc.hidden_size)
        self.blocks = OPList([Qwen4VisionEncoderLayer(vc) for _ in range(vc.num_layers)])
        self.merger = Qwen4VisionMerger(vc)
        self._rotary = _Qwen4VisionRotary(vc)
        self._merge = vc.spatial_merge_size
        self._grid_side = int(round(vc.num_position_embeddings**0.5))
        self._hidden = vc.hidden_size

    def _grid_extents(self, position_ids: torch.Tensor, padding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-image ``(height, width)`` patch-grid extents, derived from the max valid
        position id + 1 -- mirrors Gemma4's ``_avg_pool_by_positions`` deriving grid size
        from ``position_ids`` directly rather than a separate ``grid_thw`` side channel
        (``encode_images``'s 2-tensor contract has no room for one)."""
        clamped = position_ids.clamp(min=0)
        valid = ~padding
        rows = torch.where(valid, clamped[..., 0], torch.zeros_like(clamped[..., 0]))
        cols = torch.where(valid, clamped[..., 1], torch.zeros_like(clamped[..., 1]))
        heights = rows.max(dim=1).values + 1
        widths = cols.max(dim=1).values + 1
        return heights, widths

    def _interp_pos_embed(
        self, position_ids: torch.Tensor, padding: torch.Tensor, heights: torch.Tensor, widths: torch.Tensor
    ) -> torch.Tensor:
        """Bilinearly resamples the learned ``side x side`` position grid to each image's
        actual ``(h, w)`` patch grid (``align_corners=True``, matching HF's
        ``fast_pos_embed_interpolate`` / ``get_vision_interpolation_indices_and_weights``,
        whose docstring states it reproduces ``F.interpolate(mode="bilinear",
        align_corners=True)`` exactly -- used directly here instead of re-deriving the
        manual tap/weight gather)."""
        B, P, _ = position_ids.shape
        clamped = position_ids.clamp(min=0)
        side = self._grid_side
        table = self.pos_embed.weight.view(1, side, side, self._hidden).permute(0, 3, 1, 2).float()
        out = torch.zeros(B, P, self._hidden, device=position_ids.device, dtype=torch.float32)
        for b in range(B):
            h, w = int(heights[b].item()), int(widths[b].item())
            grid = F.interpolate(table, size=(h, w), mode="bilinear", align_corners=True)
            grid = grid.view(self._hidden, h, w)
            rows = clamped[b, :, 0]
            cols = clamped[b, :, 1]
            out[b] = grid[:, rows, cols].transpose(0, 1)
        return out

    def _merge_group(
        self, hidden: torch.Tensor, position_ids: torch.Tensor, padding: torch.Tensor,
        heights: torch.Tensor, widths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gathers each image's patches into 2x2-block order (block-row-major over blocks,
        then ``in_row*merge + in_col`` within a block -- the exact order HF's own
        ``get_vision_position_ids`` block permutation implies, which the merger's weights
        were trained against) and concatenates each block of 4 into one ``merge_dim``-wide
        row for the merger. Uses one-hot-gather-by-position (mirrors Gemma4's
        ``_avg_pool_by_positions``) rather than assuming contiguous input order, so caller
        patch order does not matter. The gather is an exact permutation (each output slot
        receives exactly one input row, the rest are added zeros), so it introduces no
        rounding beyond what a plain reshape would."""
        B, P, H = hidden.shape
        merge = self._merge
        clamped = position_ids.clamp(min=0)
        valid = ~padding
        blocks_w = widths // merge  # [B]
        row, col = clamped[..., 0], clamped[..., 1]
        br, bc = row // merge, col // merge
        ir, ic = row % merge, col % merge
        block_index = br * blocks_w.unsqueeze(1) + bc  # [B, P]
        sub_index = ir * merge + ic  # [B, P]
        max_out_blocks = int(((heights // merge) * (widths // merge)).max().item())
        n_slots = max_out_blocks * merge * merge
        target = block_index * (merge * merge) + sub_index  # [B, P], in [0, n_slots)
        target = torch.where(valid, target, torch.full_like(target, n_slots))
        onehot = F.one_hot(target.clamp(max=n_slots), n_slots + 1)[..., :-1]  # [B, P, n_slots]
        onehot = onehot.to(hidden.dtype)
        grouped = torch.bmm(onehot.transpose(1, 2), hidden)  # [B, n_slots, H] -- NOT yet
        # reshaped to merge_dim: the merger's own norm runs on the pre-merge H-wide rows
        # (matching HF's PatchMerger, which reshapes to merge_dim internally, *after*
        # norm), so this returns one row per patch, just reordered into block order.
        out_mask = torch.logical_not((onehot == 0).all(dim=1))  # [B, n_slots] -> collapse below
        out_mask = out_mask.view(B, max_out_blocks, merge * merge).all(dim=-1)  # [B, max_out_blocks]
        return grouped, out_mask

    def forward(self, pixel_values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        padding = (position_ids == -1).all(dim=-1)  # [B, P] True = padding patch
        heights, widths = self._grid_extents(position_ids, padding)

        h = self.patch_embed.forward(pixel_values)
        pos = self._interp_pos_embed(position_ids, padding, heights, widths)
        h = h + pos.to(h.dtype)

        cos, sin = self._rotary.cos_sin(position_ids.clamp(min=0), h.dtype)
        attn_mask = (~padding)[:, None, None, :]  # [B, 1, 1, P] True = attend
        for layer in self.blocks.op_list:
            h = layer.forward(h, cos, sin, attn_mask)

        grouped, out_mask = self._merge_group(h, position_ids, padding, heights, widths)
        B, n_slots, H = grouped.shape
        merged = self.merger.forward(grouped.reshape(B * n_slots, H))  # merger reshapes to merge_dim itself
        return merged[out_mask.reshape(-1)]


_VISUAL_HF_PREFIX = "model.visual."


def adapt_vision_tensor(short_name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Applies the one checkpoint -> module shape change the vision tower needs:
    ``patch_embed.proj.weight``'s Conv3d ``[C_out, C_in, T, P, P]`` -> this module's
    Linear-equivalent ``[C_out, C_in*T*P*P]`` (see ``Qwen4VisionPatchEmbedder``). Every
    other vision tensor passes through unchanged. ``short_name`` is the checkpoint key
    with the ``model.visual.`` prefix already stripped (i.e. a key of
    ``Qwen4VisionModel.state_dict()``).

    Single source of truth for this reshape, shared by ``load_visual_state_dict`` below
    and ``models/qwen4_exp/weight.py``'s ``iter_weights`` (the raw-HF-checkpoint dense
    stream, gated on ``vision_load_enabled()``) so the two loading paths cannot drift.
    """
    if short_name == "patch_embed.proj.weight":
        return tensor.reshape(tensor.shape[0], -1)
    return tensor


def load_visual_state_dict(model_dir: str, dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """Loads the ``model.visual.*`` tensors directly from an HF safetensors checkpoint's
    ``model.safetensors.index.json``. Used both by ``encode_images``-adjacent tooling that
    wants to load straight from an HF release, and by ``checkpoint/convert.py``'s dedicated
    vision pass-through (see ``_iter_vision_from_disk``), which carries these tensors into
    the FTW under the renamed ``vision_tower.*`` key (``checkpoint/qwen_layout.py``'s
    ``ftw_vision_name`` implements the same rename for the stdlib-only classification path).
    Returns a state dict keyed to match ``Qwen4VisionModel.state_dict()`` (the
    ``model.visual.`` prefix stripped; see ``adapt_vision_tensor`` for the one shape change
    applied along the way).
    """
    from safetensors import safe_open

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    by_file: dict[str, list[str]] = {}
    for key, fname in weight_map.items():
        if key.startswith(_VISUAL_HF_PREFIX):
            by_file.setdefault(fname, []).append(key)

    out: dict[str, torch.Tensor] = {}
    for fname, keys in by_file.items():
        with safe_open(os.path.join(model_dir, fname), framework="pt") as f:
            for key in keys:
                tensor = f.get_tensor(key).to(dtype)
                short = key[len(_VISUAL_HF_PREFIX):]
                out[short] = adapt_vision_tensor(short, tensor)
    return out


__all__ = ["Qwen4VisionModel", "adapt_vision_tensor", "load_visual_state_dict"]
