from __future__ import annotations

import json
import os

import safetensors
import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP

from .hyperconnect import grouped_rms_norm

# Reference: transformers modeling_qwen4_exp (Qwen4ExpTextNGramEmbedding /
# Qwen4ExpTextPLELayer), read 2026-08-26. Faithful naive port:
#   * hash n-gram ids = XOR of (shifted token id * per-position multiplier), one prime
#     vocab per head, offsets into ONE big embedding table;
#   * the multipliers / head vocab sizes / head offsets are checkpoint BUFFERS -- we load
#     them from the checkpoint instead of re-deriving from the seed;
#   * PLE: key per stream + shared value from the n-gram embedding, signed-sqrt dot gate
#     against the normed streams, plus a dilated depthwise causal conv (dilation =
#     ngram_size) over the normed gated value. Output adds to the 4-stream hidden.
#
# The table (~51 GB, fp8 + one global scale, 128 shards) is deliberately NOT streamed
# through iter_weights: shards are memory-mapped from the checkpoint dir (host-resident,
# page-cache managed) and rows are gathered per token -- 16 rows x 320 B per token, so
# decode traffic is trivial. Performance work (pinning, GPU-side cache) is a later phase.

_PLE_PREFIX = "ple.ple_embedding."


class _NGramTable:
    """Lazy, mmap-backed reader of the sharded fp8 n-gram embedding table."""

    def __init__(self, model_path: str, layer_idx: int, split_parts: int):
        self.model_path = model_path
        self.key_base = (
            f"model.language_model.layers.{layer_idx}.ple.ple_embedding.ngram_embedding"
        )
        self.split_parts = split_parts
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        with open(index_path) as fh:
            self._weight_map = json.load(fh)["weight_map"]
        self._handles: dict[str, object] = {}
        self._shards: list[torch.Tensor | None] = [None] * split_parts
        self.weight_scale = self._tensor(f"{self.key_base}.weight_scale").float().item()
        shard0 = self._shard(0)
        self.rows_per_shard = shard0.shape[0]
        self.head_dim = shard0.shape[1]

    def _tensor(self, key: str) -> torch.Tensor:
        shard_file = self._weight_map[key]
        h = self._handles.get(shard_file)
        if h is None:
            h = safetensors.safe_open(
                os.path.join(self.model_path, shard_file), framework="pt", device="cpu"
            )
            self._handles[shard_file] = h
        return h.get_tensor(key)

    def _shard(self, i: int) -> torch.Tensor:
        t = self._shards[i]
        if t is None:
            t = self._tensor(f"{self.key_base}.shard_{i}.weight")
            self._shards[i] = t
        return t

    def buffer(self, name: str) -> torch.Tensor:
        """Small int64 sidecar buffers: layer_multipliers / ngram_heads_offsets /
        ngram_heads_vocab_sizes (loaded verbatim, never dtype-cast)."""
        key = self.key_base.rsplit(".", 1)[0] + "." + name  # ...ple.ple_embedding.<name>
        return self._tensor(key).clone()

    def gather(self, ids: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        """ids [*] int64 global rows -> [*, head_dim] dequantized embeddings (CPU gather,
        returned on ids' original device)."""
        device = ids.device
        flat = ids.reshape(-1).cpu()
        shard_idx = torch.div(flat, self.rows_per_shard, rounding_mode="floor")
        local = flat - shard_idx * self.rows_per_shard
        out = torch.empty(flat.shape[0], self.head_dim, dtype=out_dtype)
        for s in torch.unique(shard_idx).tolist():
            mask = shard_idx == s
            rows = self._shard(int(s)).index_select(0, local[mask])
            out[mask] = rows.to(out_dtype) * self.weight_scale
        return out.to(device).reshape(*ids.shape, self.head_dim)


def _shift_right_ignore_eos(token_ids: torch.Tensor, shift: int, eos: int) -> torch.Tensor:
    """Reference-faithful: shift right by ``shift`` WITHOUT crossing eos boundaries
    (positions whose segment started after the source fall back to eos)."""
    if shift == 0:
        return token_ids
    batch, seq_len = token_ids.shape
    positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
    eos_positions = torch.where(token_ids == eos, positions, torch.full_like(positions, -1))
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat(
        [eos_positions.new_full((batch, 1), -1), previous_eos_inclusive[:, :-1]], dim=1
    )
    segment_start = previous_eos + 1
    position_in_segment = positions.unsqueeze(0) - segment_start
    source_positions = positions - shift
    gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch, -1)
    shifted = token_ids.gather(dim=1, index=gather_positions)
    valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
    return torch.where(valid, shifted, token_ids.new_full((), eos))


class Qwen4PLELayer(BaseOP):
    """Per-Layer (n-gram) Embedding block, attached to one decoder layer (layer_idx 1
    for the released checkpoint). Adds hashed lexical features to the 4-stream hidden.

    Weight keys (under ``model.layers.<L>.ple.``): key_proj / value_proj (bf16 linears),
    norm_key / norm_query / norm_conv (grouped Gemma-style norms, +1 baked by the
    loader), conv1d (depthwise, dilation=ngram_size). The ngram_embedding subtree is
    read directly from the checkpoint by ``_NGramTable``.

    Recurrent state per request slot (mirrors the GDN conv-state pattern, own buffers):
      * token history: last (ngram_size-1) input ids
      * conv tail: last (kernel-1)*dilation normed-gated-value steps
    """

    def __init__(self, config, layer_idx: int):
        args = config.qwen4_args
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.hc_count = args.hc_count
        self.eps = config.rms_norm_eps
        self.eos = args.eos_token_id
        self.ngram_size = args.ngram_size
        self.context_len = args.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = self.context_len * args.heads_per_ngram
        self.ple_embed_dim = args.ple_embed_dim
        self.conv_kernel = args.ple_conv_kernel_size
        self.conv_dilation = args.ngram_size
        self.state_len = (self.conv_kernel - 1) * self.conv_dilation
        hc_hidden = self.hc_count * self.hidden_size

        self.key_proj = _W(hc_hidden, args.ple_embed_dim)
        self.value_proj = _W(self.hidden_size, args.ple_embed_dim)
        self.norm_key = _N(hc_hidden)
        self.norm_query = _N(hc_hidden)
        self.norm_conv = _N(hc_hidden)
        self.conv1d = _Conv(hc_hidden, self.conv_kernel)

        assert args.model_path, "qwen4_args.model_path required for the n-gram table"
        self._table = _NGramTable(args.model_path, layer_idx, args.split_ngram_parts)
        # Checkpoint buffers (int64) -- authoritative over re-derivation from the seed.
        self.layer_multipliers = self._table.buffer("layer_multipliers")
        self.head_vocab_sizes = self._table.buffer("ngram_heads_vocab_sizes")
        self.head_offsets = self._table.buffer("ngram_heads_offsets")
        # Per-slot recurrent state, allocated lazily (slot count comes from the pool).
        self._tok_hist: torch.Tensor | None = None
        self._conv_tail: torch.Tensor | None = None

    # ---------- state pool ----------

    def _ensure_state(self, slots: int, device, dtype):
        if self._tok_hist is None:
            self._tok_hist = torch.full(
                (slots, self.context_len), self.eos, dtype=torch.long, device=device
            )
            self._conv_tail = torch.zeros(
                slots, self.hc_count * self.hidden_size, self.state_len,
                dtype=dtype, device=device,
            )
            if self.layer_multipliers.device != device:
                self.layer_multipliers = self.layer_multipliers.to(device)
                self.head_vocab_sizes = self.head_vocab_sizes.to(device)
                self.head_offsets = self.head_offsets.to(device)

    def _reset_slots(self, idx: torch.Tensor):
        self._tok_hist.index_fill_(0, idx, self.eos)
        self._conv_tail.index_fill_(0, idx, 0.0)

    # ---------- n-gram hashing ----------

    def _ngram_embed(self, history: torch.Tensor, out_len: int, dtype) -> torch.Tensor:
        """history [B, context_len + T] -> [B, out_len(=T), ple_embed_dim]."""
        shifted = [
            _shift_right_ignore_eos(history, s, self.eos) for s in range(self.ngram_size)
        ]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for pos in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[pos] * self.layer_multipliers[pos])
            sizes = self.head_vocab_sizes[start:end]
            offs = self.head_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes.view(1, 1, -1))
            blocks.append(ids + offs.view(1, 1, -1))
        ids = torch.cat(blocks, dim=-1)[:, -out_len:]  # [B, T, ngram_heads]
        emb = self._table.gather(ids, dtype)  # [B, T, heads, head_dim]
        return emb.flatten(-2)

    # ---------- PLE core (matches the reference forward) ----------

    def _ple_core(self, x4: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """x4 [N, hc*H] current streams, emb [N, ple_embed_dim] -> gated value [N, hc*H]
        (pre-conv part) -- caller runs the causal conv on the normed copy."""
        key = F.linear(emb, self.key_proj.weight)
        key_normed = grouped_rms_norm(key, self.norm_key.weight, self.hidden_size, self.eps)
        key_normed = key_normed.unflatten(-1, (self.hc_count, self.hidden_size))
        value = F.linear(emb, self.value_proj.weight)
        query_normed = grouped_rms_norm(
            x4, self.norm_query.weight, self.hidden_size, self.eps
        ).unflatten(-1, (self.hc_count, self.hidden_size))
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / (self.hidden_size ** 0.5)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = torch.sigmoid(gate) * value.unsqueeze(-2)  # [N, hc, H]
        return gated.flatten(-2)

    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight  # [hc*H, 1, K]

    def forward(self, x4: torch.Tensor) -> torch.Tensor:
        """Returns the PLE delta to ADD to the streams: [N, hc*H]."""
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, x4.device)
            batch.fla_metadata = fla
        li = pool.local_index(self.layer_idx)
        slots = pool.conv_states[li].shape[0]
        self._ensure_state(slots, x4.device, x4.dtype)

        input_ids = batch.input_ids.to(torch.long)

        if batch.is_decode:
            idx = fla.cache_indices  # [B] slot per row
            hist = torch.cat([self._tok_hist[idx], input_ids.view(-1, 1)], dim=-1)  # [B, 3]
            emb = self._ngram_embed(hist, 1, x4.dtype)[:, 0]  # [B, ple_embed_dim]
            gated = self._ple_core(x4, emb)  # [B, hc*H]
            normed = grouped_rms_norm(gated, self.norm_conv.weight, self.hidden_size, self.eps)
            # dilated causal conv, single step: taps at t-9, t-6, t-3, t (K=4, d=3)
            tail = self._conv_tail[idx]  # [B, C, 9]
            w = self._conv_weight().squeeze(1)  # [C, K]
            conv = w[:, -1] * normed
            for k in range(1, self.conv_kernel):
                conv = conv + w[:, -1 - k] * tail[:, :, -k * self.conv_dilation].to(normed.dtype)
            conv = F.silu(conv)
            # state update: append the current normed step
            new_tail = torch.cat([tail[:, :, 1:], normed.unsqueeze(-1).to(tail.dtype)], dim=-1)
            self._conv_tail.index_copy_(0, idx, new_tail)
            self._tok_hist.index_copy_(0, idx, hist[:, 1:])
            return gated + conv

        # ---- prefill (varlen; per-request loop, correctness over speed in P0) ----
        if fla.fresh_state_indices is not None:
            self._reset_slots(fla.fresh_state_indices)
        cu = fla.cu_seqlens.tolist()
        out = torch.empty_like(x4)
        w = self._conv_weight()
        for r in range(len(cu) - 1):
            s, e = cu[r], cu[r + 1]
            slot = int(fla.cache_indices[r])
            ids_r = input_ids[s:e].view(1, -1)  # [1, T]
            hist = torch.cat([self._tok_hist[slot].view(1, -1), ids_r], dim=-1)
            emb = self._ngram_embed(hist, e - s, x4.dtype)[0]  # [T, D]
            gated = self._ple_core(x4[s:e], emb)  # [T, C]
            normed = grouped_rms_norm(gated, self.norm_conv.weight, self.hidden_size, self.eps)
            tail = self._conv_tail[slot]  # [C, 9]
            seq = torch.cat([tail.to(normed.dtype), normed.t()], dim=-1)  # [C, 9+T]
            conv = F.conv1d(
                seq.unsqueeze(0), w.to(normed.dtype), dilation=self.conv_dilation,
                groups=w.shape[0],
            )[0]  # [C, T]
            conv = F.silu(conv)
            out[s:e] = gated + conv.t()
            # persist state
            self._conv_tail[slot] = seq[:, -self.state_len:].to(self._conv_tail.dtype)
            self._tok_hist[slot] = hist[0, -self.context_len:]
        return out


class _W(BaseOP):
    def __init__(self, out_features: int, in_features: int):
        self.weight = torch.empty(out_features, in_features)


class _N(BaseOP):
    def __init__(self, dim: int):
        self.weight = torch.empty(dim)


class _Conv(BaseOP):
    def __init__(self, channels: int, kernel: int):
        self.weight = torch.empty(channels, 1, kernel)


__all__ = ["Qwen4PLELayer"]
