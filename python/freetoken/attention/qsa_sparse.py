"""QSA sparse-attention backend (``--attention-backend qsa``).

Serves Qwen3.8-Flash-Next / qwen4_exp long-context **QSA**: GQA (head_dim 128,
``num_kv_heads=2``) with an indexer-selected token set. First version is
PyTorch-only (numerical source of truth); Triton kernelization is listed in
``kernel/triton/qsa/``.

Backend name
------------
``qsa`` (registered in ``attention/__init__.py``). Also declares
``AttnType.FULL`` so a paged-GQA ``FullAttentionGroupConfig`` can attach
before ``AttnType.QSA`` lands on the group spec. Auto-resolve will **not**
pick this backend for FULL-only models -- pass ``--attention-backend qsa``.
Hybrid GDN + QSA is allowed (``hybrid_linear_ok=True``).

Consumed ModelConfig fields
---------------------------
* ``num_qo_heads``, ``num_kv_heads``, ``head_dim``, ``attn_sm_scale``,
  ``rms_norm_eps``, ``num_layers``.
* optional ``qwen4_exp_args`` (or the same names on the config) via
  :class:`freetoken.attention.qsa_indexer.QSAIndexerConfig`:
  ``indexer_budget`` (2048), ``indexer_compress_ratio`` (4),
  ``indexer_n_heads`` (4), ``indexer_kv_heads`` (1),
  ``indexer_head_dim`` (128), ``rms_norm_eps``.

Do **not** set ``FullAttentionGroupConfig.index_head_dim``: that selects
``BSAKVCache`` (MiniMax-M3, page_size 128). QSA reuses the existing paged
GQA pool (``MHAKVCache``, any page size) and keeps indexer keys in a
**linear** buffer owned here (token × 128 bf16; 256K → 64 MiB/layer).

Interface for ``models/qwen4_exp/attention.py``
----------------------------------------------
Projections, q_layernorm, and index-q RoPE stay in the model.
``k_layernorm`` + block-start RoPE run **here** on pooled keys (official
``Qwen4ExpTextQSAIndexer``).

.. code-block:: python

    o = ctx.attn_backend.qsa_forward(
        q, k, v,            # GQA, already q/k-norm + RoPE; k/v [T, Hkv, D] or flat
        index_q,            # [T, 4, 128] q_layernorm + RoPE already applied
        index_k,            # [T, 128] RAW token keys — no norm, no RoPE
        layer_id,
        ctx.batch,
        k_ln_weight,        # [128] indexer k_layernorm.weight (Qwen 1+w)
        rope_cos, rope_sin, # [max_pos, 128] same table used to RoPE index_q
    )
    # o: [T, Hq, D]  — model applies sigmoid(gate) and o_proj

Generic ``forward(q,k,v)`` raises: this is not a drop-in FULL backend.

Cache / graph
-------------
* GQA K/V: ``kvcache.store_kv`` (paged rows via ``batch.out_loc``).
* Indexer raw keys: linear ``QSAIndexerKeyCache.raw_k[layer, out_loc]``.
* Pooled keys: published at the block-start physical row, post-ln+RoPE.
  Prefill rebuilds complete blocks eagerly; decode does a fixed-shape
  masked publish (dummy row for incomplete / pad) so the captured graph
  never reallocates or Python-branches on ``pos % 4``.
* Decode addressing is a **snapshot** of page-table rows + live lengths
  staged into static buffers (``prepare_for_replay``, dsv4/dsa precedent)
  so the next batch's ``allocate_paged`` cannot redirect an in-flight
  replay. Score / top-k / gather / attend use static widths
  (``max_blocks = stage_width // 4``, ``max_selected = 2051``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from freetoken.core import Batch, get_global_ctx
from freetoken.utils import init_logger

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata
from .qsa_indexer import (
    QSAIndexerConfig,
    QSAIndexerKeyCache,
    causal_additive_mask,
    gqa_attend,
    qsa_score_blocks,
    qsa_select_tokens,
    selected_bool_mask,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

_CPU_PINNED = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
_PREFILL_SCORE_BYTES = 128 << 20
_PREFILL_SCORE_CHUNK = 512


@dataclass
class QSAMetadata(BaseAttnMetadata):
    is_decode: bool
    last_indices: torch.Tensor
    qo_indptr_cpu: torch.Tensor
    kv_len_cpu: torch.Tensor
    # Decode snapshot: position-ordered physical rows [bs, W] + live lengths.
    rows: torch.Tensor | None = None
    kvlen: torch.Tensor | None = None

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


class QSASparseAttnBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.qsa = QSAIndexerConfig.from_model_config(config)
        self.num_heads = int(config.num_qo_heads)
        self.num_kv_heads = int(config.num_kv_heads)
        self.head_dim = int(config.head_dim)
        self.sm_scale = config.attn_sm_scale or (self.head_dim**-0.5)
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        self._idx_cache: QSAIndexerKeyCache | None = None
        self._rows_buf: torch.Tensor | None = None
        self._kvlen_buf: torch.Tensor | None = None
        self.max_seq_len = 0
        self.capture_bs: List[int] = []
        self._graph_bound = False
        logger.info(
            "qsa backend: budget=%d ratio=%d block_topk=%d max_sel=%d "
            "GQA Hq=%d Hkv=%d D=%d (PyTorch path; see kernel/triton/qsa TODOs)",
            self.qsa.indexer_budget,
            self.qsa.indexer_compress_ratio,
            self.qsa.block_topk,
            self.qsa.max_selected,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )

    def forward(self, q, k, v, layer_id, batch, attn_spec: AttentionSpec | None = None):
        raise NotImplementedError(
            "QSA layers call qsa_forward(q, k, v, index_q, index_k, layer_id, "
            "batch, k_ln_weight, rope_cos, rope_sin); see qsa_sparse.py."
        )

    # ----- metadata / CUDA graph ------------------------------------------------
    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        seqlens_q = [r.extend_len for r in reqs]
        seqlens_k = [r.device_len for r in reqs]
        # Follow the BATCH PHASE (dsa precedent): a 1-token prefill is not decode.
        is_decode = getattr(batch, "phase", None) == "decode"
        qo_indptr = torch.tensor([0] + seqlens_q, **_CPU_PINNED).cumsum_(0).to(torch.int32)
        kv_len = torch.tensor(seqlens_k, **_CPU_PINNED)
        last = (qo_indptr[1:].to(torch.int32) - 1).to(self.device, non_blocking=True)
        batch.attn_metadata = QSAMetadata(
            is_decode=is_decode,
            last_indices=last,
            qo_indptr_cpu=qo_indptr,
            kv_len_cpu=kv_len,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.max_seq_len = max_seq_len
        self.capture_bs = sorted(bs_list)
        max_bs = max(bs_list)
        width = get_global_ctx().page_table.shape[1]
        self._rows_buf = torch.full((max_bs, width), -1, dtype=torch.int32, device=self.device)
        self._kvlen_buf = torch.zeros(max_bs, dtype=torch.int32, device=self.device)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.prepare_metadata(batch)
        bs = batch.size
        dummy = torch.full(
            (bs,), batch.padded_reqs[0].table_idx, dtype=torch.int64, device=self.device
        )
        self._graph_bound = True
        self._stage_decode(batch, bs, dummy)

    def prepare_for_replay(self, batch: Batch) -> None:
        assert batch.active_table_idx is not None, "decode batch is missing its page-table rows"
        self._graph_bound = True
        self._stage_decode(
            batch, batch.padded_size, batch.active_table_idx.to(torch.int64)
        )

    def reset_capture(self) -> None:
        super().reset_capture()
        self._rows_buf = None
        self._kvlen_buf = None
        self._graph_bound = False

    def _decode_rows(self, batch: Batch) -> torch.Tensor:
        assert batch.active_table_idx is not None, "decode batch is missing its page-table rows"
        return get_global_ctx().page_table.index_select(
            0, batch.active_table_idx.to(torch.int64)
        )

    def _stage_decode(self, batch: Batch, bs: int, table_idx: torch.Tensor) -> None:
        md = batch.attn_metadata
        assert isinstance(md, QSAMetadata)
        assert self._rows_buf is not None and self._kvlen_buf is not None
        src = get_global_ctx().page_table.index_select(0, table_idx)
        w = min(src.shape[1], self._rows_buf.shape[1])
        self._rows_buf[:bs, :w].copy_(src[:bs, :w])
        self._kvlen_buf[:bs].copy_(md.kv_len_cpu.to(self.device, non_blocking=True))
        md.rows = self._rows_buf[:bs]
        md.kvlen = self._kvlen_buf[:bs]

    # ----- indexer cache --------------------------------------------------------
    def _n_slots(self) -> int:
        cache = self.kvcache
        for lid in range(int(self.config.num_layers)):
            try:
                k = cache.k_cache(lid)
            except (KeyError, IndexError, AssertionError):
                continue
            if k.dim() >= 4:
                return int(k.shape[0] * k.shape[1])
            return int(k.shape[0])
        width = int(get_global_ctx().page_table.shape[1])
        return max(width, 1)

    def _ensure_idx_cache(self, dtype: torch.dtype) -> QSAIndexerKeyCache:
        n = self._n_slots()
        cur = self._idx_cache
        if cur is None or cur.n_slots != n or cur.dtype != dtype:
            self._idx_cache = QSAIndexerKeyCache(
                num_layers=int(self.config.num_layers),
                n_slots=n,
                dim=self.qsa.indexer_head_dim,
                device=self.device,
                dtype=dtype,
            )
        return self._idx_cache

    def _as_kv(self, t: torch.Tensor) -> torch.Tensor:
        return t.reshape(-1, self.num_kv_heads, self.head_dim)

    def _k_rows(self, layer_id: int) -> torch.Tensor:
        cache = self.kvcache.k_cache(layer_id)
        return cache.view(-1, cache.shape[-2], cache.shape[-1])

    def _v_rows(self, layer_id: int) -> torch.Tensor:
        cache = self.kvcache.v_cache(layer_id)
        return cache.view(-1, cache.shape[-2], cache.shape[-1])

    # ----- public entry ---------------------------------------------------------
    def qsa_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        layer_id: int,
        batch: Batch,
        k_ln_weight: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Store GQA + raw indexer keys, select, attend. Returns ``[T, Hq, D]``."""
        md = batch.attn_metadata
        assert isinstance(md, QSAMetadata), "prepare_metadata did not run for this batch"
        if md.is_decode and md.rows is None:
            md.rows = self._decode_rows(batch).to(torch.int32)
            md.kvlen = md.kv_len_cpu.to(self.device, non_blocking=True)

        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        idx = self._ensure_idx_cache(index_k.dtype)
        idx.store_raw(layer_id, index_k.reshape(-1, self.qsa.indexer_head_dim), batch.out_loc)

        q = q.reshape(-1, self.num_heads, self.head_dim)
        index_q = index_q.reshape(-1, self.qsa.indexer_n_heads, self.qsa.indexer_head_dim)
        if md.is_decode:
            return self._decode(
                md, layer_id, q, index_q, k_ln_weight, rope_cos, rope_sin
            )
        return self._prefill(
            md, layer_id, q, index_q, batch, k_ln_weight, rope_cos, rope_sin
        )

    # ----- decode (CUDA-graph capturable, single code path when bound) ----------
    def _decode(
        self,
        md: QSAMetadata,
        layer_id: int,
        q: torch.Tensor,
        index_q: torch.Tensor,
        k_ln_weight: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        assert md.rows is not None and md.kvlen is not None
        rows, kvlen = md.rows, md.kvlen
        pos = (kvlen.long() - 1).clamp_min(0)
        idx = self._idx_cache
        assert idx is not None
        idx.update_pooled_decode(
            layer_id, rows, pos, k_ln_weight, rope_cos, rope_sin, self.qsa
        )

        # Eager short context: identity selection == dense (bit-identical path).
        # A captured graph cannot Python-branch on kv_len; it always takes the
        # fixed-shape sparse path below (all live blocks win top-k when
        # kv_len <= 2051, so the selected SET is still dense).
        if not self._graph_bound:
            host_len = int(md.kv_len_cpu.max()) if md.kv_len_cpu.numel() else 0
            if host_len <= self.qsa.dense_token_limit:
                return self._decode_dense(q, rows, kvlen, layer_id)

        return self._decode_sparse(q, index_q, rows, kvlen, layer_id)

    def _decode_dense(
        self,
        q: torch.Tensor,
        rows: torch.Tensor,
        kvlen: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        bs = q.shape[0]
        max_l = int(kvlen.max().item()) if kvlen.numel() else 0
        if max_l <= 0:
            return q.new_zeros(q.shape)
        k_rows, v_rows = self._k_rows(layer_id), self._v_rows(layer_id)
        safe = rows[:, :max_l].clamp_min(0).long()
        k = k_rows[safe].permute(0, 2, 1, 3)  # [B, Hkv, L, D]
        v = v_rows[safe].permute(0, 2, 1, 3)
        qh = q.unsqueeze(2)  # [B, Hq, 1, D]
        pos = (kvlen.long() - 1).clamp_min(0)
        causal = causal_additive_mask(pos, max_l, q.device, q.dtype)
        live = torch.arange(max_l, device=q.device).unsqueeze(0) < kvlen.unsqueeze(1)
        neg = torch.tensor(float("-inf"), dtype=q.dtype, device=q.device)
        pad = torch.where(live, torch.zeros((), dtype=q.dtype, device=q.device), neg).view(
            bs, 1, 1, max_l
        )
        out = gqa_attend(qh, k, v, self.sm_scale, attn_mask=causal + pad)
        return out.reshape(bs, self.num_heads, self.head_dim)

    def _decode_sparse(
        self,
        q: torch.Tensor,
        index_q: torch.Tensor,
        rows: torch.Tensor,
        kvlen: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        # TODO(kernelize): fuse gather-pooled + relu-dot + topk (kernel/triton/qsa).
        ratio = self.qsa.indexer_compress_ratio
        n_blocks = rows.shape[1] // ratio
        block_rows = rows[:, 0 : n_blocks * ratio : ratio]
        idx = self._idx_cache
        assert idx is not None
        if n_blocks == 0:
            sel_pos = kvlen.new_full((q.shape[0], self.qsa.max_selected), -1)
        else:
            block_k = idx.gather_pooled_blocks(layer_id, block_rows)
            scores = qsa_score_blocks(index_q, block_k)
            live = ((torch.arange(n_blocks, device=q.device) + 1) * ratio) <= kvlen.unsqueeze(1)
            scores = scores.masked_fill(~live, float("-inf"))
            n_pick = min(self.qsa.block_topk, n_blocks)
            vals, picks = scores.topk(n_pick, dim=-1)
            valid = torch.isfinite(vals)
            offs = torch.arange(ratio, device=q.device)
            tok = picks.unsqueeze(-1) * ratio + offs
            tok = torch.where(valid.unsqueeze(-1), tok, tok.new_full((), -1)).reshape(
                q.shape[0], n_pick * ratio
            )
            n_complete = kvlen.long() // ratio
            tail_w = ratio - 1
            tail_base = n_complete * ratio
            tail = tail_base.unsqueeze(1) + torch.arange(tail_w, device=q.device)
            tail = torch.where(tail < kvlen.unsqueeze(1), tail, tail.new_full((), -1))
            sel_pos = torch.cat([tok, tail], dim=-1)
            if sel_pos.shape[-1] < self.qsa.max_selected:
                sel_pos = torch.cat(
                    [
                        sel_pos,
                        sel_pos.new_full(
                            (q.shape[0], self.qsa.max_selected - sel_pos.shape[-1]), -1
                        ),
                    ],
                    dim=-1,
                )
        return self._attend_gathered(q, sel_pos.to(torch.int32), rows, layer_id)

    def _attend_gathered(
        self,
        q: torch.Tensor,
        sel_pos: torch.Tensor,
        rows: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        # TODO(kernelize): gathered GQA over max_selected slots.
        bs, n_sel = sel_pos.shape
        width = rows.shape[1]
        in_range = (sel_pos >= 0) & (sel_pos < width)
        phys = rows.gather(1, sel_pos.clamp(0, width - 1).long())
        phys = torch.where(in_range, phys, phys.new_full((), -1))
        k_rows, v_rows = self._k_rows(layer_id), self._v_rows(layer_id)
        k = k_rows[phys.clamp_min(0).long()]  # [B, S, Hkv, D]
        v = v_rows[phys.clamp_min(0).long()]
        qh = q.unsqueeze(2)
        k_h = k.permute(0, 2, 1, 3)
        v_h = v.permute(0, 2, 1, 3)
        keep = (phys >= 0) & in_range
        neg = torch.tensor(float("-inf"), dtype=q.dtype, device=q.device)
        add = torch.where(keep, torch.zeros((), dtype=q.dtype, device=q.device), neg).view(
            bs, 1, 1, n_sel
        )
        out = gqa_attend(qh, k_h, v_h, self.sm_scale, attn_mask=add)
        return out.reshape(bs, self.num_heads, self.head_dim)

    # ----- prefill / extend (eager, vectorized per request) ---------------------
    def _prefill(
        self,
        md: QSAMetadata,
        layer_id: int,
        q: torch.Tensor,
        index_q: torch.Tensor,
        batch: Batch,
        k_ln_weight: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        page_table = get_global_ctx().page_table
        qo = md.qo_indptr_cpu.tolist()
        out = q.new_empty(q.shape[0], self.num_heads, self.head_dim)
        idx = self._idx_cache
        assert idx is not None
        for i, r in enumerate(reqs):
            s0, s1 = qo[i], qo[i + 1]
            if s1 <= s0:
                continue
            kv_len = int(r.device_len)
            rows = page_table[r.table_idx, :kv_len].to(torch.int32)
            idx.publish_complete_blocks(
                layer_id, rows, kv_len, k_ln_weight, rope_cos, rope_sin, self.qsa
            )
            q_i = q[s0:s1]
            iq_i = index_q[s0:s1]
            pos = batch.positions[s0:s1]
            if kv_len <= self.qsa.dense_token_limit:
                out[s0:s1] = self._prefill_dense(q_i, rows, pos, kv_len, layer_id)
                continue
            # Chunk queries so the fp32 [chunk, n_blocks] score tile stays bounded.
            n_blocks = max(kv_len // self.qsa.indexer_compress_ratio, 1)
            chunk = max(16, min(_PREFILL_SCORE_CHUNK, _PREFILL_SCORE_BYTES // max(n_blocks * 4, 1)))
            raw_seq = idx.raw_k[layer_id][rows.long()]
            for c0 in range(0, s1 - s0, chunk):
                c1 = min(c0 + chunk, s1 - s0)
                sel = qsa_select_tokens(
                    iq_i[c0:c1],
                    raw_seq,
                    pos[c0:c1],
                    k_ln_weight,
                    rope_cos,
                    rope_sin,
                    self.qsa,
                )
                out[s0 + c0 : s0 + c1] = self._prefill_from_sel(
                    q_i[c0:c1], sel, rows, pos[c0:c1], kv_len, layer_id
                )
        return out

    def _prefill_dense(
        self,
        q: torch.Tensor,
        rows: torch.Tensor,
        positions: torch.Tensor,
        kv_len: int,
        layer_id: int,
    ) -> torch.Tensor:
        k_rows, v_rows = self._k_rows(layer_id), self._v_rows(layer_id)
        k = k_rows[rows.long()].permute(1, 0, 2).unsqueeze(0)  # [1, Hkv, L, D]
        v = v_rows[rows.long()].permute(1, 0, 2).unsqueeze(0)
        qh = q.unsqueeze(0).permute(0, 2, 1, 3)  # [1, Hq, Q, D]
        causal = causal_additive_mask(positions, kv_len, q.device, q.dtype)
        out = gqa_attend(qh, k, v, self.sm_scale, attn_mask=causal)
        return out.reshape(q.shape[0], self.num_heads, self.head_dim)

    def _prefill_from_sel(
        self,
        q: torch.Tensor,
        sel: torch.Tensor,
        rows: torch.Tensor,
        positions: torch.Tensor,
        kv_len: int,
        layer_id: int,
    ) -> torch.Tensor:
        """Mask-over-full GQA (official) so a complete selection is bit-dense.

        Long contexts still go through this path; the selected mask is sparse
        in *values* (most -inf) even though the score axis is ``kv_len``.
        # TODO(kernelize): switch long prefills to gathered-KV GQA.
        """
        bool_sel = selected_bool_mask(sel, kv_len)
        causal = positions.to(q.device).unsqueeze(1) >= torch.arange(kv_len, device=q.device)
        keep = bool_sel & causal
        neg = torch.tensor(float("-inf"), dtype=q.dtype, device=q.device)
        add = torch.where(keep, torch.zeros((), dtype=q.dtype, device=q.device), neg).view(
            1, 1, q.shape[0], kv_len
        )
        k_rows, v_rows = self._k_rows(layer_id), self._v_rows(layer_id)
        k = k_rows[rows.long()].permute(1, 0, 2).unsqueeze(0)
        v = v_rows[rows.long()].permute(1, 0, 2).unsqueeze(0)
        qh = q.unsqueeze(0).permute(0, 2, 1, 3)
        out = gqa_attend(qh, k, v, self.sm_scale, attn_mask=add)
        return out.reshape(q.shape[0], self.num_heads, self.head_dim)


__all__ = ["QSASparseAttnBackend", "QSAMetadata"]
