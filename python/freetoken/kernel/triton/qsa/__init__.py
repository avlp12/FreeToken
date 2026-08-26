"""Triton kernels for Qwen QSA (indexer + selected-token GQA).

The first backend version (``attention/qsa_sparse.py`` +
``attention/qsa_indexer.py``) is **PyTorch-only** and is the numerical
source of truth. Kernelize against those functions; do not change the
score / selection / RMSNorm / block-start RoPE semantics.

TODO(kernelize)
---------------
* ``qsa_index_decode``: fused gather of post-ln-post-rope pooled keys off
  the decode row snapshot + ReLU-dot + 4-head sum + ``/sqrt(128)`` +
  top-512. Live block count is read from device ``kvlen`` (CUDA-graph
  tracks the position, not the staged width). Output: ``[bs, 512]`` block
  ids and a ``[bs, 3]`` tail (or a packed ``[bs, 2051]`` token-pos list).
* ``qsa_index_prefill``: tiled ``index_q @ pooled_k.T`` for a query chunk
  with a causal complete-block mask (``b >= (pos+1)//4`` → ``-inf``).
* ``qsa_sparse_attn_decode``: gathered GQA over the 2051 selected slots
  (``-1`` masked), split-K optional.
* ``qsa_sparse_attn_prefill``: per-query gathered GQA; replace the
  mask-over-full ``O(T·L)`` PyTorch path for ``L > 2051``.

Until those land, ``QSASparseAttnBackend`` calls the PyTorch helpers in
``qsa_indexer``.
"""
