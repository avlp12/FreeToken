"""DeepSeek-V4-Flash support for FreeToken.

This package ports the official ``inference/model.py`` reference (MLA + CSA/HCA
compressors + Lightning Indexer + manifold-constrained Hyper-Connections) onto
FreeToken's primitives. The exotic ops are reimplemented as Triton kernels (see
``freetoken/kernel/triton/dsv4_*.py``); the routed FP4 experts are served from
FreeToken's :class:`~freetoken.moe.offload_cache.OffloadMoeCache` so only a subset
of experts is resident on the GPU (the framework's core acceleration).

DeepSeek-V4-Flash is a first-class registered model on the shared paged-KV engine:
its window / compressed-attention / compressed-index KV live in DSV4-owned pools
addressed by page tables, and sparse attention is a physical-slot gather (see
:mod:`freetoken.attention.dsv4_sparse` and :mod:`freetoken.kvcache.dsv4_paged_pool`).
"""

from .args import DeepseekV4Args, load_args
from .config import parse_config
from .model import DeepseekV4ForCausalLM
from .weight import adapt_weights, iter_weights, load_dsfp4_expert_sources

__all__ = [
    "DeepseekV4Args",
    "load_args",
    "parse_config",
    "DeepseekV4ForCausalLM",
    "iter_weights",
    # models.weight.load_weight's per-model load-time hook (storage form -> runtime form;
    # see weight.py's module docstring). It has to be reachable from the PACKAGE, because
    # ModelSpec.module names the package rather than the weight module.
    "adapt_weights",
    "load_dsfp4_expert_sources",
]
