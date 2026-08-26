"""Per-layer copy widths, end to end through OffloadMoeCache.

The kernel-level contract lives in tests/kernels/test_pitched_index_copy.py. What
this file covers is the part that only exists once a real cache is holding real
banks: resolving a per-(bank, layer) width table into one descriptor per layer, and
the fact that "narrow" is a per-LAYER property.

The case that actually broke a server boot: in the shipped q2_k_ud table, gate_up
layer 26 is IQ3_XXS, whose native row IS that bank's pitch. So layer 26 has nothing
to narrow while every other layer does -- its descriptor is None while the shared
weight-row stride is not -- and the kernel takes the two together or not at all.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

GU_PITCH, GU_NATIVE = 1568, 1184
DN_PITCH, DN_NATIVE = 1088, 784
EXPERTS, SLOTS, LAYERS = 8, 8, 3
GU_ROWS, DN_ROWS = 16, 8
POISON = 0xAB


# The shipped shape in miniature: gate_up narrows on most layers but not on layer 1
# (its native row IS the pitch, like the real layer 26), and down carries a native
# width the DECODE path is told to ignore.
WIDTHS = {
    "gate_up": [GU_NATIVE, GU_PITCH, GU_NATIVE],
    "down": [DN_NATIVE, DN_NATIVE, DN_PITCH],
}
DECODE_NARROW = ("gate_up",)


def _cache(prefill_overlap: bool = False):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.moe.offload_cache import OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    # The double buffers borrow 2 * num_experts slots off the front of the cache.
    slots = max(SLOTS, 2 * EXPERTS) if prefill_overlap else SLOTS
    cache = OffloadMoeCache(
        num_layers=LAYERS,
        num_experts=EXPERTS,
        cache_size=slots,
        device=torch.device("cuda"),
        quant_format="q2_k_ud",
        prefill_overlap=prefill_overlap,
    )
    g = torch.Generator().manual_seed(0)
    sources = {
        "gate_up": [
            torch.randint(0, 256, (EXPERTS, GU_ROWS, GU_PITCH), generator=g, dtype=torch.uint8)
            .pin_memory()
            for _ in range(LAYERS)
        ],
        "down": [
            torch.randint(0, 256, (EXPERTS, DN_ROWS, DN_PITCH), generator=g, dtype=torch.uint8)
            .pin_memory()
            for _ in range(LAYERS)
        ],
    }
    # Real banks leave the bytes past a row's native width zero -- HostBank never
    # writes them -- and the width declaration is verified against exactly that.
    for name, widths in WIDTHS.items():
        for layer_id, w in enumerate(widths):
            sources[name][layer_id][:, :, w:] = 0
    cache.set_bank_sources(sources)
    assert cache._copy_fused_ok, "the fused copy plan is a precondition for this test"
    return cache, sources


def test_descriptor_is_present_only_where_there_is_something_to_narrow():
    cache, _ = _cache()
    assert all(w is None for w in cache._copy_bytes), "widths before they were declared"

    cache.set_layer_copy_bytes(WIDTHS, decode_narrow=DECODE_NARROW)
    args = [cache._copy_width_args(i) for i in range(LAYERS)]
    assert set(args[0]) == {"copy_bytes", "row_pitch"}
    assert args[1] == {}, "a layer with nothing to narrow must pass NEITHER descriptor"
    assert set(args[2]) == {"copy_bytes", "row_pitch"}
    # down carries a native width, but decode_narrow excludes it, so the DECODE
    # descriptor reports its full pitch...
    assert args[0]["copy_bytes"].tolist() == [GU_NATIVE, DN_PITCH]
    assert args[0]["row_pitch"].tolist() == [GU_PITCH, DN_PITCH]
    # ...while the PREFILL table keeps the honest widths for both banks. The two paths
    # are different hardware and the split between them is a measurement, not a rule.
    assert cache._prefill_widths[0] == [GU_NATIVE, DN_NATIVE]
    assert cache._prefill_widths[2] == [GU_NATIVE, DN_PITCH]
    # One shared stride tensor, not one per layer: it is a property of the banks.
    assert args[0]["row_pitch"].data_ptr() == args[2]["row_pitch"].data_ptr()


def test_clearing_and_rebuilding_restore_the_full_pitch():
    cache, _ = _cache()
    cache.set_layer_copy_bytes(WIDTHS, decode_narrow=DECODE_NARROW)
    assert cache._copy_width_args(0)

    cache.set_layer_copy_bytes(None)
    assert all(cache._copy_width_args(i) == {} for i in range(LAYERS))
    assert all(w == [GU_PITCH, DN_PITCH] for w in cache._prefill_widths)

    # rebuild() reallocates the slot caches and re-runs the plan; a declaration made
    # before it must survive, or the first decode after a cache resize silently goes
    # back to full-pitch copies.
    cache.set_layer_copy_bytes(WIDTHS, decode_narrow=DECODE_NARROW)
    cache.rebuild(SLOTS + 4)
    assert cache._copy_width_args(0)
    assert cache._copy_width_args(1) == {}
    assert cache._prefill_widths[0] == [GU_NATIVE, DN_NATIVE]


def test_rejects_a_width_the_copy_unit_cannot_express():
    cache, _ = _cache()
    with pytest.raises(ValueError, match="multiple of 16"):
        cache.set_layer_copy_bytes({"gate_up": [GU_NATIVE - 1] * LAYERS})
    with pytest.raises(ValueError, match="multiple of 16"):
        cache.set_layer_copy_bytes({"gate_up": [GU_PITCH + 16] * LAYERS})
    with pytest.raises(ValueError, match="one per layer"):
        cache.set_layer_copy_bytes({"gate_up": [GU_NATIVE]})
    with pytest.raises(ValueError, match="unknown bank"):
        cache.set_layer_copy_bytes({"nope": [16] * LAYERS})
    with pytest.raises(ValueError, match="decode_narrow"):
        cache.set_layer_copy_bytes(WIDTHS, decode_narrow=("nope",))


def test_slot_cache_starts_zeroed():
    """Not load-bearing, but it keeps a cold slot deterministic instead of leaving
    allocator residue in bytes the copy path deliberately stops writing."""
    cache, _ = _cache()
    for name, slot_cache in cache.bank_caches.items():
        assert not slot_cache.any(), f"bank {name!r} slot cache was not zero-initialized"
    cache.rebuild(SLOTS + 4)
    for name, slot_cache in cache.bank_caches.items():
        assert not slot_cache.any(), f"bank {name!r} not zeroed after rebuild"


def test_prefill_layer_copy_moves_the_payload_and_leaves_the_tail():
    """The prefill fill narrows BOTH banks, through the copy engine's 2D mode."""
    cache, sources = _cache(prefill_overlap=True)
    cache.set_layer_copy_bytes(WIDTHS, decode_narrow=DECODE_NARROW)
    assert cache.prefill_bank_buffers, "prefill overlap buffers were not built"
    for _, slot_cache in cache.banks:
        slot_cache.fill_(POISON)

    layer_id, buffer_id = 0, 0
    cache._prefill_layer_copy(layer_id, buffer_id)
    torch.cuda.synchronize()

    poison = torch.full((1,), POISON, dtype=torch.uint8, device="cuda")
    for b, name in enumerate(cache.bank_schema):
        src = sources[name][layer_id].cuda()
        got = cache.prefill_bank_buffers[b][buffer_id]
        w = WIDTHS[name][layer_id]
        assert torch.equal(got[:, :, :w], src[:, :, :w]), f"{name} payload differs"
        assert (got[:, :, w:] == poison).all(), (
            f"{name} tail was written -- the fill is still moving the padding"
        )


def test_a_truncating_width_is_rejected_against_the_host_bank():
    """The evidence that a declared width is the WHOLE packed row.

    A width that is too small silently truncates real weights, and the truncated rows
    still decode to finite-looking numbers because the IQ grids are total over their
    lookup tables -- so the failure mode is a quietly worse model. The check lives on
    the host banks because a device slot's tail may legitimately hold another layer's
    weights once copies stop rewriting whole rows.
    """
    cache, sources = _cache()
    # Payload out to the full pitch, so ANY narrowing would be a truncation.
    for name in cache.bank_schema:
        for t in sources[name]:
            t.fill_(0x11)
    with pytest.raises(ValueError, match="is not the whole"):
        cache.set_layer_copy_bytes(WIDTHS, decode_narrow=DECODE_NARROW)


def test_declaring_widths_retires_the_device_pad_tail_check():
    """The device-side check would now fire on correct data: slots are shared across
    layers, and a layer at the full pitch has no padding, so its real weights end up in
    a narrower layer's tail."""
    from freetoken.moe import prefill_dequant_gemm as _dq

    cache, _ = _cache()
    _dq._PAD_CHECK_ENABLED = True
    try:
        cache.set_layer_copy_bytes(WIDTHS, decode_narrow=DECODE_NARROW)
        assert not _dq._PAD_CHECK_ENABLED
    finally:
        _dq._PAD_CHECK_ENABLED = True


@pytest.mark.parametrize("layer_id", [0, 1])
def test_copy_missing_moves_the_payload_and_leaves_the_tail(layer_id):
    """The real path: stage this layer's misses, run the fused copy, inspect the slots.

    Layer 0 narrows gate_up, layer 1 does not -- so the same assertions run once with
    the pitched kernel and once with the original one.
    """
    cache, sources = _cache()
    cache.set_layer_copy_bytes(WIDTHS, decode_narrow=DECODE_NARROW)
    for _, slot_cache in cache.banks:
        slot_cache.fill_(POISON)

    ids = torch.arange(EXPERTS, dtype=torch.int32, device="cuda").reshape(1, EXPERTS)
    cache.ensure_experts(layer_id, ids)
    cache.copy_missing()
    torch.cuda.synchronize()

    slots = ids.reshape(-1).long()  # ensure_experts rewrote ids in place -> slot ids
    poison = torch.full((1,), POISON, dtype=torch.uint8, device="cuda")
    # decode_narrow excludes down, so only gate_up's declared width applies here.
    expect = {"gate_up": WIDTHS["gate_up"][layer_id], "down": DN_PITCH}
    for name in ("gate_up", "down"):
        src = sources[name][layer_id].cuda()
        got = cache.bank_caches[name][slots]
        w = expect[name]
        assert torch.equal(got[:, :, :w], src[:, :, :w]), f"{name} payload differs"
        if w < src.shape[2]:
            assert (got[:, :, w:] == poison).all(), (
                f"{name} tail was written -- the copy is still moving the padding"
            )
