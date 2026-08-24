from __future__ import annotations

from dataclasses import dataclass, field

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    # True when --max-prefill-length was given on the CLI. Models that widen the chunk for
    # single-pass prefill (DSV4) must not clobber an explicit user value.
    max_extend_tokens_explicit: bool = False
    # Host-RAM KV tier budget in GiB (0 = disabled). swa_radix (DSV4) only: evicted full-KV
    # spans are snapshotted to host RAM and restored at admission instead of re-prefilling.
    host_kv_cache_gb: float = 0.0
    # Disk budget for the host KV tier in GiB (0 = RAM-only). Entries are flushed to
    # ~/.cache/freetoken/hostkv/<model>/ and reloaded at startup: sessions survive restarts.
    host_kv_disk_gb: float = 0.0
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/freetoken_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/freetoken_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/freetoken_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
