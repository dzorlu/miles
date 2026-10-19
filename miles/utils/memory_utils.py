import ctypes
import gc
import logging
import sys

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def clear_memory(clear_host_memory: bool = False):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    """GPU memory of the current device (``total``, ``free`` and ``used`` are
    device-wide, ``allocated`` and ``reserved`` are this process's allocator)
    plus this process's host memory (``host``, see ``host_memory``)."""
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.cuda.memory_reserved(device)),
        "host": host_memory(),
    }


_STATUS_FIELDS = {
    "VmRSS": "rss_GB",
    "VmHWM": "rss_peak_GB",
    "RssAnon": "anon_GB",
    "RssShmem": "shmem_GB",
    "RssFile": "file_GB",
}


def host_memory(status_text: str | None = None) -> dict:
    """This process's host memory: resident set and its peak, the anonymous
    heap, the shared part (pinned CUDA host memory is counted here) and the
    file-backed part, from ``/proc/self/status``; plus the pinned-memory
    allocator's current and reserved bytes where torch reports them. A
    colocated trainer keeps pinned offload state and a heap that peaks at
    the rollout shard it deserialised; the two move for different reasons."""
    if status_text is None:
        try:
            status_text = open("/proc/self/status").read()
        except OSError:
            status_text = ""
    out = {}
    for line in status_text.splitlines():
        key, _, rest = line.partition(":")
        if key in _STATUS_FIELDS:
            out[_STATUS_FIELDS[key]] = round(int(rest.split()[0]) * 1024 / (1024**3), 2)
    stats = getattr(torch.cuda, "host_memory_stats", None)
    if stats is not None and torch.cuda.is_available():
        try:
            host = stats()
            out["pinned_allocated_GB"] = _byte_to_gb(host.get("allocated_bytes.all.current", 0))
            out["pinned_reserved_GB"] = _byte_to_gb(host.get("reserved_bytes.all.current", 0))
        except Exception as e:  # noqa: BLE001
            out["pinned_error"] = str(e)
    return out


def trim_host_heap() -> bool:
    """Return the heap's free pages to the OS (glibc ``malloc_trim``): a
    process that deserialised a large rollout shard keeps the whole peak as
    resident memory otherwise, and the peak grows with every step whose data
    is bigger than any before. True when the call was made."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        return True
    except (OSError, AttributeError):
        return False


def release_host_memory() -> None:
    """After a train step: drop garbage, give the pinned-memory cache back to
    CUDA (the replay queues pin buffers of a new size every step) and trim the
    heap, so the host footprint between steps stays at the state the trainer
    keeps, not at the largest step so far."""
    gc.collect()
    if hasattr(torch._C, "_host_emptyCache"):
        torch._C._host_emptyCache()
    trim_host_heap()


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info
