import gc
import logging
import subprocess

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def clear_memory(clear_host_memory: bool = False):
    torch.accelerator.synchronize()
    gc.collect()
    torch.accelerator.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    """Memory of the current device. ``total``, ``free`` and ``used`` come from
    ``get_memory_info`` and cover every process on the GPU; ``allocated``,
    ``reserved``, ``inactive_split`` (reserved bytes the allocator holds in
    fragments it cannot hand out) and ``segments`` are this process's caching
    allocator."""
    device = torch.accelerator.current_device_index()
    free, total = torch.accelerator.get_memory_info(device)
    stats = torch.accelerator.memory_stats(device)
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.accelerator.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.accelerator.memory_reserved(device)),
        "inactive_split_GB": _byte_to_gb(stats.get("inactive_split_bytes.all.current", 0)),
        "segments": stats.get("segment.all.current", 0),
        "alloc_retries": stats.get("num_alloc_retries", 0),
    }


NVIDIA_SMI_TIMEOUT_S = 15


def gpu_processes(device: int | None = None, query=None) -> list[dict]:
    """Every process with memory on the current device, from ``nvidia-smi``
    (``pid``, ``used_MiB``, ``name``): the only way to tell this process's
    share of ``used_GB`` from a colocated engine's. ``query`` is the
    ``nvidia-smi`` runner, replaced in tests. Returns ``[]`` with a logged
    warning when the tool is missing, times out or answers in another shape."""
    if not torch.cuda.is_available():
        return []
    if device is None:
        device = torch.cuda.current_device()
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    args = [
        "nvidia-smi",
        f"--id=GPU-{uuid}",
        "--query-compute-apps=pid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    ]
    try:
        text = (query or _run_nvidia_smi)(args)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"gpu_processes: {e}")
        return []
    return parse_compute_apps(text)


def _run_nvidia_smi(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=NVIDIA_SMI_TIMEOUT_S).stdout


def parse_compute_apps(text: str) -> list[dict]:
    """``pid, used_memory, process_name`` csv rows (nounits: MiB) to records;
    a row that does not parse is skipped."""
    rows = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            continue
        try:
            rows.append({"pid": int(parts[0]), "used_MiB": int(float(parts[1])), "name": parts[2]})
        except ValueError:
            continue
    return rows


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False, processes: bool = False):
    """Log the current device's memory; with ``processes`` also every process
    on it (one ``nvidia-smi`` call, for the moments a colocated engine shares
    the GPU: after the trainer releases its cache, ``used_GB`` is the sum of
    both and only the per-process list says which one grew)."""
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    if processes:
        memory_info["processes"] = gpu_processes()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info
