"""Fast distributed-launch probe for retriever shell scripts."""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    timeout_seconds = int(os.environ.get("DISTRIBUTED_PROBE_TIMEOUT_SECONDS", "60"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.empty(1, device=f"cuda:{local_rank}")

    dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_seconds))
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if backend == "nccl":
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()
        print(
            f"distributed_probe_ok rank={rank} world_size={world_size} "
            f"local_rank={local_rank} backend={backend}",
            flush=True,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
