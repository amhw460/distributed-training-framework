"""Single-host CPU stress test of all five model-parallel axes together.

Fork-based CPU workers share read-only Python/PyTorch pages, avoiding the
large duplicated interpreter footprint of 32 independent torchrun processes.
The workers still use real torch.distributed Gloo process groups and collectives.
Linux-only test runner, not the general training entry point.
"""
import os
import socket
import sys

import torch.multiprocessing as mp


WORLD_SIZE = 32


def _worker(rank, port):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(WORLD_SIZE),
        OMP_NUM_THREADS="1",
    )
    sys.argv = [
        "scripts.pretrain_hybrid",
        "--synthetic",
        "--fsdp", "2",
        "--tp", "2",
        "--ep", "2",
        "--cp", "2",
        "--pp", "2",
        "--steps", "2",
        "--local-batch-size", "2",
        "--microbatches", "2",
        "--num-layers", "2",
        "--experts", "2",
        "--top-k", "1",
        "--num-heads", "2",
        "--num-kv-heads", "2",
        "--dim", "8",
        "--moe-inter-dim", "8",
        "--head-dim", "4",
        "--vocab-size", "17",
        "--seq-len", "4",
    ]
    from scripts.pretrain_hybrid import main
    main()


def main():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    print("Starting 32 real distributed CPU ranks for FSDP+TP+EP+CP+PP", flush=True)
    mp.start_processes(_worker, args=(port,), nprocs=WORLD_SIZE,
                       join=True, start_method="fork")
    print("PASS: five-way FSDP+TP+EP+CP+PP training finished on 32 CPU ranks", flush=True)


if __name__ == "__main__":
    main()
