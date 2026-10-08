#!/usr/bin/env python3
"""Compare sequential and concurrent GEMM + convolution execution on an Intel XPU.

The concurrent case submits GEMM and convolution work to independent XPU streams.
For sufficiently balanced workloads, the device can overlap their execution and its
total time should be lower than the one-stream (sequential) time.

Example:
    python scripts/debug/xpu_parallel_gemm_conv.py
    python scripts/debug/xpu_parallel_gemm_conv.py --iterations 40 --repeats 5
    python scripts/debug/xpu_parallel_gemm_conv.py --dtype float32
    python scripts/debug/xpu_parallel_gemm_conv.py --compile

This is a throughput demonstration, not a guarantee: a GEMM that already consumes
all available XPU execution resources, or very small workloads, will not overlap
enough to produce a speedup. Tune the shape options for the target device.
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time
from collections.abc import Callable

import torch
import torch.nn.functional as functional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20, help="operations per timed trial")
    parser.add_argument("--repeats", type=int, default=5, help="number of timed trials")
    parser.add_argument("--warmup", type=int, default=5, help="untimed trials before measurement")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="run each operation through torch.compile before submitting it to its stream",
    )
    parser.add_argument("--gemm-size", type=int, default=2048, help="square GEMM dimension")
    parser.add_argument("--conv-batch", type=int, default=16)
    parser.add_argument("--conv-channels", type=int, default=64)
    parser.add_argument("--conv-size", type=int, default=128, help="square convolution spatial size")
    args = parser.parse_args()
    if min(args.iterations, args.repeats, args.warmup, args.gemm_size, args.conv_batch,
           args.conv_channels, args.conv_size) < 1:
        parser.error("all numeric options must be positive")
    return args


def median_runtime_ms(
    submit: Callable[[], tuple[torch.Tensor, torch.Tensor]], repeats: int
) -> tuple[float, tuple[torch.Tensor, torch.Tensor]]:
    """Time submission plus completion; XPU work is asynchronous from the host."""

    samples_ms: list[float] = []
    result: tuple[torch.Tensor, torch.Tensor] | None = None
    for _ in range(repeats):
        torch.xpu.synchronize()
        start = time.perf_counter()
        result = submit()
        torch.xpu.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1_000)
    assert result is not None
    return statistics.median(samples_ms), result


def main() -> None:
    args = parse_args()
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise SystemExit("No PyTorch XPU device is available.")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = torch.device("xpu:0")
    torch.xpu.set_device(device)
    torch.manual_seed(0)

    # Create the data once. Allocation and initialization are deliberately outside
    # the benchmark: only GEMM and convolution execution is measured.
    gemm_lhs = torch.randn((args.gemm_size, args.gemm_size), device=device, dtype=dtype)
    gemm_rhs = torch.randn((args.gemm_size, args.gemm_size), device=device, dtype=dtype)
    conv_input = torch.randn(
        (args.conv_batch, args.conv_channels, args.conv_size, args.conv_size),
        device=device,
        dtype=dtype,
    )
    conv_weight = torch.randn(
        (args.conv_channels, args.conv_channels, 3, 3), device=device, dtype=dtype
    )
    torch.xpu.synchronize()

    def gemm() -> torch.Tensor:
        return gemm_lhs @ gemm_rhs

    def conv() -> torch.Tensor:
        return functional.conv2d(conv_input, conv_weight, padding=1)

    if args.compile:
        # Compile each independent operation separately so their stream placement
        # remains explicit. Compilation and kernel-cache population are warmup work.
        gemm = torch.compile(gemm, fullgraph=True)
        conv = torch.compile(conv, fullgraph=True)

    def sequential() -> tuple[torch.Tensor, torch.Tensor]:
        gemm_output: torch.Tensor | None = None
        conv_output: torch.Tensor | None = None
        for _ in range(args.iterations):
            # Both launches use the default stream, so the convolution waits for GEMM.
            gemm_output = gemm()
            conv_output = conv()
        assert gemm_output is not None and conv_output is not None
        return gemm_output, conv_output

    gemm_stream = torch.xpu.Stream(device=device)
    conv_stream = torch.xpu.Stream(device=device)

    def concurrent() -> tuple[torch.Tensor, torch.Tensor]:
        gemm_output: torch.Tensor | None = None
        conv_output: torch.Tensor | None = None
        for _ in range(args.iterations):
            # Independent streams remove the artificial ordering between operations.
            with torch.xpu.stream(gemm_stream):
                gemm_output = gemm()
            with torch.xpu.stream(conv_stream):
                conv_output = conv()
        assert gemm_output is not None and conv_output is not None
        return gemm_output, conv_output

    def threaded_concurrent() -> tuple[torch.Tensor, torch.Tensor]:
        """Submit each stream from its own host thread.

        This is intentionally compared with ``concurrent`` above: host threads do
        not themselves make GPU work concurrent. They can only change CPU-side
        launch behavior; stream independence is what permits device overlap.
        """

        outputs: list[torch.Tensor | None] = [None, None]
        errors: list[BaseException] = []
        ready = threading.Barrier(3)
        launch = threading.Event()

        def submit(
            output_index: int, stream: torch.xpu.Stream, operation: Callable[[], torch.Tensor]
        ) -> None:
            try:
                with torch.xpu.stream(stream):
                    ready.wait()
                    launch.wait()
                    output: torch.Tensor | None = None
                    for _ in range(args.iterations):
                        output = operation()
                    outputs[output_index] = output
            except BaseException as error:
                errors.append(error)
                ready.abort()

        workers = (
            threading.Thread(target=submit, args=(0, gemm_stream, gemm)),
            threading.Thread(target=submit, args=(1, conv_stream, conv)),
        )
        for worker in workers:
            worker.start()
        try:
            ready.wait()
        except threading.BrokenBarrierError:
            for worker in workers:
                worker.join()
            raise RuntimeError("a worker failed before XPU submission") from errors[0]
        launch.set()
        for worker in workers:
            worker.join()
        if errors:
            raise RuntimeError("a worker failed during XPU submission") from errors[0]
        gemm_output, conv_output = outputs
        assert gemm_output is not None and conv_output is not None
        return gemm_output, conv_output

    # Warm up kernel selection, caching, and allocator state in both configurations.
    for _ in range(args.warmup):
        sequential()
        concurrent()
        threaded_concurrent()
    torch.xpu.synchronize()

    sequential_ms, sequential_outputs = median_runtime_ms(sequential, args.repeats)
    concurrent_ms, concurrent_outputs = median_runtime_ms(concurrent, args.repeats)
    threaded_ms, threaded_outputs = median_runtime_ms(threaded_concurrent, args.repeats)

    # Force both result paths to be consumed after timing. This also catches failures
    # without adding an unintended synchronization inside the measured interval.
    for output in (*sequential_outputs, *concurrent_outputs, *threaded_outputs):
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("an operation produced non-finite output")
    if args.compile:
        # This gate is deliberately outside the timing interval. torch.compile is
        # allowed to choose different kernels, but must preserve the operation.
        torch.testing.assert_close(threaded_outputs[0], gemm_lhs @ gemm_rhs)
        torch.testing.assert_close(
            threaded_outputs[1],
            functional.conv2d(conv_input, conv_weight, padding=1),
        )

    stream_speedup = sequential_ms / concurrent_ms
    threaded_speedup = sequential_ms / threaded_ms
    print(f"XPU: {torch.xpu.get_device_name(device)}")
    print(
        f"workload: {args.iterations}x GEMM [{args.gemm_size}, {args.gemm_size}] "
        f"+ Conv2d [{args.conv_batch}, {args.conv_channels}, {args.conv_size}, {args.conv_size}]"
    )
    compile_mode = "torch.compile" if args.compile else "eager"
    print(f"dtype: {args.dtype}; mode: {compile_mode}; median of {args.repeats} trials")
    print(f"sequential (default stream): {sequential_ms:.2f} ms")
    print(f"concurrent (two streams):    {concurrent_ms:.2f} ms")
    print(f"two streams + CPU threads:   {threaded_ms:.2f} ms")
    print(f"two-stream speedup:          {stream_speedup:.2f}x")
    print(f"threaded speedup:            {threaded_speedup:.2f}x")
    if args.compile:
        print("compiled output gate:        PASS (matches eager)")
    if max(stream_speedup, threaded_speedup) <= 1.0:
        print("No measured overlap; reduce one workload or increase the other and retry.")


if __name__ == "__main__":
    main()
