#!/usr/bin/env python3
"""Free device memory through the CUDA driver API, sampled — the check before a production-sized
run on a GPU that may be shared (I-G16-027): a swinging free-memory line means another tenant, and
the arm's ≈ 8 GB working set on that tenant's peak fails either side.

    gpu-free.py [samples] [interval-seconds]      (defaults: 5 samples, 2 s)

Needs only libcuda.so.1 on the loader path (the driver's user-space library, not a toolkit). Exit 0
when the minimum free memory across the samples is at least GPU_FREE_MIN_GIB (default 10), else 1.
"""
import ctypes
import os
import sys
import time


def sample():
    lib = ctypes.CDLL("libcuda.so.1")
    if lib.cuInit(0) != 0:
        raise SystemExit("cuInit failed: no usable device")
    dev = ctypes.c_int()
    lib.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p()
    lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    lib.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
    lib.cuCtxDestroy_v2(ctx)
    return free.value / 2**30, total.value / 2**30


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    need = float(os.environ.get("GPU_FREE_MIN_GIB", "10"))
    frees = []
    for i in range(n):
        free, total = sample()
        frees.append(free)
        print(f"free {free:6.2f} GiB of {total:.2f} GiB")
        if i + 1 < n:
            time.sleep(interval)
    lo, hi = min(frees), max(frees)
    verdict = "quiet" if hi - lo < 1.0 else "SHARED (another tenant is allocating)"
    print(f"min {lo:.2f} GiB, max {hi:.2f} GiB, swing {hi - lo:.2f} GiB: {verdict}")
    return 0 if lo >= need else 1


if __name__ == "__main__":
    sys.exit(main())
