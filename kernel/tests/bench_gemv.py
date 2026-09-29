"""Benchmark the ternary GEMV through the lean XRT runtime (weights resident).

usage: python tests/bench_gemv.py -M 5120 -K 5120 [--iters 50] [--nop] [--xclbin-from M,K]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from npurun.build import build_tgemv  # noqa: E402
from npurun.pack import Geometry, pack_weights, pad_rows, reference  # noqa: E402
from npurun.runtime import NpuProgram, TernaryGemv  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-M", type=int, default=5120)
    p.add_argument("-K", type=int, default=5120)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--force", action="store_true")
    p.add_argument("--nop", action="store_true", help="data movement only (no compute)")
    p.add_argument("--xclbin-from", default=None, help="M,K of another build whose xclbin to load")
    p.add_argument("--TM", type=int, default=32)
    a = p.parse_args()

    g = Geometry(pad_rows(a.M, a.TM), a.K, TM=a.TM)
    t0 = time.perf_counter()
    xclbin, insts = build_tgemv(g, force=a.force, nop=a.nop)
    if a.xclbin_from:
        m2, k2 = map(int, a.xclbin_from.split(","))
        xclbin, _ = build_tgemv(Geometry(pad_rows(m2), k2), nop=a.nop)
    print(f"build: {time.perf_counter() - t0:.1f}s  (xclbin {xclbin.parent.name})")

    rng = np.random.default_rng(0)
    codes = rng.integers(0, 3, size=(a.M, a.K), dtype=np.uint8)
    scales = (rng.random((a.M, a.K // 128)) * 0.02 + 0.005).astype(np.float16)
    x = rng.standard_normal(a.K).astype(np.float32)

    prog = NpuProgram(xclbin)
    op = TernaryGemv(prog, insts, pack_weights(codes, scales, g), g, a.M)
    y = op(x)
    ref = reference(codes, scales, x)
    cos = float(np.dot(y, ref) / (np.linalg.norm(y) * np.linalg.norm(ref) + 1e-30))
    print(f"cosine {cos:.6f}")

    ts = []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        op(x)
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    best, med = ts.min(), np.median(ts)
    print(
        f"{a.M}x{a.K}: median {med * 1e3:.3f} ms  best {best * 1e3:.3f} ms  "
        f"-> {g.w_bytes / med / 1e9:.2f} GB/s (median), {g.w_bytes / best / 1e9:.2f} GB/s (best)"
    )
    xpk = op.xp.pack(x)
    op.x.write(xpk, 0)
    rs = []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        op.p.run(op.insts, op.w, op.x, op.y)
        rs.append(time.perf_counter() - t0)
    rs = np.array(rs)
    t0 = time.perf_counter()
    for _ in range(20):
        op.xp.pack(x)
    tp = (time.perf_counter() - t0) / 20
    print(f"  npu run only: median {np.median(rs) * 1e3:.3f} ms best {rs.min() * 1e3:.3f} ms -> {g.w_bytes / rs.min() / 1e9:.2f} GB/s best; host x-pack {tp * 1e6:.0f} us")
    print("PASS" if cos > 0.999 else "FAIL", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
