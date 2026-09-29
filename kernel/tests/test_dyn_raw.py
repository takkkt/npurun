"""Run a dynamic-builder instruction stream (from txndump) through the lean runtime."""

import faulthandler
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from npurun.build import build_tgemv  # noqa: E402
from npurun.pack import NROWS, Geometry, XPacker, pack_weights, pad_rows, reference, unpack_y  # noqa: E402
from npurun.runtime import FROM_DEV, TO_DEV, NpuProgram  # noqa: E402


def main():
    faulthandler.dump_traceback_later(60, exit=True)
    insts_path = Path(sys.argv[1])
    M, K = 1024, 5120
    g = Geometry(pad_rows(M), K)
    xclbin, _ = build_tgemv(Geometry(pad_rows(5120), 5120))  # any shape's xclbin
    if len(sys.argv) > 2:
        xclbin = Path(sys.argv[2])
    prog = NpuProgram(xclbin)
    insts = prog.insts(insts_path)
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 3, size=(M, K), dtype=np.uint8)
    scales = (rng.random((M, K // 128)) * 0.02 + 0.005).astype(np.float16)
    x = rng.standard_normal(K).astype(np.float32)
    wp = pack_weights(codes, scales, g)
    W = prog.buffer(wp.nbytes, 0)
    W.write(wp, 0)
    W.sync(TO_DEV)
    X = prog.buffer(g.x_bytes, 1)
    X.write(XPacker(g).pack(x), 0)
    X.sync(TO_DEV)
    Y = prog.buffer(g.y_floats * 4, 2)
    print("running", flush=True)
    prog.run(insts, W, X, Y)
    Y.sync(FROM_DEV)
    y = unpack_y(np.frombuffer(Y.read(g.y_floats * 4, 0), np.float32), g, M)
    ref = reference(codes, scales, x)
    print("cosine", float(np.dot(y, ref) / (np.linalg.norm(y) * np.linalg.norm(ref))), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
