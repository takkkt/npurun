"""IRON design: shape-agnostic ternary GEMV across 8 columns x 4 AIE cores (XDNA2).

Per AIE column, both shim MM2S channels stream header-tagged objects; each
channel feeds a memtile split to two cores. Every core runs the same endless
loop (acquire object, tg_step, emit TM floats), so the compiled core program
and fifo topology are identical for every matrix shape: only the runtime DMA
sequence (insts) is shape specific, and one loaded xclbin serves a whole model.
"""

from __future__ import annotations

import os
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.utils.config import cxx_header_path

from .pack import FMT, KMAX_DEFAULT, NROWS, NT, Geometry

KERNEL_SRC = Path(__file__).resolve().parent.parent / "tgemv.cc"
MAX_DIM = 1023  # shim BD wrap limit


def _split_bytes(n: int) -> tuple[int, int]:
    """n bytes -> (count, run) with run*count == n, run a multiple of 4 and <= 1023 words."""
    words = n // 4
    for run in range(min(words, MAX_DIM), 0, -1):
        if words % run == 0 and words // run <= MAX_DIM:
            return words // run, run * 4
    raise ValueError(f"cannot express {n} bytes in two DMA dims")


def _chunks(total: int, step: int = MAX_DIM):
    off = 0
    while off < total:
        n = min(step, total - off)
        yield off, n
        off += n


@iron.jit
def tgemv(
    W: In,
    X: In,
    Y: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    TM: CompileTime[int],
    TK: CompileTime[int],
    ncols: CompileTime[int],
    KMAX: CompileTime[int] = KMAX_DEFAULT,
    nop: CompileTime[bool] = False,
):
    g = Geometry(M, K, TM, TK, ncols)
    ob = g.obj_bytes

    obj_ty = np.ndarray[(ob,), np.dtype[np.uint8]]
    pair_ty = np.ndarray[(2 * ob,), np.dtype[np.uint8]]
    y_tile_ty = np.ndarray[(TM * NT,), np.dtype[np.float32]]
    y_obj_ty = np.ndarray[(NROWS * TM * NT,), np.dtype[np.float32]]

    flags = [f"-DTG_M={TM}", f"-DTG_K={TK}", f"-DTG_KMAX={KMAX}", f"-DTG_FMT={FMT}"]
    if nop:
        flags.append("-DTG_NOP")
    if os.environ.get("NPURUN_DEBUG"):
        flags.append("-DTG_DEBUG")
    step = ExternalFunction(
        "tg_step",
        source_file=str(KERNEL_SRC),
        arg_types=[obj_ty, y_tile_ty],
        include_dirs=[cxx_header_path()],
        compile_flags=flags,
    )

    def core_fn(of_w, of_y, step):
        o = of_w.acquire(1)
        y = of_y.acquire(1)
        step(o, y)
        of_w.release(1)
        of_y.release(1)

    w_l3 = []
    y_l3 = []
    workers = []
    for c in range(ncols):
        yf = ObjectFifo(y_obj_ty, name=f"y_l3_{c}", depth=2)
        y_l1 = yf.prod().join(
            [r * TM * NT for r in range(NROWS)],
            obj_types=[y_tile_ty] * NROWS,
            names=[f"y_l1_{c}_{r}" for r in range(NROWS)],
            depths=[2] * NROWS,
        )
        y_l3.append(yf)
        for h in range(2):
            wf = ObjectFifo(pair_ty, name=f"w_l3_{c}_{h}", depth=2)
            w_l1 = wf.cons().split(
                [0, ob],
                obj_types=[obj_ty] * 2,
                names=[f"w_l1_{c}_{2 * h + rr}" for rr in range(2)],
            )
            w_l3.append(wf)
            for rr in range(2):
                workers.append(
                    Worker(core_fn, [w_l1[rr].cons(), y_l1[2 * h + rr].prod(), step], stack_size=0x2000)
                )

    W_ty = np.ndarray[(g.w_bytes,), np.dtype[np.uint8]]
    X_ty = np.ndarray[(g.x_bytes,), np.dtype[np.uint8]]
    Y_ty = np.ndarray[(g.y_floats,), np.dtype[np.float32]]

    n1, run = _split_bytes(ob)
    yrun = NROWS * TM * NT  # floats per joined output object

    def sequence(Wb, Xb, Yb, w_hs, y_hs):
        for c in range(ncols):
            for h in range(2):
                hs = w_hs[2 * c + h]
                hs.fill(Xb, offset=0, sizes=[1, g.NX * 2, n1, run], strides=[0, ob, run, 1])
                base = (2 * c + h) * g.half_bytes
                for off, n in _chunks(g.half_objs * 2):
                    hs.fill(Wb, offset=base + off * ob, sizes=[1, n, n1, run], strides=[0, ob, run, 1])
        for c in range(ncols):
            base = c * g.y_objs * yrun
            parts = list(_chunks(g.y_objs))
            for i, (off, n) in enumerate(parts):
                y_hs[c].drain(
                    Yb,
                    offset=base + off * yrun,
                    sizes=[1, 1, n, yrun],
                    strides=[0, 0, yrun, 1],
                    wait=(i == len(parts) - 1),
                )

    rt = Runtime(sequence, [W_ty, X_ty, Y_ty, [f.prod() for f in w_l3], [f.cons() for f in y_l3]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()
