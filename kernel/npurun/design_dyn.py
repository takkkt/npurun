"""Parameterized (DispatchTime) runtime sequence for the shape-agnostic tgemv core.

The core program and fifo topology are exactly those of design.tgemv; only the
host DMA sequence differs: every size and offset is a runtime scalar, so one
compiled design (and the C++ transaction builder aiecc emits for it) serves
any weight matrix, and up to NMAT matrices that share one input vector can be
streamed in a single NPU launch ("fused" call).

Runtime scalars (int32):
  nx          XLOAD pairs at the head of every channel stream
  b<j>, h<j>  matrix j: byte offset of its packed stream in W, and weight pairs
              per channel (T*KC); channel (c, h) reads
              W[b<j> + (2c+h) * h<j> * 2*obj_bytes : ... + h<j> * 2*obj_bytes]
Outputs: column c writes (nx + sum_j h<j>) joined objects of 4*TM*NT floats at
Y[c * (nx + sum_j h<j>) * 4*TM*NT].

nmat == 0 is a K-sliced single matrix (batch launches over wide K): b0 base,
h0 T (row tiles), b1 KC (chunks per row), h1 kc0, b2 nkc; every channel streams
chunks [kc0, kc0+nkc) of each row tile, and outputs nx + T*nkc objects.
"""

from __future__ import annotations

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.extras.dialects import arith
from aie.helpers.util import np_dtype_to_mlir_type
from aie.iron import (
    CompileTime,
    DispatchTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
)
from aie.utils.config import cxx_header_path

from .design import _split_bytes
from .pack import FMT, HDR, KMAX_DEFAULT, NROWS, NT, SG_BYTES, SGK

KERNEL_SRC = Path(__file__).resolve().parent.parent / "tgemv.cc"

# Largest buffers the host may pass (only used for memref typing).
W_MAX = 2**31 - 64
X_MAX = 1 << 20
Y_MAX = 1 << 26


@iron.jit
def tgemv_dyn(
    W: In,
    X: In,
    Y: Out,
    *,
    nmat: CompileTime[int] = 1,
    TM: CompileTime[int] = 32,
    TK: CompileTime[int] = 1024,
    ncols: CompileTime[int] = 8,
    KMAX: CompileTime[int] = KMAX_DEFAULT,
    wbytes: CompileTime[int] = W_MAX,
    xbytes: CompileTime[int] = X_MAX,
    yfloats: CompileTime[int] = Y_MAX,
    nx: DispatchTime[np.int32] = 1,
    b0: DispatchTime[np.int32] = 0,
    h0: DispatchTime[np.int32] = 1,
    b1: DispatchTime[np.int32] = 0,
    h1: DispatchTime[np.int32] = 1,
    b2: DispatchTime[np.int32] = 0,
    h2: DispatchTime[np.int32] = 1,
):
    tile_bytes = (TM // 8) * (TK // SGK) * SG_BYTES
    ob = HDR + tile_bytes

    obj_ty = np.ndarray[(ob,), np.dtype[np.uint8]]
    pair_ty = np.ndarray[(2 * ob,), np.dtype[np.uint8]]
    y_tile_ty = np.ndarray[(TM * NT,), np.dtype[np.float32]]
    y_obj_ty = np.ndarray[(NROWS * TM * NT,), np.dtype[np.float32]]

    step = ExternalFunction(
        "tg_step",
        source_file=str(KERNEL_SRC),
        arg_types=[obj_ty, y_tile_ty],
        include_dirs=[cxx_header_path()],
        compile_flags=[f"-DTG_M={TM}", f"-DTG_K={TK}", f"-DTG_KMAX={KMAX}", f"-DTG_FMT={FMT}"],
    )

    def core_fn(of_w, of_y, step):
        o = of_w.acquire(1)
        y = of_y.acquire(1)
        step(o, y)
        of_w.release(1)
        of_y.release(1)

    w_l3, y_l3, workers = [], [], []
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

    W_ty = np.ndarray[(wbytes,), np.dtype[np.uint8]]
    X_ty = np.ndarray[(xbytes,), np.dtype[np.uint8]]
    Y_ty = np.ndarray[(yfloats,), np.dtype[np.float32]]
    yrun = NROWS * TM * NT
    pair = 2 * ob
    n1, run = _split_bytes(ob)  # one object = n1 runs of `run` bytes (<= 1023 words each)

    def sequence(Wb, Xb, Yb, nx_, b0_, h0_, b1_, h1_, b2_, h2_, w_hs, y_hs):
        i32 = np_dtype_to_mlir_type(np.int32)
        i64 = np_dtype_to_mlir_type(np.int64)
        mats = [(b0_, h0_), (b1_, h1_), (b2_, h2_)][:nmat]
        c_pair = arith.constant(pair, i32)
        c_pair64 = arith.constant(pair, i64)
        c2 = arith.constant(2, i64)
        nx64 = arith.extsi(i64, nx_)
        ytot = nx_
        for _, hj in mats:
            ytot = ytot + hj
        if nmat == 0:  # K slice: b0 base, h0 T, b1 KC, h1 kc0, b2 nkc
            ytot = nx_ + h0_ * b2_
        ytot64 = arith.extsi(i64, ytot)
        tg = TaskGroup()
        for c in range(ncols):
            for h in range(2):
                hs = w_hs[2 * c + h]
                ch = arith.constant(2 * c + h, i32)
                hs.fill(
                    Xb,
                    offset=arith.constant(0, i32),
                    sizes=[1, 1, 1, nx64 * c_pair64],  # contiguous: lowers to a linear BD
                    strides=[0, 0, 0, 1],
                    transfer_len=nx_ * c_pair,
                    group=tg,
                )
                if nmat == 0:
                    T_, KC_, kc0_, nkc_ = h0_, b1_, h1_, b2_
                    hs.fill(
                        Wb,
                        offset=b0_ + ch * T_ * KC_ * c_pair + kc0_ * c_pair,
                        sizes=[1, arith.extsi(i64, T_), arith.extsi(i64, nkc_) * arith.constant(2 * n1, i64), run],
                        strides=[0, arith.extsi(i64, KC_) * c_pair64, run, 1],
                        transfer_len=T_ * nkc_ * c_pair,
                        group=tg,
                    )
                for bj, hj in mats:
                    hj64 = arith.extsi(i64, hj)
                    hs.fill(
                        Wb,
                        offset=bj + ch * hj * c_pair,
                        sizes=[1, 1, 1, hj64 * c_pair64],
                        strides=[0, 0, 0, 1],
                        transfer_len=hj * c_pair,
                        group=tg,
                    )
        c_yrun = arith.constant(yrun, i32)
        c_yrun64 = arith.constant(yrun, i64)
        for c in range(ncols):
            y_hs[c].drain(
                Yb,
                offset=arith.constant(c, i32) * ytot * c_yrun,
                sizes=[1, 1, 1, ytot64 * c_yrun64],
                strides=[0, 0, 0, 1],
                transfer_len=ytot * c_yrun,
                wait=True,
                group=tg,
            )
        tg.finish()

    rt = Runtime(
        sequence,
        [W_ty, X_ty, Y_ty, nx, b0, h0, b1, h1, b2, h2, [f.prod() for f in w_l3], [f.cons() for f in y_l3]],
    )
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()
