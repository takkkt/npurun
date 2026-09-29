"""Bit-level NumPy emulation of kernel/tgemv.cc over a packed npurun stream.

Used to check host packing independently of the NPU.
"""

from __future__ import annotations

import numpy as np

from .pack import FMT, HDR, KIND_TILE, KIND_XLOAD, NROWS, NT, SG_BYTES, SGK, Geometry


def tile_mv(tile: np.ndarray, x: np.ndarray, g: Geometry) -> np.ndarray:
    """tile: tile_bytes uint8; x: one activation chunk (xchunk_bytes uint8) -> TM floats."""
    TK, TM = g.TK, g.TM
    ng = TK // 128
    xq = x[:TK].view(np.int8).astype(np.int64)
    sx = x[TK : TK + 4 * ng].view(np.float32)
    sumx = x[TK + 4 * ng : TK + 8 * ng].view(np.int32).astype(np.int64)
    y = np.zeros(TM, np.float64)
    blocks = tile.reshape(TM // 8, TK // SGK, SG_BYTES)
    for rb in range(TM // 8):
        for s in range(TK // SGK):
            blk = blocks[rb, s]
            if FMT == 3:
                qs = blk[: 26 * 64].reshape(26, 8, 8).astype(np.int64)  # v, kk, r
                sc = blk[26 * 64 :].view(np.float16).astype(np.float32).reshape(8, 8)  # n, r
                codes = np.stack([(qs // 3**t) % 3 for t in range(5)], axis=1)  # v, t, kk, r
                codes = codes.reshape(130, 8, 8)[:128]  # chunk c, kk, r
            else:
                qs = blk[: 32 * 64].reshape(32, 8, 8)  # q, kk, r
                sc = blk[32 * 64 :].view(np.float32).reshape(8, 8)  # n, r
                codes = np.stack([(qs >> (2 * t)) & 3 for t in range(4)], axis=1)  # q, t, kk, r
                codes = codes.reshape(128, 8, 8).astype(np.int64)  # chunk c, kk, r
            xc = xq[s * 1024 : (s + 1) * 1024].reshape(128, 8)  # chunk c, kk
            dots = np.einsum("ckr,ck->cr", codes, xc).reshape(8, 16, 8).sum(axis=1)  # n, r
            dots -= sumx[s * 8 : (s + 1) * 8, None]
            y[rb * 8 : rb * 8 + 8] += (dots * sc * sx[s * 8 : (s + 1) * 8, None]).sum(axis=0)
    return y.astype(np.float32)


def run_stream(w: np.ndarray, xbuf: np.ndarray, g: Geometry) -> np.ndarray:
    """Emulate the whole NPU call; returns the raw joined Y buffer (y_floats)."""
    ob = g.obj_bytes
    xres = np.zeros(g.NX * g.tile_bytes + 64, np.uint8)
    xs = xbuf.reshape(g.NX, 2, ob)
    for i in range(g.NX):
        hdr = xs[i, 0, :HDR].view(np.int32)
        assert hdr[0] == KIND_XLOAD
        xres[hdr[2] : hdr[2] + hdr[3]] = xs[i, 0, HDR : HDR + hdr[3]]
    Y = np.zeros((g.ncols, g.y_objs, NROWS, NT, g.TM), np.float32)
    ws = w.reshape(g.ncols, 2, g.T * g.KC, 2, ob)
    for c in range(g.ncols):
        for h in range(2):
            for j in range(g.T * g.KC):
                for rr in range(2):
                    obj = ws[c, h, j, rr]
                    hdr = obj[:HDR].view(np.int32)
                    assert hdr[0] == KIND_TILE
                    kc = hdr[1]
                    x = xres[kc * g.xchunk_bytes : (kc + 1) * g.xchunk_bytes]
                    part = tile_mv(obj[HDR:], x, g)
                    if kc > 0:  # running sum over the K chunks of a row tile
                        part = part + Y[c, g.NX + j - 1, 2 * h + rr, 0]
                    Y[c, g.NX + j, 2 * h + rr, 0] = part
    return Y.reshape(-1)
