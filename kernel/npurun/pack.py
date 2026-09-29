"""Host-side packing between GGUF PQ2_0 weights and the npurun NPU stream format.

The NPU program is shape-agnostic (see kernel/tgemv.cc): it consumes stream
objects of HDR + tile_bytes bytes and emits TM floats per object.

Each AIE column c is fed by its two shim MM2S channels ("halves"); half h
serves core rows 2h and 2h+1 through a memtile split, so every stream object
on a half is a *pair* (one object per core). A half's stream is
    NX pairs of XLOAD objects (activations, same for both cores)
    for t in row tiles: for kc in K chunks: pair of TILE objects
Rows served by column c, row tile t, core r (= 2h + rr):
    c*Mc + (t*NROWS + r)*TM + [0, TM)          Mc = M / ncols
The joined output of a column is one 4*TM-float object per stream step;
the first NX are dummies, the rest are per-chunk partials summed on the host.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

QK = 128  # PQ2_0 group size
SGK = 1024  # super-group: 8 groups share one 8x8x8 accumulation
NROWS = 4  # AIE compute rows per column on XDNA2
NT = 4  # tokens per batch tile (kernel batch path); every output object holds NT * TM floats per core
FMT = 3  # 3: base-3 codes + fp16 scales (1.75 b/w); 2: 2-bit codes + fp32 scales (2.25 b/w)
SG_BYTES = 26 * 64 + 64 * 2 if FMT == 3 else 32 * 64 + 64 * 4  # per 8 rows x 1024 columns
HDR = 64
KIND_XLOAD, KIND_TILE = 1, 2
TM_DEFAULT, TK_DEFAULT, KMAX_DEFAULT, NCOLS = 32, 1024, 17408, 8


def row_multiple(TM: int = TM_DEFAULT, ncols: int = NCOLS) -> int:
    return ncols * NROWS * TM


def pad_rows(M: int, TM: int = TM_DEFAULT, ncols: int = NCOLS) -> int:
    m = row_multiple(TM, ncols)
    return -(-M // m) * m


@dataclass(frozen=True)
class Geometry:
    M: int  # padded rows
    K: int
    TM: int = TM_DEFAULT
    TK: int = TK_DEFAULT
    ncols: int = NCOLS

    def __post_init__(self):
        if self.TK % SGK:
            raise ValueError(f"TK={self.TK} must be a multiple of {SGK}")
        if self.TM % 8:
            raise ValueError(f"TM={self.TM} must be a multiple of 8")
        if self.K % self.TK:
            raise ValueError(f"K={self.K} not a multiple of TK={self.TK}")
        if self.M % row_multiple(self.TM, self.ncols):
            raise ValueError(f"M={self.M} not a multiple of {row_multiple(self.TM, self.ncols)}")

    @property
    def Mc(self) -> int:
        return self.M // self.ncols

    @property
    def T(self) -> int:  # row tiles per core
        return self.Mc // (NROWS * self.TM)

    @property
    def KC(self) -> int:  # K chunks
        return self.K // self.TK

    @property
    def tile_bytes(self) -> int:
        return (self.TM // 8) * (self.TK // SGK) * SG_BYTES

    @property
    def obj_bytes(self) -> int:
        return HDR + self.tile_bytes

    @property
    def xchunk_bytes(self) -> int:
        return self.TK + (self.TK // QK) * 8

    @property
    def xraw_bytes(self) -> int:
        return self.KC * self.xchunk_bytes

    @property
    def NX(self) -> int:  # XLOAD objects per call
        return -(-self.xraw_bytes // self.tile_bytes)

    @property
    def half_objs(self) -> int:  # weight pairs on one channel
        return self.T * self.KC

    @property
    def half_bytes(self) -> int:
        return self.half_objs * 2 * self.obj_bytes

    @property
    def w_bytes(self) -> int:
        return self.ncols * 2 * self.half_bytes

    @property
    def x_bytes(self) -> int:  # host X buffer: NX pairs
        return self.NX * 2 * self.obj_bytes

    @property
    def y_objs(self) -> int:  # joined output objects per column
        return self.NX + self.T * self.KC

    @property
    def y_floats(self) -> int:
        return self.ncols * self.y_objs * NROWS * NT * self.TM


def pq2_to_codes(qs: np.ndarray) -> np.ndarray:
    """PQ2_0 qs bytes (R, K/4) -> codes (R, K). GGUF: byte j/4, bits 2*(j%4)."""
    R, B = qs.shape
    out = np.empty((R, B, 4), np.uint8)
    for t in range(4):
        out[:, :, t] = (qs >> (2 * t)) & 3
    return out.reshape(R, B * 4)


def _blocks(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """(M,K) codes + (M,K/128) scales -> (M/8, K/1024, SG_BYTES) kernel blocks."""
    if FMT == 3:
        return _blocks3(codes, scales)
    M, K = codes.shape
    c = codes.reshape(M // 8, 8, K // SGK, 32, 4, 8)  # rb, r, sg, q, t, kk
    c = c.transpose(0, 2, 3, 5, 1, 4).astype(np.uint8)  # rb, sg, q, kk, r, t
    qs = c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)
    qs = qs.reshape(M // 8, K // SGK, 32 * 64)
    s = scales.astype(np.float32).reshape(M // 8, 8, K // SGK, 8)  # rb, r, sg, n
    s = np.ascontiguousarray(s.transpose(0, 2, 3, 1)).reshape(M // 8, K // SGK, 64)
    return np.concatenate([qs, s.view(np.uint8)], axis=-1)


def _blocks3(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Base-3 format: word v, byte kk*8+r = sum_t code(r, 8*(5v+t)+kk) * 3**t."""
    M, K = codes.shape
    c = codes.reshape(M // 8, 8, K // SGK, 128, 8).transpose(0, 2, 3, 4, 1)  # rb, sg, chunk, kk, r
    c = np.concatenate([c, np.zeros(c.shape[:2] + (2, 8, 8), c.dtype)], axis=2)  # 130 chunks
    c = c.reshape(M // 8, K // SGK, 26, 5, 64).astype(np.uint16)  # rb, sg, v, t, byte
    qs = (c[:, :, :, 0] + 3 * c[:, :, :, 1] + 9 * c[:, :, :, 2] + 27 * c[:, :, :, 3] + 81 * c[:, :, :, 4])
    qs = qs.astype(np.uint8).reshape(M // 8, K // SGK, 26 * 64)
    s = scales.astype(np.float16).reshape(M // 8, 8, K // SGK, 8)  # rb, r, sg, n
    s = np.ascontiguousarray(s.transpose(0, 2, 3, 1)).reshape(M // 8, K // SGK, 64)
    return np.concatenate([qs, s.view(np.uint8)], axis=-1)


def pack_weights(codes: np.ndarray, scales: np.ndarray, g: Geometry) -> np.ndarray:
    """codes (M0,K) uint8 0..3; scales (M0, K/128), M0 <= g.M -> flat uint8 NPU stream."""
    M0 = codes.shape[0]
    if M0 < g.M:  # pad with zero-scale rows
        codes = np.concatenate([codes, np.ones((g.M - M0, g.K), np.uint8)])
        scales = np.concatenate([scales, np.zeros((g.M - M0, g.K // QK), scales.dtype)])
    b = _blocks(codes, scales)
    b = b.reshape(g.ncols, g.T, 2, 2, g.TM // 8, g.KC, g.TK // SGK, SG_BYTES)
    b = b.transpose(0, 2, 1, 5, 3, 4, 6, 7)  # c, h, t, kc, rr, rb, sg, bytes
    b = b.reshape(g.ncols, 2, g.T, g.KC, 2, g.tile_bytes)
    out = np.zeros((g.ncols, 2, g.T, g.KC, 2, g.obj_bytes), np.uint8)
    hdr = out[..., :HDR].view(np.int32)  # (..., 16)
    hdr[..., 0] = KIND_TILE
    hdr[..., 1] = np.arange(g.KC, dtype=np.int32)[None, None, None, :, None]
    out[..., HDR:] = b
    out = out.reshape(-1)
    assert out.nbytes == g.w_bytes
    return out


def quantize_x_raw(x: np.ndarray, g: Geometry) -> np.ndarray:
    """float activations (K,) -> KC chunks of [xq int8 | sx f32 | sumx i32] as uint8."""
    xg = x.astype(np.float32).reshape(-1, QK)
    amax = np.abs(xg).max(axis=1)
    s = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(xg / s[:, None]), -127, 127).astype(np.int8)
    sums = q.astype(np.int32).sum(axis=1).astype(np.int32)
    ng = g.TK // QK
    out = np.concatenate(
        [
            q.reshape(g.KC, g.TK).view(np.uint8),
            s.reshape(g.KC, ng).view(np.uint8),
            sums.reshape(g.KC, ng).view(np.uint8),
        ],
        axis=1,
    )
    return np.ascontiguousarray(out).reshape(-1)


class XPacker:
    """Reusable host X buffer image; only the payload changes between calls."""

    def __init__(self, g: Geometry):
        self.g = g
        self.buf = np.zeros((g.NX, 2, g.obj_bytes), np.uint8)
        hdr = self.buf[..., :HDR].view(np.int32)
        hdr[..., 0] = KIND_XLOAD
        for i in range(g.NX):
            n = min(g.tile_bytes, g.xraw_bytes - i * g.tile_bytes)
            hdr[i, :, 2] = i * g.tile_bytes
            hdr[i, :, 3] = -(-n // 64) * 64

    def pack(self, x: np.ndarray) -> np.ndarray:
        g = self.g
        raw = quantize_x_raw(x, g)
        pay = np.zeros(g.NX * g.tile_bytes, np.uint8)
        pay[: raw.nbytes] = raw
        self.buf[:, :, HDR:] = pay.reshape(g.NX, 1, g.tile_bytes)
        return self.buf.reshape(-1)


def unpack_y(y: np.ndarray, g: Geometry, M0: int | None = None) -> np.ndarray:
    """Joined NPU outputs -> (M0,) float32: drop XLOAD dummies; the cores keep a
    running sum over K chunks, so the last chunk's object holds the result."""
    y = y.reshape(g.ncols, g.y_objs, NROWS, NT, g.TM)[:, g.NX :, :, 0]  # token 0
    y = y.reshape(g.ncols, g.T, g.KC, NROWS, g.TM)[:, :, -1]
    y = y.reshape(g.M)
    return y if M0 is None else y[:M0]


def reference(codes: np.ndarray, scales: np.ndarray, x: np.ndarray) -> np.ndarray:
    w = (codes.astype(np.float32) - 1.0).reshape(codes.shape[0], -1, QK)
    w = w * scales.astype(np.float32)[:, :, None]
    return w.reshape(codes.shape[0], -1) @ x.astype(np.float32)
