"""Lean XRT host runtime for npurun.

One xclbin (the shape-agnostic core program) is loaded once; every weight
matrix brings its own small DMA instruction stream. Weights are uploaded once
into NPU-visible memory; a call writes the activations, launches, and reads
back the per-chunk partial outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyxrt

from .pack import Geometry, XPacker, unpack_y

TO_DEV = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
FROM_DEV = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

_DEVICE = None


def device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = pyxrt.device(0)
    return _DEVICE


class NpuProgram:
    """The loaded core program (xclbin) and its hardware context."""

    def __init__(self, xclbin_path: Path):
        dev = device()
        self.xclbin = pyxrt.xclbin(str(xclbin_path))
        dev.register_xclbin(self.xclbin)
        self.ctx = pyxrt.hw_context(dev, self.xclbin.get_uuid())
        self.kernel = pyxrt.kernel(self.ctx, self.xclbin.get_kernels()[0].get_name())

    def insts(self, insts_path: Path) -> tuple[pyxrt.bo, int]:
        data = np.fromfile(insts_path, dtype=np.uint32)
        bo = pyxrt.bo(device(), data.nbytes, pyxrt.bo.cacheable, self.kernel.group_id(1))
        bo.write(data.view(np.uint8), 0)
        bo.sync(TO_DEV)
        return bo, data.nbytes

    def buffer(self, nbytes: int, arg_index: int) -> pyxrt.bo:
        """Host-visible BO for runtime argument ``arg_index`` (0-based over data args)."""
        return pyxrt.bo(device(), nbytes, pyxrt.bo.host_only, self.kernel.group_id(3 + arg_index))

    def run(self, insts, *bos):
        h = self.kernel(3, insts[0], insts[1], *bos)
        state = h.wait()
        if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"NPU run failed: {state}")


class TernaryGemv:
    """y = W x for one packed ternary weight matrix resident in NPU-visible memory."""

    def __init__(self, program: NpuProgram, insts_path: Path, wpacked: np.ndarray, g: Geometry, M0: int):
        self.p = program
        self.g = g
        self.M0 = M0
        self.insts = program.insts(insts_path)
        self.w = program.buffer(wpacked.nbytes, 0)
        self.w.write(wpacked.view(np.uint8), 0)
        self.w.sync(TO_DEV)
        self.x = program.buffer(g.x_bytes, 1)
        self.y = program.buffer(g.y_floats * 4, 2)
        self.xp = XPacker(g)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.x.write(self.xp.pack(x), 0)
        self.x.sync(TO_DEV)
        self.p.run(self.insts, self.w, self.x, self.y)
        self.y.sync(FROM_DEV)
        y = np.frombuffer(self.y.read(self.g.y_floats * 4, 0), dtype=np.float32)
        return unpack_y(y, self.g, self.M0)
