"""Ahead-of-time compile of npurun designs into build/ (xclbin + per-shape insts)."""

from __future__ import annotations

from pathlib import Path

from .design import tgemv
from .pack import Geometry

BUILD_DIR = Path(__file__).resolve().parent.parent / "build"


def build_tgemv(g: Geometry, force: bool = False, nop: bool = False) -> tuple[Path, Path]:
    """Compile the design for one shape; returns (xclbin, insts).

    The xclbin does not depend on the shape, so any shape's xclbin can host
    every other shape's insts (see design.py).
    """
    name = f"tg5_M{g.M}_K{g.K}_tm{g.TM}_tk{g.TK}_c{g.ncols}" + ("_nop" if nop else "")
    d = BUILD_DIR / name
    xclbin, insts = d / "final.xclbin", d / "insts.bin"
    if force or not (xclbin.exists() and insts.exists()):
        d.mkdir(parents=True, exist_ok=True)
        design = tgemv.specialize(M=g.M, K=g.K, TM=g.TM, TK=g.TK, ncols=g.ncols, nop=nop)
        xclbin, insts = design.compile(xclbin_path=xclbin, inst_path=insts)
    return Path(xclbin), Path(insts)
