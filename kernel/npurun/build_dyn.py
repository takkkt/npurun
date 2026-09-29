"""Rebuild the NPU core program and the C++ transaction builders.

Outputs (src/npu/, embedded into the build):
  tgemv.xclbin            shape-agnostic core program (from the static design)
  gen/tgemv_dyn<n>.inc    generate_txn_tgemv<n>(...): n = 1..3 fused matrices,
                          n = 0 one K-sliced matrix

usage (IRON environment active): python -m npurun.build_dyn
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from .build import build_tgemv
from .design_dyn import tgemv_dyn
from .pack import Geometry

ROOT = Path(__file__).resolve().parent.parent          # kernel/
NPU_DIR = ROOT.parent / "src" / "npu"
GEN = NPU_DIR / "gen"


def build_all(nmats=(0, 1, 2, 3)):
    GEN.mkdir(parents=True, exist_ok=True)
    # The core program comes from the static design: a DispatchTime build turns the
    # runtime scalars into extra kernel arguments, which the dynamic transactions
    # (buffers only) do not supply. Cores and fifos are identical in both designs.
    static_xclbin, _ = build_tgemv(Geometry(1024, 5120), force=True)
    shutil.copy(static_xclbin, NPU_DIR / "tgemv.xclbin")
    for n in nmats:
        d = ROOT / "build" / f"dyn{n}"
        d.mkdir(parents=True, exist_ok=True)
        design = tgemv_dyn.specialize(nmat=n)
        xclbin, _ = design.compile(xclbin_path=d / "final.xclbin")
        cpp = next(Path(xclbin).parent.rglob("dispatch_gen.cpp"), None)
        if cpp is None:
            cands = sorted(Path.home().joinpath(".npu", "cache").rglob("dispatch_gen.cpp"),
                           key=lambda p: p.stat().st_mtime)
            cpp = cands[-1]
        src = cpp.read_text()
        src = re.sub(r"\bgenerate_txn_main_sequence\b", f"generate_txn_tgemv{n}", src)
        cut = src.find('extern "C" AIE_DISPATCH_EXPORT')  # drop per-file C dispatch shims
        if cut >= 0:
            src = src[:cut]
        (GEN / f"tgemv_dyn{n}.inc").write_text(src)
        print(f"nmat={n}: {xclbin}  builder from {cpp}")


if __name__ == "__main__":
    build_all()
