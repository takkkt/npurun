# npurun NPU kernel and tools

Everything needed to change the NPU side of npurun. Using npurun does **not** need any of
this: the compiled core program is committed (`src/npu/tgemv.xclbin`) and
embedded into the binaries at build time.

## Layout

| Path | What |
|---|---|
| `tgemv.cc` | AI Engine (AIE2P) kernel: ternary GEMV / 4-token batch GEMM, base-3 decode, running sums |
| `npurun/design.py` | IRON design (static shapes): 8 columns × 4 cores, weight/activation/output data movement |
| `npurun/design_dyn.py` | Same design with runtime (DispatchTime) sizes; its C++ transaction builders ship in `src/npu/gen/` |
| `npurun/build_dyn.py` | Rebuilds `tgemv.xclbin` + `gen/tgemv_dyn{0..3}.inc` in place |
| `npurun/pack.py`, `emulate.py` | Host packing format and a bit-exact NumPy emulator of the kernel |
| `native/test_npurun.cpp` | `npurun-test`: correctness vs a CPU reference + timing for any shape / token count |
| `native/bw_test.cpp` | `npurun-bw`: CPU / NPU / combined DRAM bandwidth |
| `scripts/` | `weight_stats.py` (ternary code entropy), `ggufinfo.py` (tensor shapes), `txndecode.py` (decode NPU instruction streams) |
| `../src/npu/npurun.cpp` | XRT host runtime: packing, launches, activation quantization |
| `../src/npu/npu.cpp` | ggml CPU extra buffer type: row split, fused groups, CPU/NPU overlap |

## Kernel toolchain (only for kernel changes)

1. Install Python 3.13, the AMD NPU driver, and set up IRON per
   [mlir-aie's native Windows guide](https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostWinNative.md)
   (`python utils\iron_setup.py`, XRT SDK at `C:\Xilinx\XRT`). Install the `mlir_aie`
   wheel that matches your mlir-aie checkout's commit:
   `pip install --pre "mlir_aie==<version>+g<commit>" -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4`
2. From a VS x64 prompt with the IRON environment active:

```bat
cd kernel
python -m npurun.build_dyn
```

3. Rebuild npurun (the new xclbin is re-embedded automatically) and run the tests:

```bat
build\bin\npurun-test src\npu\tgemv.xclbin 5120 10240,6144 50 1
build\bin\npurun-test src\npu\tgemv.xclbin 5120 17408,17408 10 4
```

Arguments: `xclbin K M0[,M1,M2] iterations tokens`. Every run prints the worst cosine
similarity against a double-precision CPU reference (must be > 0.999) and the time per call.

## Invariants to keep

- The core program (fifo topology, object sizes, kernel ABI) must be identical between
  `design.py` and `design_dyn.py`; the static design's xclbin runs the dynamic builders'
  instruction streams. A DispatchTime build's own xclbin adds scalar kernel arguments and
  must **not** be used.
- Object = 64-byte header + tile. XLOAD objects set `hdr[4]` (first K chunk of the launch)
  and `hdr[5]` (0 = single token, 4 = batch); TILE objects carry `hdr[1]` = K chunk.
- Every core emits `NT * TM` floats per object; the last K chunk's object of a row tile
  holds the finished (running-sum) result.
- Core data memory is 64 KB: the stack (8 KB) sits below the statics and grows upward;
  scalar fill loops were miscompiled — use vector stores.
- Shim DMA wrap dimensions are ≤ 1023 words; the runtime keeps launches within that and
  within 32-bit offsets (≤ 2 GiB per group buffer).
