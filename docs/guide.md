# npurun: ternary LLMs on the AMD Ryzen AI NPU

npurun runs **ternary (1.58-bit) GGUF models** — PrismML's Bonsai family and any other
model stored as `PQ2_0` or `PTQ1_0` — on the **AMD XDNA 2 NPU** found in Ryzen AI 300/400
laptops, together with the CPU. It is a fork of llama.cpp: everything you know from
llama.cpp (`llama-cli`, `llama-server`, `llama-bench`, GGUF files, the OpenAI-compatible
API) works the same way. The NPU is used automatically when it is present.

- [Requirements](#requirements)
- [Install](#install)
- [Run](#run)
- [Which models work](#which-models-work)
- [Performance](#performance)
- [Settings](#settings)
- [How it works](#how-it-works)
- [Troubleshooting](#troubleshooting)

## Requirements

| | |
|---|---|
| OS | Windows 11 (x64). Linux is not supported yet. |
| NPU | AMD XDNA 2 ("NPU Strix / Krackan / Gorgon Point"): Ryzen AI 300 and 400 series |
| Driver | AMD NPU driver with XRT 2.19 or newer (ships with AMD Software / Adrenalin). Check with `C:\Windows\System32\AMD\xrt-smi.exe examine` |
| RAM | Model size + ~1 GB. The 27B model (6.7 GB) wants 16 GB of RAM, 32 GB recommended |
| To build | Visual Studio 2022 Build Tools (C++ workload + "C++ Clang Compiler"), CMake ≥ 3.21, Ninja, Git |

The NPU program is prebuilt and embedded in the binaries; you do **not** need AMD's
compiler toolchain (IRON / MLIR-AIE) unless you want to change the NPU kernel
([kernel/](../kernel/README.md)).

## Install

### Download (no build tools needed)

Get `npurun-<version>-win-x64.zip` from
[Releases](https://github.com/wybe-labs/npurun/releases/latest) and unzip it anywhere. It
contains every llama.cpp program (`llama-cli`, `llama-server`, `llama-bench`, ...), the
`npurun-chat.cmd` / `npurun-server.cmd` launchers and the Visual C++ runtime. The NPU
runtime comes from your AMD driver.

### Build from source

Same flow as llama.cpp. From a **"x64 Native Tools Command Prompt for VS 2022"**:

```bat
git clone https://github.com/wybe-labs/npurun
cd npurun
cmake --preset release
cmake --build build -j
```

The binaries land in `build\bin`. The first configure fetches llama.cpp (the
`third_party/llama.cpp` submodule, pinned), applies npurun's small patch to it, and
downloads the XRT SDK (headers and import library, ~70 MB) into the build folder. The
XRT runtime DLL comes from your NPU driver, so nothing is installed system-wide.

Offline or custom XRT: unpack the [XRT Windows SDK](https://github.com/Xilinx/XRT/releases)
anywhere and add `-DNPURUN_XRT_DIR=<dir that contains include\ and lib\>`.

The preset targets Zen 5 (every CPU with an XDNA 2 NPU): AVX-512 on, no OpenMP runtime
needed. If you configure without it, keep AVX-512 enabled (e.g. `-DGGML_NATIVE=ON`);
without it the CPU share runs more than twice as slow.

## Run

Download a model, e.g. Bonsai 2 27B (6.7 GB):

```bat
hf download prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-PQ2_0.gguf --local-dir models
```

Chat:

```bat
build\bin\llama-cli -m models\Ternary-Bonsai-2-27B-PQ2_0.gguf -t 6 --poll 100 --no-mmap -c 8192
```

Server (OpenAI-compatible, `http://127.0.0.1:8080/v1`):

```bat
build\bin\llama-server -m models\Ternary-Bonsai-2-27B-PQ2_0.gguf -t 6 --poll 100 --no-mmap -c 8192 -np 1
```

With the release zip, run the same commands inside the unzipped folder, without the
`build\bin\` prefix.

`npurun-chat.cmd` and `npurun-server.cmd` (in the zip, or `scripts\` in the source
tree) wrap these commands with the recommended flags. At start-up you should see `npu: XDNA2 NPU ready` in the log.

Recommended flags and why:

| Flag | Why |
|---|---|
| `-t 6` | The NPU does most of the matrix work; fewer CPU threads synchronize faster and leave power budget for the NPU. Try 6–8. |
| `--poll 100` | Keeps CPU threads awake between NPU calls. Without it, every small operation pays a wake-up delay. |
| `--no-mmap` | Loads the CPU-side tensors into RAM instead of mapping the file, so they are not paged out while the NPU buffers occupy memory. |
| `-np 1` (server) | One conversation slot; several slots run as several sequences and are slower on this path. |

## Which models work

Any GGUF whose large weight matrices are **`PQ2_0`** or **`PTQ1_0`** (ternary, group 128):

| Model | Status |
|---|---|
| Ternary Bonsai 2 27B (`PQ2_0`, `PTQ1_0`) | tested: output identical to the CPU path |
| Ternary Bonsai 8B (`PQ2_0`, `PTQ1_0`) | tested: perplexity equal to the CPU path |
| Ternary Bonsai 1.7B / 4B | supported (any width is padded to the NPU tile size); not benchmarked |
| Larger ternary models | supported: wide layers are split across NPU launches (tested up to 32768-wide matrices); memory is the limit |
| `Q2_0` (group 64), 1-bit `Q1_0`, ordinary Q4/Q8/F16 models | run on the CPU as in upstream llama.cpp |

Everything that is not a ternary matrix multiplication (attention, recurrent layers,
norms, embeddings) runs on the CPU exactly as in llama.cpp.

## Performance

Measured on a Ryzen AI 9 465 laptop (32 GB LPDDR5X) with background load, so ±15 % run
to run. Decode = generating tokens; prompt = processing the input.

| Bonsai 2 27B (PQ2_0) | CPU only (12 threads) | npurun (NPU + CPU, 6 threads) |
|---|---|---|
| Decode | 3.3 – 3.9 tok/s | **4.1 – 5.2 tok/s** |
| Prompt (64 tokens) | 10.1 tok/s | **12.2 – 13.8 tok/s** |

Honest caveats:

- The NPU reads memory at ~35–44 GB/s, the CPU at up to ~70 GB/s. On **smaller models
  (8B and below) the CPU alone is often as fast or faster** for decoding; the NPU still
  takes most of the load off the CPU, which saves power and keeps the machine responsive.
- On `PTQ1_0` files the NPU path is several times faster than the CPU, because the CPU
  has no optimized kernel for that format.
- Speculative decoding works (it is lossless) but only pays off on repetitive text
  (code, edits); see `--spec-type` in llama.cpp.

## Settings

Environment variables (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `GGML_NPU` | on | `0` disables the NPU (pure CPU, same as upstream) |
| `GGML_NPU_SPLIT` | `0.75` | Fraction of each ternary matrix's rows computed on the NPU; the CPU does the rest at the same time. `1.0` = NPU only (lowest CPU/power use) |
| `GGML_NPU_CPU_COPY` | auto | `1` keeps a full CPU copy of the weights so long prompts run on the CPU's fast GEMM; auto = on when the weights are smaller than 4 GB |
| `GGML_NPU_PMODE` | `turbo` | NPU power mode while running (`turbo`, `performance`, `balanced`, `powersaver`, `none`); restored to `default` on exit |
| `GGML_NPU_LOCK` | on | `0` stops pinning the CPU share of the weights in RAM |
| `GGML_NPU_STATS` | off | `1` prints NPU launch statistics at exit |
| `GGML_NPU_XCLBIN` | embedded | Load the NPU core program from a file instead (kernel development) |
| `NPURUN_BATCH` | on | `0` processes prompts one token per NPU weight pass (debugging) |
| `LLAMA_N_RS_SEQ` | 1 | Recurrent rollback window; ≥1 enables in-place recurrent state updates for hybrid models (qwen35). `0` = upstream behaviour |

## How it works

- **Weights.** At load time every ternary matrix is split row-wise: 75 % of the rows are
  repacked into NPU memory in a native format (5 ternary weights per byte + fp16 scales
  = 1.75 bits/weight, which is within 1 % of the information content of ternary
  weights), the rest stay on the CPU in llama.cpp's AVX-512 repacked layout.
- **One NPU program for all shapes.** All 32 AI Engine cores run one shape-agnostic
  kernel; each matrix call only differs in a small DMA instruction stream generated at
  run time (C++ transaction builders emitted by AMD's MLIR-AIE compiler).
- **Fused calls.** Matrices that read the same input (q/k/v, gate/up, qkv/gate) run in a
  single NPU launch. The NPU and the CPU threads work on the same operation at the same
  time.
- **Batching.** Prompts are processed 4 tokens per pass over the weights; wide layers are
  split into K slices that fit the cores' 64 KB memory.
- **Accuracy.** Activations are quantized to int8 per group of 128 (the models apply a
  Hadamard rotation first, which keeps this accurate); results match the CPU path within
  normal quantization noise.

## Troubleshooting

| Symptom | Fix |
|---|---|
| No `npu: XDNA2 NPU ready` line | Check the driver: `xrt-smi examine` must list an NPU. Update AMD Software. `GGML_NPU` must not be `0`. |
| `npu: disabled (...)` | The NPU could not be opened (another app holding it, old driver). The run continues on the CPU. |
| Very slow or stuttering | Too little free RAM: close memory-heavy apps, use `--no-mmap`, keep `-c` modest. Other programs using the CPU heavily also slow it down. |
| `could not pin CPU weights in RAM` | Harmless; the weights may be paged under memory pressure. |
| Build: XRT SDK download fails | Download it yourself and pass `-DNPURUN_XRT_DIR`. |
