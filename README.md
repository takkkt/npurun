# npurun

**Run ternary LLMs on the AMD Ryzen AI NPU.** npurun adds XDNA 2 NPU offload to
[llama.cpp](https://github.com/ggml-org/llama.cpp), so ternary (1.58-bit) models such as
[Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) run on the
NPU of Ryzen AI 300 / 400 / Max laptops together with the CPU, with the same GGUF files,
tools and OpenAI-compatible API you know from llama.cpp.

By [Wybe Labs](https://github.com/wybe-labs). MIT licensed, contributions welcome.

**Needs:** Windows 11, an AMD Ryzen AI 300 / 400 / Max processor, the AMD NPU driver
(part of AMD Software / Adrenalin).

## Quick start: download

1. Download `npurun-<version>-win-x64.zip` from
   [Releases](https://github.com/wybe-labs/npurun/releases/latest) and unzip it.
2. Get a model:
   ```bat
   pip install -U huggingface_hub
   hf download prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-PQ2_0.gguf --local-dir models
   ```
3. Chat, or start an OpenAI-compatible server with a web UI at http://127.0.0.1:8080:
   ```bat
   npurun-chat.cmd models\Ternary-Bonsai-2-27B-PQ2_0.gguf
   npurun-server.cmd models\Ternary-Bonsai-2-27B-PQ2_0.gguf
   ```

## Build from source

From an "x64 Native Tools Command Prompt for VS 2022":

```bat
git clone https://github.com/wybe-labs/npurun
cd npurun
cmake --preset release
cmake --build build -j
scripts\npurun-chat.cmd models\Ternary-Bonsai-2-27B-PQ2_0.gguf
```

The first configure fetches llama.cpp and AMD's XRT headers automatically.

## Performance

| Bonsai 2 27B on a Ryzen AI 9 465 | CPU only | npurun (NPU + CPU) |
|---|---|---|
| Generation | 3.3 – 3.9 tok/s | **4.1 – 5.2 tok/s** |
| Prompt processing | 10.1 tok/s | **12.2 – 13.8 tok/s** |

Every ternary (`PQ2_0` / `PTQ1_0`) matrix multiplication is split 75/25 between the NPU
and the CPU; everything else, and every non-ternary model, runs on the CPU exactly as in
llama.cpp. Full guide: [docs/guide.md](docs/guide.md).

## What is in this repository

This repository contains only the NPU integration; llama.cpp comes in as a pinned
submodule.

| Path | What |
|---|---|
| `src/npu/` | The NPU backend: weight placement and NPU/CPU split, XRT runtime, the prebuilt NPU program |
| `kernel/` | The AI Engine kernel and its IRON designs (only needed to change the NPU program) |
| `tools/` | `npurun-test` (correctness and timing), `npurun-bw` (memory bandwidth) |
| `patches/llama.cpp.patch` | ~30 lines of hooks into llama.cpp, applied at configure time |
| `third_party/llama.cpp` | [PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (adds the ternary formats), pinned |
| `scripts/` | Launchers and release packaging |

Contributing: [CONTRIBUTING.md](CONTRIBUTING.md). For AI agents: [AGENTS.md](AGENTS.md).

## Credits

npurun builds on [llama.cpp](https://github.com/ggml-org/llama.cpp) by the ggml authors and
[PrismML's fork](https://github.com/PrismML-Eng/llama.cpp) with the ternary Bonsai formats,
both MIT licensed, and on AMD's [XRT](https://github.com/Xilinx/XRT) and
[MLIR-AIE / IRON](https://github.com/Xilinx/mlir-aie).
