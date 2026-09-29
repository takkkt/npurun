# Contributing to npurun

Thanks for helping! npurun is MIT licensed and open to everyone. It is a fork of
llama.cpp (via PrismML's fork), so most of the tree is upstream code.

## Where changes belong

| Change | Where |
|---|---|
| NPU offload, NPU kernel, packing, launch planning | here: `src/npu/`, `kernel/`, `tools/` |
| Docs, launchers, packaging, CI | here: `docs/`, `scripts/`, `.github/` |
| Hooks into llama.cpp | here: `patches/llama.cpp.patch` (keep it minimal) |
| General llama.cpp features and fixes | upstream: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (we bump the pinned submodule regularly) |
| Ternary formats (PQ2_0 / PTQ1_0) on other backends | [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) |

## Getting started

1. Build with the preset (see [docs/guide.md](docs/guide.md#install)).
2. Most work needs no NPU toolchain. Changing the NPU kernel needs IRON / MLIR-AIE:
   see [kernel/README.md](kernel/README.md).
3. AI agents: read [AGENTS.md](AGENTS.md) first.

## Before opening a pull request

Run on an XDNA 2 machine and paste the results in the PR:

- Correctness: `npurun-test` for the shapes you touched (worst cosine must be > 0.999), e.g.
  `build\bin\npurun-test src\npu\tgemv.xclbin 5120 17408,17408 10 4`
- Model output: same text as the CPU path (`GGML_NPU=0`) for a greedy run (`--temp 0`), or
  equal perplexity with `llama-perplexity`.
- Speed: `llama-bench -t 6 --poll 100 -mmp 0 -p 64 -n 16` before and after, same conditions.

Keep PRs focused, follow the surrounding code style, and explain why, not only what.
AI-assisted contributions are welcome; you are responsible for every line you submit
and should say how AI was used.

## Reporting bugs

Open an issue with your processor, `xrt-smi examine` output, the model file, the exact
command, and the log (the `npu:` lines matter most).
