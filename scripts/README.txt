npurun - llama.cpp for the AMD Ryzen AI NPU
https://github.com/wybe-labs/npurun

Runs GGUF language models with llama.cpp's tools. Ternary models (PQ2_0 / PTQ1_0,
e.g. PrismML's Ternary Bonsai family) run on the XDNA 2 NPU together with the CPU;
every other GGUF model runs on the CPU exactly as in llama.cpp.

NEEDS
  Windows 11 x64, an AMD Ryzen AI 300 / 400 / Max processor, and the AMD NPU driver
  (included in AMD Software / Adrenalin). Check with:
      C:\Windows\System32\AMD\xrt-smi.exe examine

1. GET A MODEL (example: Ternary Bonsai 2 27B, 6.7 GB; 16 GB RAM minimum, 32 GB recommended)
      pip install -U huggingface_hub
      hf download prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-PQ2_0.gguf --local-dir models
   Smaller: prism-ml/Ternary-Bonsai-8B-gguf (Ternary-Bonsai-8B-PQ2_0.gguf, 2 GB).

2. CHAT
      npurun-chat.cmd models\Ternary-Bonsai-2-27B-PQ2_0.gguf

3. OR RUN AN OPENAI-COMPATIBLE SERVER (web UI + API at http://127.0.0.1:8080)
      npurun-server.cmd models\Ternary-Bonsai-2-27B-PQ2_0.gguf

The log shows "npu: XDNA2 NPU ready" when the NPU is in use. All llama.cpp programs
(llama-cli, llama-server, llama-bench, llama-perplexity, ...) are included and accept
the usual llama.cpp options; for NPU runs use -t 6 --poll 100 --no-mmap.

Settings, troubleshooting and how it works:
  https://github.com/wybe-labs/npurun/blob/main/docs/guide.md

MIT License - see LICENSE.txt. Built on llama.cpp (MIT, see LICENSE-llama.cpp.txt);
other components: THIRD_PARTY_NOTICES.md. Includes the Microsoft Visual C++ runtime DLLs
(msvcp140, vcruntime140, vcruntime140_1), redistributed under Microsoft's terms.
