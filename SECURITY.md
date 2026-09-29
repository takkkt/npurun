# Security Policy

## Reporting a vulnerability

Please report security issues privately through
[GitHub security advisories](https://github.com/wybe-labs/npurun/security/advisories/new),
not as public issues. We aim to respond within a week.

Issues in code inherited unchanged from llama.cpp should also be reported upstream:
see [llama.cpp's security policy](https://github.com/ggml-org/llama.cpp/blob/master/SECURITY.md).

## Using npurun safely

npurun inherits llama.cpp's security model:

- Only load model files from sources you trust; GGUF parsing runs native code on untrusted input.
- `llama-server` listens on 127.0.0.1 by default. Do not expose it to a network without
  an authenticating reverse proxy and `--api-key`.
- The NPU runtime shares host memory with the NPU driver; run one npurun process at a time
  per NPU.
