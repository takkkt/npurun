#pragma once

#include "ggml-backend.h"

// AMD XDNA2 NPU offload of PQ2_0 matrix-vector products (npurun), exposed as a
// CPU extra buffer type: weights placed in this buffer are split row-wise
// between the NPU and the CPU repack kernels, which run concurrently.
// Returns nullptr when no NPU/runtime is available or GGML_NPU=0.
ggml_backend_buffer_type_t ggml_backend_cpu_npu_buffer_type(void);
