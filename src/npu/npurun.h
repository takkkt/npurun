// npurun - ternary (PQ2_0) matrix-vector products on AMD XDNA2 NPUs.
//
// One universal core program (tgemv.xclbin) runs on all 32 AI Engine cores;
// weight matrices live in NPU-visible host memory in the npurun stream format
// (see kernels/tgemv.cc, npurun/pack.py). A "group" is 1..3 matrices sharing
// the same input vector (e.g. q/k/v or ffn gate/up); one NPU launch computes
// every member of a group.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace npurun {

// Kernel geometry (must match the compiled core program).
constexpr int TM = 32;        // rows per core tile
constexpr int TK = 1024;      // columns per K chunk
constexpr int KMAX = 17408;   // widest K one single-token launch covers (wider K is sliced)
constexpr int NCOLS = 8;      // AIE columns
constexpr int NROWS = 4;      // AIE core rows
constexpr int QK = 128;       // PQ2_0 group size
// per 8 rows x 1024 columns: 26 x 64 bytes of base-3 codes + 64 fp16 scales (1.75 bits/weight)
constexpr int SG_BYTES = 26 * 64 + 64 * 2;
constexpr int HDR = 64;
constexpr int TILE_BYTES = (TM / 8) * (TK / 1024) * SG_BYTES;
constexpr int OBJ_BYTES = HDR + TILE_BYTES;
constexpr int ROW_MULTIPLE = NCOLS * NROWS * TM; // padded row granularity
constexpr int MAX_GROUP = 3;
// compute() with ntok > 1 streams the weights once per 4 tokens (batch launches);
// NPURUN_BATCH=0 falls back to one weight pass per token.

// PQ2_0 block as stored in GGUF: fp16 scale + 128 2-bit codes.
struct BlockPQ2 {
    uint16_t d;
    uint8_t qs[QK / 4];
};

struct Matrix {
    int64_t M = 0;       // real rows
    int64_t Mpad = 0;    // rows padded to ROW_MULTIPLE
    int64_t K = 0;       // real columns (multiple of QK)
    int64_t Kpad = 0;    // columns padded to TK (zero weights beyond K)
    int64_t half_objs = 0;   // weight pairs per shim channel (T*KC)
    int64_t base = 0;        // byte offset of this matrix's stream in the group BO
    int64_t bytes = 0;       // packed stream bytes
};

class Device;

class Group {
  public:
    // Pack one member from row-major PQ2_0 blocks (M rows x K/128 blocks).
    // Rows may arrive in several calls: [row0, row0+nrows).
    void pack(int member, const BlockPQ2 *blocks, int64_t row0, int64_t nrows);
    // y_j = W_j x for every member into internal buffers: result(j) is
    // row-major [ntok][M_j]. x is [ntok] rows of K floats, row stride ldx floats.
    void compute(const float *x, int64_t ldx, int ntok);
    const float *result(int member) const;
    // compute() + copy into caller rows: ys[j] is [ntok][ld_j].
    void run(const float *x, float *const *ys, const int64_t *ld, int ntok);

    int size() const { return (int)mats.size(); }
    const Matrix &matrix(int j) const { return mats[j]; }
    int64_t K() const { return mats.empty() ? 0 : mats[0].K; }

  private:
    friend class Device;
    Device *dev = nullptr;
    std::vector<Matrix> mats;
    struct Impl;
    std::shared_ptr<Impl> impl;
};

class Device {
  public:
    // xclbin: the universal core program; loads onto NPU 0.
    explicit Device(const std::string &xclbin_path);
    // Same, from an in-memory xclbin image (e.g. the one embedded at build time).
    Device(const void *xclbin_data, size_t xclbin_size);
    ~Device();

    // Create a group of 1..3 matrices sharing K; allocates NPU-visible memory.
    std::unique_ptr<Group> make_group(const std::vector<std::pair<int64_t, int64_t>> &shapes_MK);

    struct Stats {
        uint64_t launches = 0;
        double npu_us = 0, host_us = 0;
    } stats;

    struct Impl;
    std::unique_ptr<Impl> impl;

  private:
    void open(); // bind impl->xclbin to NPU 0 and allocate shared buffers
};

// Optional: switch the NPU power mode ("turbo", "performance", "default") via xrt-smi.
bool set_power_mode(const char *mode);

} // namespace npurun
