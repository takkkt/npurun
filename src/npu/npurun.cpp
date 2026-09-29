// npurun runtime: XRT host side of the ternary GEMV NPU program.
#include "npurun.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

#include "gen/tgemv_dyn0.inc"
#include "gen/tgemv_dyn1.inc"
#include "gen/tgemv_dyn2.inc"
#include "gen/tgemv_dyn3.inc"

namespace npurun {

namespace {

constexpr int KIND_XLOAD = 1;
constexpr int KIND_TILE = 2;
constexpr int64_t MAX_DMA_DIM = 1023;           // shim BD wrap limit
constexpr int64_t MAX_HALF = MAX_DMA_DIM / 2;   // weight pairs per channel per fill
constexpr int64_t PAIR = 2 * OBJ_BYTES;

using clk = std::chrono::steady_clock;

// NPURUN_POLL=1: spin on the run state instead of sleeping on the completion interrupt.
bool poll_wait() {
    static const bool p = [] {
        const char *e = std::getenv("NPURUN_POLL");
        return e && std::atoi(e) != 0;
    }();
    return p;
}

constexpr float QMAX = 127.f; // int8 activation range

// 128 floats -> int8 with a per-group absmax scale, plus the sum of the codes.
inline void quantize_group(const float *x, int8_t *q, float &scale, int32_t &sum) {
#if defined(__AVX2__)
    const __m256 sign = _mm256_set1_ps(-0.0f);
    __m256 m = _mm256_setzero_ps();
    for (int e = 0; e < QK; e += 8) m = _mm256_max_ps(m, _mm256_andnot_ps(sign, _mm256_loadu_ps(x + e)));
    __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
    const float amax = _mm_cvtss_f32(m4);
    scale = amax > 0.f ? amax / QMAX : 1.f;
    const __m256 inv = _mm256_set1_ps(1.f / scale);
    __m256i acc = _mm256_setzero_si256();
    for (int e = 0; e < QK; e += 32) {
        __m256i a = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + e), inv));
        __m256i b = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + e + 8), inv));
        __m256i c = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + e + 16), inv));
        __m256i d = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + e + 24), inv));
        acc = _mm256_add_epi32(acc, _mm256_add_epi32(_mm256_add_epi32(a, b), _mm256_add_epi32(c, d)));
        __m256i ab = _mm256_packs_epi32(a, b), cd = _mm256_packs_epi32(c, d);
        __m256i v = _mm256_packs_epi16(ab, cd);
        v = _mm256_permutevar8x32_epi32(v, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256((__m256i *)(q + e), v);
    }
    __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4e));
    s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xb1));
    sum = _mm_cvtsi128_si32(s4);
#else
    float amax = 0.f;
    for (int e = 0; e < QK; e++) amax = std::max(amax, std::fabs(x[e]));
    scale = amax > 0.f ? amax / QMAX : 1.f;
    const float inv = 1.f / scale;
    sum = 0;
    for (int e = 0; e < QK; e++) {
        int v = (int)std::nearbyint(x[e] * inv);
        v = std::min(127, std::max(-127, v));
        q[e] = (int8_t)v;
        sum += v;
    }
#endif
}

int64_t xchunk_bytes() { return TK + (TK / QK) * 8; }

template <typename F> void parallel_for(int64_t n, F &&f) {
    int nt = (int)std::min<int64_t>(n, std::max(1u, std::thread::hardware_concurrency()));
    if (nt <= 1) {
        for (int64_t i = 0; i < n; i++) f(i);
        return;
    }
    std::atomic<int64_t> next{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < nt; t++)
        ts.emplace_back([&] {
            for (int64_t i; (i = next.fetch_add(1)) < n;) f(i);
        });
    for (auto &t : ts) t.join();
}

// ---------------------------------------------------------------- Device ----

constexpr int NTB = 4;                       // tokens per batch launch (kernel NT)
constexpr int64_t KBMAX = 6144;              // widest K one batch launch covers (kernel KBMAX)
constexpr int YOBJ = NROWS * TM * NTB;       // floats per joined output object
constexpr int64_t XPAIRS = 8;                // XLOAD pairs the X buffer can hold

int64_t xchunk_b() { return (int64_t)TK * NTB + (TK / QK) * NTB * 8; }
int64_t nx_for(int64_t bytes) { return (bytes + TILE_BYTES - 1) / TILE_BYTES; }

bool batch_enabled() {
    static const bool b = [] {
        const char *e = std::getenv("NPURUN_BATCH");
        return !(e && std::strcmp(e, "0") == 0);
    }();
    return b;
}

} // namespace

struct Device::Impl {
    xrt::device dev;
    xrt::xclbin xclbin;
    xrt::hw_context ctx;
    xrt::kernel kernel;
    xrt::bo xbo, ybo;
    size_t ybytes = 0;
    int ygen = 0; // bumped when ybo is reallocated
};

Device::Device(const std::string &xclbin_path) : impl(new Impl) {
    impl->xclbin = xrt::xclbin(xclbin_path);
    open();
}

Device::Device(const void *xclbin_data, size_t xclbin_size) : impl(new Impl) {
    const char *p = (const char *)xclbin_data;
    impl->xclbin = xrt::xclbin(std::vector<char>(p, p + xclbin_size));
    open();
}

void Device::open() {
    impl->dev = xrt::device(0);
    impl->dev.register_xclbin(impl->xclbin);
    impl->ctx = xrt::hw_context(impl->dev, impl->xclbin.get_uuid());
    impl->kernel = xrt::kernel(impl->ctx, impl->xclbin.get_kernels().at(0).get_name());
    const size_t xbytes = (size_t)XPAIRS * PAIR;
    impl->xbo = xrt::bo(impl->dev, xbytes, xrt::bo::flags::host_only, impl->kernel.group_id(4));
    std::memset(impl->xbo.map<uint8_t *>(), 0, xbytes);
}

Device::~Device() = default;

// ----------------------------------------------------------------- Group ----

namespace {
struct Chunk {           // row range of one member, packed contiguously
    int member;
    int64_t row0, rows;  // rows padded to ROW_MULTIPLE (rows beyond M are zero)
    int64_t half_objs;   // T * KC
    int64_t base;        // byte offset in the group BO
};
struct Call {            // one NPU launch
    std::vector<int> chunks;
    bool batch = false;  // NTB tokens per weight pass
    bool sliced = false; // batch over K chunks [kc0, kc0 + nkc) of one chunk
    int64_t kc0 = 0, nkc = 0;
    int64_t nx = 0;      // XLOAD pairs at the head of each channel stream
    int64_t ytot = 0;    // joined output objects per column
    xrt::bo insts;
    size_t insts_bytes = 0;
    xrt::run run;
    int ygen = -1;
};
} // namespace

struct Group::Impl {
    xrt::bo wbo;
    uint8_t *wmap = nullptr;
    std::vector<Chunk> chunks;
    std::vector<Call> calls;  // single-token launches
    std::vector<Call> bcalls; // batch launches
    std::vector<std::vector<float>> out; // per member [ntok][M]
};

std::unique_ptr<Group> Device::make_group(const std::vector<std::pair<int64_t, int64_t>> &shapes) {
    if (shapes.empty() || shapes.size() > (size_t)MAX_GROUP) throw std::runtime_error("npurun: bad group size");
    auto g = std::make_unique<Group>();
    g->dev = this;
    g->impl = std::make_shared<Group::Impl>();
    auto &gi = *g->impl;
    const int64_t K = shapes[0].second;
    if (K <= 0 || K % QK) throw std::runtime_error("npurun: K must be a positive multiple of 128");
    const int64_t Kp = (K + TK - 1) / TK * TK; // padded with zero weights / activations
    const int64_t KC = Kp / TK;

    int64_t off = 0;
    for (size_t j = 0; j < shapes.size(); j++) {
        Matrix m;
        m.M = shapes[j].first;
        m.K = shapes[j].second;
        m.Kpad = Kp;
        if (m.K != K) throw std::runtime_error("npurun: group members must share K");
        m.Mpad = (m.M + ROW_MULTIPLE - 1) / ROW_MULTIPLE * ROW_MULTIPLE;
        m.base = off;
        // split rows so that one chunk's pairs per channel stay within the DMA limit
        const int64_t max_T = std::max<int64_t>(1, MAX_HALF / KC);
        const int64_t max_rows = max_T * ROW_MULTIPLE;
        for (int64_t r0 = 0; r0 < m.Mpad; r0 += max_rows) {
            Chunk c;
            c.member = (int)j;
            c.row0 = r0;
            c.rows = std::min(max_rows, m.Mpad - r0);
            c.half_objs = (c.rows / ROW_MULTIPLE) * KC;
            c.base = off;
            off += c.half_objs * PAIR * 2 * NCOLS;
            gi.chunks.push_back(c);
        }
        m.bytes = off - m.base;
        m.half_objs = m.bytes / (PAIR * 2 * NCOLS);
        g->mats.push_back(m);
    }

    if (off >= ((int64_t)1 << 31)) throw std::runtime_error("npurun: group larger than 2 GiB (DMA offsets are 32-bit)");
    gi.wbo = xrt::bo(impl->dev, (size_t)off, xrt::bo::flags::host_only, impl->kernel.group_id(3));
    gi.wmap = gi.wbo.map<uint8_t *>();

    auto finish = [&](Call &call, const int32_t v[7], int nmat) {
        std::optional<std::vector<uint32_t>> txn;
        switch (nmat) {
        case 0: txn = generate_txn_tgemv0(v[0], v[1], v[2], v[3], v[4], v[5], v[6]); break;
        case 1: txn = generate_txn_tgemv1(v[0], v[1], v[2], v[3], v[4], v[5], v[6]); break;
        case 2: txn = generate_txn_tgemv2(v[0], v[1], v[2], v[3], v[4], v[5], v[6]); break;
        default: txn = generate_txn_tgemv3(v[0], v[1], v[2], v[3], v[4], v[5], v[6]); break;
        }
        if (!txn) throw std::runtime_error("npurun: transaction builder rejected the call");
        call.insts_bytes = txn->size() * 4;
        call.insts = xrt::bo(impl->dev, call.insts_bytes, xrt::bo::flags::cacheable, impl->kernel.group_id(1));
        call.insts.write(txn->data());
        call.insts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    };
    // Greedily form launches of <= 3 chunks with <= 1023 output objects per column.
    auto form = [&](std::vector<Call> &calls, int64_t nx, bool batch) {
        size_t i = 0;
        while (i < gi.chunks.size()) {
            Call call;
            call.batch = batch;
            call.nx = nx;
            call.ytot = nx;
            call.kc0 = 0;
            call.nkc = KC;
            while (i < gi.chunks.size() && call.chunks.size() < (size_t)MAX_GROUP &&
                   call.ytot + gi.chunks[i].half_objs <= MAX_DMA_DIM) {
                call.ytot += gi.chunks[i].half_objs;
                call.chunks.push_back((int)i++);
            }
            if (call.chunks.empty()) throw std::runtime_error("npurun: chunk too large");
            int32_t v[7] = {(int32_t)nx, 0, 0, 0, 0, 0, 0};
            for (size_t c = 0; c < call.chunks.size(); c++) {
                v[1 + 2 * c] = (int32_t)gi.chunks[call.chunks[c]].base;
                v[2 + 2 * c] = (int32_t)gi.chunks[call.chunks[c]].half_objs;
            }
            finish(call, v, (int)call.chunks.size());
            calls.push_back(std::move(call));
        }
    };
    // Wide K: one launch per (chunk, K slice) -- the cores keep only one slice of
    // the activations resident -- and the host sums the slice partials.
    auto slices = [&](std::vector<Call> &calls, int64_t skc, int64_t xcb, bool batch) {
        for (size_t ci = 0; ci < gi.chunks.size(); ci++) {
            const Chunk &ch = gi.chunks[ci];
            const int64_t T = ch.rows / ROW_MULTIPLE;
            for (int64_t kc0 = 0; kc0 < KC; kc0 += skc) {
                Call call;
                call.batch = batch;
                call.sliced = true;
                call.chunks = {(int)ci};
                call.kc0 = kc0;
                call.nkc = std::min(skc, KC - kc0);
                call.nx = nx_for(call.nkc * xcb);
                call.ytot = call.nx + T * call.nkc;
                int32_t v[7] = {(int32_t)call.nx, (int32_t)ch.base, (int32_t)T, (int32_t)KC,
                                (int32_t)kc0, (int32_t)call.nkc, 0};
                finish(call, v, 0);
                calls.push_back(std::move(call));
            }
        }
    };
    if (Kp <= KMAX) form(gi.calls, nx_for(KC * xchunk_bytes()), false);
    else slices(gi.calls, KMAX / TK, xchunk_bytes(), false);
    if (Kp <= KBMAX) form(gi.bcalls, nx_for(KC * xchunk_b()), true);
    else slices(gi.bcalls, KBMAX / TK, xchunk_b(), true);
    int64_t max_ytot = 0;
    for (auto &c : gi.calls) max_ytot = std::max(max_ytot, c.ytot);
    for (auto &c : gi.bcalls) {
        max_ytot = std::max(max_ytot, c.ytot);
        if (c.nx > XPAIRS) throw std::runtime_error("npurun: X buffer too small");
    }
    const size_t ybytes = (size_t)NCOLS * max_ytot * YOBJ * sizeof(float);
    if (ybytes > impl->ybytes) {
        impl->ybo = xrt::bo(impl->dev, ybytes, xrt::bo::flags::host_only, impl->kernel.group_id(5));
        impl->ybytes = ybytes;
        impl->ygen++;
    }
    return g;
}

// Packs member `member` (row-major PQ2_0, all M rows) into its chunks.
void Group::pack(int member, const BlockPQ2 *blocks, int64_t row0, int64_t nrows) {
    auto &gi = *impl;
    const Matrix &m = mats[member];
    const int64_t K = m.K, KC = m.Kpad / TK, nb = K / QK;
    const int64_t row_end = row0 + nrows;
    for (const Chunk &ch : gi.chunks) {
        if (ch.member != member) continue;
        const int64_t Mcol = ch.rows / NCOLS;       // rows per AIE column
        const int64_t T = Mcol / (NROWS * TM);
        const int64_t nobj = (int64_t)NCOLS * 2 * T * KC * 2;
        uint8_t *dst0 = gi.wmap + ch.base;
        parallel_for(nobj, [&](int64_t o) {
            // o = (((c*2 + h)*T + t)*KC + kc)*2 + rr
            int64_t rem = o;
            const int64_t rr = rem % 2; rem /= 2;
            const int64_t kc = rem % KC; rem /= KC;
            const int64_t t = rem % T; rem /= T;
            const int64_t h = rem % 2; rem /= 2;
            const int64_t c = rem;
            const int64_t core = 2 * h + rr;
            const int64_t trow = ch.row0 + c * Mcol + (t * NROWS + core) * TM;
            // only (re)pack objects whose rows intersect [row0, row_end)
            if (trow + TM <= row0 || trow >= row_end) return;
            uint8_t *obj = dst0 + o * OBJ_BYTES;
            int32_t *hdr = (int32_t *)obj;
            std::memset(hdr, 0, HDR);
            hdr[0] = KIND_TILE;
            hdr[1] = (int32_t)kc;
            uint8_t *pay = obj + HDR;
            for (int rb = 0; rb < TM / 8; rb++) {
                for (int sg = 0; sg < TK / 1024; sg++) {
                    uint8_t *blk = pay + (rb * (TK / 1024) + sg) * SG_BYTES;
                    uint16_t *sc = (uint16_t *)(blk + 26 * 64);
                    const int64_t k0 = kc * TK + sg * 1024;
                    for (int r = 0; r < 8; r++) {
                        const int64_t row = trow + rb * 8 + r;
                        const bool valid = row < m.M && row >= row0 && row < row_end;
                        const BlockPQ2 *rowb = valid ? blocks + (row - row0) * nb : nullptr;
                        for (int n = 0; n < 8; n++)
                            sc[n * 8 + r] = valid && k0 + n * QK < K ? rowb[k0 / QK + n].d : (uint16_t)0;
                        // word v, byte kk*8+r: sum_t code(r, 8*(5v+t)+kk) * 3^t (chunks < 128)
                        for (int v = 0; v < 26; v++) {
                            for (int kk = 0; kk < 8; kk++) {
                                int byte = 0, p3 = 1;
                                for (int tt = 0; tt < 5; tt++, p3 *= 3) {
                                    const int chunk = 5 * v + tt;
                                    if (chunk >= 128) break;
                                    int code = 1; // zero weight for padding rows
                                    const int64_t col = k0 + 8 * chunk + kk;
                                    if (valid && col < K) { // columns beyond K: zero weight
                                        const BlockPQ2 &b = rowb[col / QK];
                                        const int e = (int)(col % QK);
                                        code = (b.qs[e / 4] >> (2 * (e % 4))) & 3;
                                        if (code > 2) code = 2; // PQ2_0 code 3 (+2) is not ternary; clamp
                                    }
                                    byte += code * p3;
                                }
                                blk[v * 64 + kk * 8 + r] = (uint8_t)byte;
                            }
                        }
                    }
                }
            }
        });
        gi.wbo.sync(XCL_BO_SYNC_BO_TO_DEVICE, (size_t)(nobj * OBJ_BYTES), (size_t)ch.base);
    }
}

const float *Group::result(int member) const { return impl->out[member].data(); }

void Group::run(const float *x, float *const *ys, const int64_t *ld, int ntok) {
    compute(x, mats[0].K, ntok);
    for (size_t j = 0; j < mats.size(); j++)
        for (int t = 0; t < ntok; t++)
            std::memcpy(ys[j] + (int64_t)t * ld[j], impl->out[j].data() + (int64_t)t * mats[j].M,
                        (size_t)mats[j].M * sizeof(float));
}

namespace {
// Launch one prepared call and wait; returns false on failure.
void launch(Device &dev, Call &call, xrt::bo &wbo) {
    auto &di = *dev.impl;
    if (!call.run) {
        call.run = xrt::run(di.kernel);
        call.run.set_arg(0, 3);
        call.run.set_arg(1, call.insts);
        call.run.set_arg(2, (unsigned)call.insts_bytes);
        call.run.set_arg(3, wbo);
        call.run.set_arg(4, di.xbo);
    }
    if (call.ygen != di.ygen) {
        call.run.set_arg(5, di.ybo);
        call.ygen = di.ygen;
    }
    auto a = clk::now();
    call.run.start();
    if (poll_wait()) {
        while (call.run.state() != ERT_CMD_STATE_COMPLETED) {
            const auto st = call.run.state();
            if (st == ERT_CMD_STATE_ERROR || st == ERT_CMD_STATE_ABORT || st == ERT_CMD_STATE_TIMEOUT)
                throw std::runtime_error("npurun: NPU run failed");
        }
    } else {
        call.run.wait();
    }
    dev.stats.launches++;
    dev.stats.npu_us += std::chrono::duration<double, std::micro>(clk::now() - a).count();
    di.ybo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, (size_t)NCOLS * call.ytot * YOBJ * sizeof(float), 0);
}

// Write `nx` XLOAD pairs carrying `bytes` of activations starting at `src`.
void write_xload(uint8_t *xmap, const int8_t *src, int64_t bytes, int64_t nx, int32_t kc0, int32_t mode) {
    for (int64_t i = 0; i < nx; i++) {
        const int64_t n = std::min<int64_t>(TILE_BYTES, bytes - i * TILE_BYTES);
        for (int cp = 0; cp < 2; cp++) {
            uint8_t *obj = xmap + (i * 2 + cp) * OBJ_BYTES;
            int32_t *hdr = (int32_t *)obj;
            std::memset(hdr, 0, HDR);
            hdr[0] = KIND_XLOAD;
            hdr[2] = (int32_t)(i * TILE_BYTES);
            hdr[3] = (int32_t)((n + 63) / 64 * 64);
            hdr[4] = kc0;
            hdr[5] = mode;
            std::memcpy(obj + HDR, src + i * TILE_BYTES, (size_t)n);
        }
    }
}
} // namespace

void Group::compute(const float *x, int64_t ldx, int ntok) {
    auto &gi = *impl;
    gi.out.resize(mats.size());
    for (size_t j = 0; j < mats.size(); j++)
        if ((int64_t)gi.out[j].size() < (int64_t)ntok * mats[j].M) gi.out[j].resize((size_t)(ntok * mats[j].M));
    auto &di = *dev->impl;
    const int64_t K = mats[0].K, Kp = mats[0].Kpad, KC = Kp / TK, ng = TK / QK;
    uint8_t *xmap = di.xbo.map<uint8_t *>();
    const float *ymap = di.ybo.map<const float *>();

    // Batch mode streams the weights once per NTB tokens; single mode once per token.
    const bool batch = ntok > 1 && batch_enabled();
    const int step = batch ? NTB : 1;
    std::vector<Call> &calls = batch ? gi.bcalls : gi.calls;
    const int64_t xcb = batch ? xchunk_b() : xchunk_bytes(); // activation bytes per K chunk
    std::vector<int8_t> raw((size_t)(KC * xcb + 64));
    std::vector<float> xpad(K == Kp ? 0 : (size_t)Kp, 0.f);  // zero-padded token when K % TK
    alignas(32) int8_t q[QK];

    for (int tb = 0; tb < ntok; tb += step) {
        const int nb = std::min(step, ntok - tb);
        auto t0 = clk::now();
        // ---- quantize the tokens into the launch's activation layout
        std::fill(raw.begin(), raw.end(), (int8_t)0);
        for (int t = 0; t < nb; t++) {
            const float *xt = x + (int64_t)(tb + t) * ldx;
            if (!xpad.empty()) {
                std::memcpy(xpad.data(), xt, (size_t)K * sizeof(float));
                xt = xpad.data();
            }
            for (int64_t kc = 0; kc < KC; kc++) {
                int8_t *xc = raw.data() + kc * xcb;
                if (batch) { // [128 chunks][NTB][8] | per group: sx[NTB], sumx[NTB]
                    float *gp = (float *)(xc + TK * NTB);
                    for (int64_t gidx = 0; gidx < ng; gidx++) {
                        float *sx = gp + gidx * NTB * 2;
                        int32_t *sm = (int32_t *)(sx + NTB);
                        quantize_group(xt + kc * TK + gidx * QK, q, sx[t], sm[t]);
                        for (int j = 0; j < 16; j++) std::memcpy(xc + ((gidx * 16 + j) * NTB + t) * 8, q + j * 8, 8);
                    }
                } else {     // xq[TK] | sx[ng] | sumx[ng]
                    float *sx = (float *)(xc + TK);
                    int32_t *sm = (int32_t *)(sx + ng);
                    for (int64_t gidx = 0; gidx < ng; gidx++)
                        quantize_group(xt + kc * TK + gidx * QK, xc + gidx * QK, sx[gidx], sm[gidx]);
                }
            }
        }
        // ---- launch every call; calls over the same K range share one activation load
        int64_t loaded_kc0 = -1, loaded_nkc = -1;
        for (auto &call : calls) {
            if (call.kc0 != loaded_kc0 || call.nkc != loaded_nkc) {
                write_xload(xmap, raw.data() + call.kc0 * xcb, call.nkc * xcb, call.nx, (int32_t)call.kc0,
                            batch ? NTB : 0);
                di.xbo.sync(XCL_BO_SYNC_BO_TO_DEVICE, (size_t)call.nx * PAIR, 0);
                loaded_kc0 = call.kc0;
                loaded_nkc = call.nkc;
            }
            launch(*dev, call, gi.wbo);
            // The cores keep a running sum over the launch's K chunks, so the last
            // chunk's object of each row tile holds the rows; core r's token t
            // rows sit at (r * NTB + t) * TM in the joined object.
            int64_t obase = call.nx;
            for (int ci : call.chunks) {
                const Chunk &ch = gi.chunks[ci];
                const Matrix &m = mats[ch.member];
                const int64_t Mcol = ch.rows / NCOLS, T = Mcol / (NROWS * TM);
                const bool accumulate = call.kc0 > 0; // later K slices add to the first
                for (int c = 0; c < NCOLS; c++) {
                    const float *yc = ymap + ((int64_t)c * call.ytot + obase) * YOBJ;
                    for (int64_t t = 0; t < T; t++) {
                        const float *res = yc + (t * call.nkc + call.nkc - 1) * YOBJ;
                        for (int r = 0; r < NROWS; r++) {
                            const int64_t rbase = ch.row0 + c * Mcol + (t * NROWS + r) * TM;
                            const int64_t n = std::min<int64_t>(TM, m.M - rbase);
                            if (n <= 0) continue;
                            for (int tt = 0; tt < nb; tt++) {
                                const float *src = res + (r * NTB + tt) * TM;
                                float *dst = gi.out[ch.member].data() + (int64_t)(tb + tt) * m.M + rbase;
                                if (accumulate)
                                    for (int64_t i = 0; i < n; i++) dst[i] += src[i];
                                else
                                    std::memcpy(dst, src, (size_t)n * sizeof(float));
                            }
                        }
                    }
                }
                obase += T * call.nkc;
            }
        }
        dev->stats.host_us += std::chrono::duration<double, std::micro>(clk::now() - t0).count();
    }
}
bool set_power_mode(const char *mode) {
    std::string cmd = std::string("\"C:\\Windows\\System32\\AMD\\xrt-smi.exe\" configure --pmode ") + mode + " >nul 2>&1";
    return std::system(cmd.c_str()) == 0;
}

} // namespace npurun
