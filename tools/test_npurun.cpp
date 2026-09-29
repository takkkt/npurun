// test_npurun: correctness + timing of npurun groups against a CPU reference.
//   test_npurun <xclbin> [K] [M0,M1,...] [iters]
#include "npurun.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace npurun;

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = (x >> 13) & 0x3ffu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | (exp << 10) | man);
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ffu, bits;
    if (exp == 0) bits = sign;
    else bits = sign | ((exp + 112) << 23) | (man << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_npurun <xclbin> [K] [M0,M1,..] [iters]\n");
        return 2;
    }
    const int64_t K = argc > 2 ? std::atoll(argv[2]) : 5120;
    std::vector<int64_t> Ms;
    {
        std::stringstream ss(argc > 3 ? argv[3] : "1024,3072");
        std::string tok;
        while (std::getline(ss, tok, ',')) Ms.push_back(std::atoll(tok.c_str()));
    }
    const int iters = argc > 4 ? std::atoi(argv[4]) : 50;
    const int ntok = argc > 5 ? std::atoi(argv[5]) : 1;

    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("opening device\n");
    Device dev(argv[1]);
    std::printf("device ok\n");
    std::vector<std::pair<int64_t, int64_t>> shapes;
    for (auto M : Ms) shapes.push_back({M, K});
    auto t0 = std::chrono::steady_clock::now();
    auto grp = dev.make_group(shapes);
    std::printf("group ok\n");

    std::mt19937 rng(1);
    std::uniform_int_distribution<int> code(0, 2);
    std::uniform_real_distribution<float> sd(0.005f, 0.025f);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int64_t nb = K / QK;
    std::vector<std::vector<BlockPQ2>> W(Ms.size());
    for (size_t j = 0; j < Ms.size(); j++) {
        W[j].resize((size_t)(Ms[j] * nb));
        for (auto &b : W[j]) {
            b.d = f32_to_f16(sd(rng));
            for (auto &q : b.qs) q = (uint8_t)(code(rng) | code(rng) << 2 | code(rng) << 4 | code(rng) << 6);
        }
        grp->pack((int)j, W[j].data(), 0, Ms[j]);
    }
    auto t1 = std::chrono::steady_clock::now();
    std::printf("group of %zu, K=%lld: packed in %.2f s\n", Ms.size(), (long long)K,
                std::chrono::duration<double>(t1 - t0).count());

    std::vector<float> x((size_t)(K * ntok));
    for (auto &v : x) v = nd(rng);
    std::vector<std::vector<float>> y(Ms.size());
    std::vector<float *> yp;
    std::vector<int64_t> ld;
    for (size_t j = 0; j < Ms.size(); j++) {
        y[j].assign((size_t)(Ms[j] * ntok), 0.f);
        yp.push_back(y[j].data());
        ld.push_back(Ms[j]);
    }
    std::printf("running\n");
    grp->run(x.data(), yp.data(), ld.data(), ntok);
    std::printf("ran\n");

    bool ok = true;
    for (size_t j = 0; j < Ms.size(); j++) {
        double worst = 1.0;
        for (int t = 0; t < ntok; t++) {
            const float *xt = x.data() + (size_t)t * K;
            const float *yt = y[j].data() + (size_t)t * Ms[j];
            double dot = 0, n1 = 0, n2 = 0;
            for (int64_t r = 0; r < Ms[j]; r++) {
                double ref = 0;
                for (int64_t b = 0; b < nb; b++) {
                    const BlockPQ2 &blk = W[j][(size_t)(r * nb + b)];
                    const float d = f16_to_f32(blk.d);
                    for (int e = 0; e < QK; e++) {
                        int c = (blk.qs[e / 4] >> (2 * (e % 4))) & 3;
                        ref += (double)(c - 1) * d * xt[(size_t)(b * QK + e)];
                    }
                }
                dot += ref * yt[r];
                n1 += ref * ref;
                n2 += (double)yt[r] * yt[r];
            }
            worst = std::min(worst, dot / std::sqrt(n1 * n2 + 1e-30));
            if (std::getenv("NPURUN_TESTDBG") && j == 0) {
                std::printf("   tok %d y:", t);
                for (int i = 0; i < 6; i++) std::printf(" %.4f", yt[i]);
                std::printf("  |y|=%.3g |ref|=%.3g dot=%.3g\n", std::sqrt(n2), std::sqrt(n1), dot);
            }
        }
        std::printf("  member %zu (%lld x %lld): worst cosine over %d tokens %.6f\n", j, (long long)Ms[j],
                    (long long)K, ntok, worst);
        ok = ok && worst > 0.999;
    }

    dev.stats = {};
    std::vector<double> ts;
    for (int i = 0; i < iters; i++) {
        auto a = std::chrono::steady_clock::now();
        grp->run(x.data(), yp.data(), ld.data(), ntok);
        ts.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count());
    }
    std::sort(ts.begin(), ts.end());
    double bytes = 0;
    for (size_t j = 0; j < Ms.size(); j++) bytes += (double)grp->matrix(j).bytes;
    std::printf("  per call (%d tok): min %.0f us, median %.0f us = %.0f us/token (%llu launches/call) -> %.1f GB/s at median;"
                " npu %.0f us/launch, host %.0f us/call\n",
                ntok, ts.front(), ts[ts.size() / 2], ts[ts.size() / 2] / ntok, (unsigned long long)(dev.stats.launches / iters),
                bytes / (ts[ts.size() / 2] * 1e3), dev.stats.npu_us / dev.stats.launches,
                (dev.stats.host_us - dev.stats.npu_us) / iters);
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
