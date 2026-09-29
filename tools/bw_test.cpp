// bw_test: CPU read bandwidth alone and while the NPU streams a large matrix.
//   bw_test <xclbin> [threads]
#include "npurun.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <thread>
#include <vector>

static double cpu_read(const uint8_t *buf, size_t bytes, int nthreads, int reps) {
    std::atomic<uint64_t> sink{0};
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    for (int t = 0; t < nthreads; t++)
        ts.emplace_back([&, t] {
            const size_t chunk = bytes / nthreads;
            const __m256i *p = (const __m256i *)(buf + chunk * t);
            __m256i acc = _mm256_setzero_si256();
            for (int r = 0; r < reps; r++)
                for (size_t i = 0; i < chunk / 32; i += 4) {
                    acc = _mm256_xor_si256(acc, _mm256_load_si256(p + i));
                    acc = _mm256_xor_si256(acc, _mm256_load_si256(p + i + 1));
                    acc = _mm256_xor_si256(acc, _mm256_load_si256(p + i + 2));
                    acc = _mm256_xor_si256(acc, _mm256_load_si256(p + i + 3));
                }
            sink += (uint64_t)_mm256_extract_epi64(acc, 0);
        });
    for (auto &t : ts) t.join();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return (double)bytes * reps / s / 1e9;
}

int main(int argc, char **argv) {
    const int nth = argc > 2 ? std::atoi(argv[2]) : 12;
    const size_t bytes = (size_t)1 << 30;
    uint8_t *buf = (uint8_t *)_aligned_malloc(bytes, 64);
    std::memset(buf, 1, bytes);
    std::printf("CPU alone (%d threads): %.1f GB/s\n", nth, cpu_read(buf, bytes, nth, 4));
    for (int n : {4, 8, 16}) std::printf("CPU alone (%d threads): %.1f GB/s\n", n, cpu_read(buf, bytes, n, 4));

    npurun::Device dev(argv[1]);
    const int64_t M = 34816, K = 5120; // ~50 MB of packed weights
    auto grp = dev.make_group({{M, K}});
    std::vector<npurun::BlockPQ2> w((size_t)(M * (K / 128)));
    for (auto &b : w) {
        b.d = 0x2000;
        std::memset(b.qs, 0x55, sizeof(b.qs));
    }
    grp->pack(0, w.data(), 0, M);
    std::vector<float> x((size_t)K, 0.5f), y((size_t)M);
    float *yp = y.data();
    int64_t ld = M;
    grp->run(x.data(), &yp, &ld, 1);
    auto npu_rate = [&](int n) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; i++) grp->run(x.data(), &yp, &ld, 1);
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return (double)grp->matrix(0).bytes * n / s / 1e9;
    };
    std::printf("NPU alone: %.1f GB/s\n", npu_rate(40));
    std::atomic<bool> stop{false};
    std::atomic<int> nruns{0};
    std::thread npu([&] {
        while (!stop) {
            grp->run(x.data(), &yp, &ld, 1);
            nruns++;
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    auto t0 = std::chrono::steady_clock::now();
    int r0 = nruns;
    double cpu = cpu_read(buf, bytes, nth, 4);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    int r1 = nruns;
    stop = true;
    npu.join();
    double npu_bw = (double)grp->matrix(0).bytes * (r1 - r0) / s / 1e9;
    std::printf("together: CPU %.1f GB/s + NPU %.1f GB/s = %.1f GB/s\n", cpu, npu_bw, cpu + npu_bw);
    return 0;
}
