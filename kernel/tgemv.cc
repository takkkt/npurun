// tgemv.cc - shape-agnostic ternary (2-bit coded) GEMV step for AIE2P (XDNA2).
//
// The core loops forever over stream objects. Every object is a 64-byte
// header followed by TG_TILE payload bytes, and yields one TG_M-float output:
//   kind 1 (XLOAD): payload = hdr[3] activation bytes (multiple of 64) copied
//                   to xbuf at byte offset hdr[2]; the output is unused.
//   kind 2 (TILE) : payload = one weight tile of TG_M rows x TG_K columns for
//                   K chunk hdr[1]; the output is that chunk's partial dot
//                   product, which the host sums over chunks.
// The matrix shape therefore lives only in the host's DMA instruction stream,
// and one NPU program serves every weight matrix of a model.
//
// The inner product runs on the native 8x8x8 int8 matrix unit:
//   A (8x8, int8)  : row n holds the 8 activations of the current k-chunk when
//                    that chunk belongs to group slot n (other rows zero)
//   B (8x8, uint8) : B[kk][r] = 2-bit code of row r at column k+kk (0..3)
//   C (8x8, acc32) : after one 1024-column super-group, C[n][r] = sum over
//                    group n of code * x, i.e. 8 rows x 8 groups of dots.
// The ternary weight is code-1, so the true dot is C[n][r] - sum(x in group n);
// the per-group sums come precomputed with the activations.
//
// Weight tile layout (bytes), produced by npurun/pack.py; for rb in TG_M/8 row
// blocks, for sg in TG_K/1024 super-groups:
//   TG_FMT 3 (default, 1.75 bits/weight):
//     qs[26][64] : word v holds k-chunks 5v..5v+4 (the last word only 125..127);
//                  byte kk*8+r = sum_t code(row r, column 8*(5v+t)+kk) * 3^t
//     sc[64]     : fp16 scale, lane n*8+r = scale of (row r, group n)
//   TG_FMT 2 (2.25 bits/weight):
//     qs[32][64] : word q holds k-chunks 4q..4q+3; byte kk*8+r holds the code
//                  of (row r, column 8*(4q+t)+kk) in bits [2t, 2t+2)
//     sc[64]     : float32 scale, lane n*8+r = scale of (row r, group n)
// Activation layout in xbuf (per K chunk):
//   xq[TG_K] int8 | sx[TG_K/128] float32 | sumx[TG_K/128] int32

#define NOCPP
#include <stdint.h>

#include <aie_api/aie.hpp>

#ifndef TG_M
#define TG_M 32
#endif
#ifndef TG_K
#define TG_K 1024
#endif
#ifndef TG_KMAX
#define TG_KMAX 17408
#endif
#ifndef TG_FMT
#define TG_FMT 3
#endif
#if TG_FMT == 3
#define TG_SG_BYTES (26 * 64 + 64 * 2)
#else
#define TG_SG_BYTES (32 * 64 + 64 * 4)
#endif
#ifndef TG_TILE
#define TG_TILE ((TG_M / 8) * (TG_K / 1024) * TG_SG_BYTES)
#endif

static_assert(TG_M % 8 == 0, "TG_M must be a multiple of 8");
static_assert(TG_K % 1024 == 0, "TG_K must be a multiple of 1024");
static_assert(TG_KMAX % TG_K == 0, "TG_KMAX must be a multiple of TG_K");

constexpr int SG = TG_K / 1024;                          // super-groups per chunk
constexpr int NG = TG_K / 128;                           // groups per chunk
constexpr int XCHUNK = TG_K + NG * 8;                    // activation bytes per chunk
constexpr int XBYTES = (TG_KMAX / TG_K) * XCHUNK;        // max resident activations
constexpr int XBUF = (XBYTES + 63) / 64 * 64;
constexpr int NT = 4;                                    // batch path: tokens per tile
constexpr int XCHB = TG_K * NT + NG * NT * 8;            // batch activation bytes per chunk
constexpr int KBMAX = 6144;                              // widest K one batch launch covers
constexpr int XBUF_ALL = XBUF > (KBMAX / TG_K) * XCHB ? XBUF : (KBMAX / TG_K) * XCHB;
constexpr int SG_BYTES = TG_SG_BYTES;                    // codes + scales per (rb, sg)
constexpr int HDR = 64;

enum { KIND_XLOAD = 1, KIND_TILE = 2 };

using MMUL = aie::mmul<8, 8, 8, int8_t, uint8_t>;

alignas(64) static int8_t xbuf[XBUF_ALL];
// Per super-group lane vectors for the current chunk (static: the core stack
// does not guarantee the 64-byte alignment vector loads need).
alignas(64) static float sxv[SG][64];
alignas(64) static int32_t corr[SG][64];
// A-operand table of the single-token path; the batch path reuses it to hold
// one super-group of decoded digit matrices (130 incl. the padding chunks).
alignas(64) static int8_t atab[TG_K / 8 + 2][64];
static bool atab_ready = false;

// Running sum over the K chunks of the current row tile: a TILE object with
// hdr[1] == 0 starts a new sum, and every TILE object outputs the sum so far,
// so the object of the last chunk carries the full dot products.
alignas(64) static float yrun[TG_M];

static void tile_mv(const uint8_t *__restrict w, const int8_t *__restrict x,
                    float *__restrict y, bool first) {
  const float *__restrict sx = (const float *)(x + TG_K);
  const int32_t *__restrict sumx = (const int32_t *)(sx + NG);

  // Per super-group lane vectors: lane n*8+r -> value of group n. Built with
  // vector loads + broadcasts (a scalar fill loop here was miscompiled).
  for (int s = 0; s < SG; s++) {
    const aie::vector<float, 8> s8 = aie::load_v<8>(sx + s * 8);
    const aie::vector<int32_t, 8> c8 = aie::load_v<8>(sumx + s * 8);
    for (int n = 0; n < 8; n++) {
      aie::store_v(&sxv[s][n * 8], aie::broadcast<float, 8>(s8[n]));
      aie::store_v(&corr[s][n * 8], aie::broadcast<int32_t, 8>(c8[n]));
    }
  }

  // A-operand table: chunk c (8 activations) sits in row (c/16)%8 of its own
  // 8x8 matrix. Only those 8 bytes change, so the zeros are written once.
  if (!atab_ready) {
    for (int c = 0; c < TG_K / 8; c++)
      aie::store_v(atab[c], aie::zeros<int8_t, 64>());
    atab_ready = true;
  }
  {
    const int32_t *__restrict x32 = (const int32_t *)x;
    for (int c = 0; c < TG_K / 8; c++) {
      int32_t *dst = (int32_t *)(atab[c] + ((c >> 4) & 7) * 8);
      dst[0] = x32[2 * c];
      dst[1] = x32[2 * c + 1];
    }
  }

#if TG_FMT == 3
  aie::set_rounding(aie::rounding_mode::floor); // x*171 >> 9 == x/3 needs truncation
#else
  const aie::vector<uint8_t, 64> m3 = aie::broadcast<uint8_t, 64>(3);
#endif

  for (int rb = 0; rb < TG_M / 8; rb++) {
    aie::accum<accfloat, 64> yacc = aie::zeros<accfloat, 64>();
    for (int s = 0; s < SG; s++) {
      const uint8_t *__restrict qs = w + (rb * SG + s) * SG_BYTES;
      const int8_t(*__restrict as)[64] = atab + s * 128;
      MMUL acc;
      acc.mul(aie::zeros<int8_t, 64>(), aie::zeros<uint8_t, 64>());
#if TG_FMT == 3

      // word v: base-3 digits t = 0..4 are the codes of k-chunks 5v+t
      // Digits of d (< 243) without a dependency chain: every quotient d / 3^t
      // comes straight from d via an exact multiply-shift on the 8-bit
      // multiplier, and digit t = q_t - 3 q_{t+1}.
      auto quot = [](const aie::vector<uint8_t, 64> &d, uint8_t m, int sh) {
        return aie::mul(d, m).template to_vector<uint8_t>(sh);
      };
      auto times3 = [](const aie::vector<uint8_t, 64> &q) {
        return aie::mul(q, (uint8_t)3).template to_vector<uint8_t>(0);
      };
      const int8_t *__restrict ap = as[0];   // A operands, walked sequentially
      const uint8_t *__restrict wp = qs;     // code words
#pragma clang loop unroll_count(5)
      for (int v = 0; v < 25; v++) {
        const aie::vector<uint8_t, 64> d = aie::load_v<64>(wp);
        wp += 64;
        const aie::vector<uint8_t, 64> q1 = quot(d, 171, 9), q2 = quot(d, 57, 9);
        const aie::vector<uint8_t, 64> q3 = quot(d, 19, 9), q4 = quot(d, 203, 14);
        const aie::vector<int8_t, 64> a0 = aie::load_v<64>(ap), a1 = aie::load_v<64>(ap + 64),
                                      a2 = aie::load_v<64>(ap + 128), a3 = aie::load_v<64>(ap + 192),
                                      a4 = aie::load_v<64>(ap + 256);
        ap += 320;
        acc.mac(a0, aie::sub(d, times3(q1)));
        acc.mac(a1, aie::sub(q1, times3(q2)));
        acc.mac(a2, aie::sub(q2, times3(q3)));
        acc.mac(a3, aie::sub(q3, times3(q4)));
        acc.mac(a4, q4);
      }
      {
        const aie::vector<uint8_t, 64> d = aie::load_v<64>(wp);
        const aie::vector<uint8_t, 64> q1 = quot(d, 171, 9), q2 = quot(d, 57, 9), q3 = quot(d, 19, 9);
        const aie::vector<int8_t, 64> a0 = aie::load_v<64>(ap), a1 = aie::load_v<64>(ap + 64),
                                      a2 = aie::load_v<64>(ap + 128);
        acc.mac(a0, aie::sub(d, times3(q1)));
        acc.mac(a1, aie::sub(q1, times3(q2)));
        acc.mac(a2, aie::sub(q2, times3(q3)));
      }
      // fp16 -> fp32 scales (normal values; zero maps to a tiny value that only
      // ever multiplies zero-weight padding rows)
      alignas(64) static uint32_t scbits[64];
      for (int i = 0; i < 4; i++) {
        aie::vector<uint16_t, 16> h16 = aie::load_v<16>((const uint16_t *)(qs + 26 * 64) + 16 * i);
        aie::vector<uint32_t, 16> h = aie::unpack(h16);
        aie::vector<uint32_t, 16> f = aie::bit_or(
            aie::upshift(aie::bit_and((uint32_t)0x8000u, h), 16),
            aie::add(aie::upshift(aie::bit_and((uint32_t)0x7fffu, h), 13), (uint32_t)0x38000000u));
        aie::store_v(scbits + 16 * i, f);
      }
      const aie::vector<float, 64> scw = aie::load_v<64>((const float *)scbits);
#else
      for (int q = 0; q < 32; q++) {
        // quad q: k-chunks 4q..4q+3 (group slot q/4)
        aie::vector<uint8_t, 64> v = aie::load_v<64>(qs + 64 * q);
        aie::vector<uint8_t, 64> b0 = aie::bit_and(v, m3);
        aie::vector<uint8_t, 64> b1 = aie::bit_and(aie::logical_downshift(v, 2), m3);
        aie::vector<uint8_t, 64> b2 = aie::bit_and(aie::logical_downshift(v, 4), m3);
        aie::vector<uint8_t, 64> b3 = aie::logical_downshift(v, 6);
        acc.mac(aie::load_v<64>(as[4 * q]), b0);
        acc.mac(aie::load_v<64>(as[4 * q + 1]), b1);
        acc.mac(aie::load_v<64>(as[4 * q + 2]), b2);
        acc.mac(aie::load_v<64>(as[4 * q + 3]), b3);
      }
      const aie::vector<float, 64> scw = aie::load_v<64>((const float *)(qs + 32 * 64));
#endif
      aie::vector<int32_t, 64> c = acc.to_vector<int32_t>();

      c = aie::sub(c, aie::load_v<64>(corr[s]));
      aie::vector<float, 64> cf = aie::to_float(c);
      aie::vector<float, 64> scale = aie::mul(scw, aie::load_v<64>(sxv[s])).to_vector<float>();
      yacc = aie::mac(yacc, cf, scale);
    }
    // y[r] = sum_n yacc[n*8 + r]
    aie::vector<float, 64> yv = yacc.to_vector<float>();
    aie::vector<float, 8> ysum = yv.extract<8>(0);
    for (int n = 1; n < 8; n++)
      ysum = aie::add(ysum, yv.extract<8>(n));
    if (!first) ysum = aie::add(ysum, aie::load_v<8>(yrun + rb * 8));
    aie::store_v(yrun + rb * 8, ysum);
    aie::store_v(y + rb * 8, ysum);
  }
}

// ---------------------------------------------------------------- batch path
//
// Launches whose XLOAD objects carry hdr[5] == NT use the 4x8x8 matrix unit with the tokens as
// A rows: A (4 tok x 8 k) . B (8 k x 8 rows) -> C (4 tok x 8 rows). One decode
// of the weight codes serves all NT tokens. Accumulation restarts per 128-column
// group (every 16 chunks) because scales are per (row, group) and per (token,
// group). Batch activation chunk (per 1024 columns, XCHB bytes):
//   xa[128 chunks][NT tok][8] int8 | per group g: sx[NT] f32, sumx[NT] i32
// Outputs: y[t * TG_M + row] for t < NT.

static_assert(SG == 1 && TG_FMT == 3, "batch path assumes TG_K == 1024 and base-3 codes");

using MMUL4 = aie::mmul<4, 8, 8, int8_t, uint8_t>;

alignas(64) static float yrunb[NT * TG_M];

static void tile_mv_batch(const uint8_t *__restrict w, const int8_t *__restrict x,
                          float *__restrict y, bool first) {
  aie::set_rounding(aie::rounding_mode::floor);
  atab_ready = false; // the digit buffer overwrites the single-token table
  uint8_t(*__restrict dig)[64] = (uint8_t(*)[64])atab;
  const float *__restrict gpar = (const float *)(x + TG_K * NT); // per group: sx[NT], sumx[NT]
  // lane t*8+r vectors per group, shared by every row block of this tile
  alignas(64) static int32_t corrb[8][32];
  alignas(64) static float sxb[8][32];
  for (int g = 0; g < 8; g++) {
    const float *__restrict gp = gpar + g * NT * 2;
    const int32_t *__restrict sm = (const int32_t *)(gp + NT);
    for (int t = 0; t < NT; t++) {
      aie::store_v(&corrb[g][t * 8], aie::broadcast<int32_t, 8>(sm[t]));
      aie::store_v(&sxb[g][t * 8], aie::broadcast<float, 8>(gp[t]));
    }
  }
  auto quot = [](const aie::vector<uint8_t, 64> &d, uint8_t m, int sh) {
    return aie::mul(d, m).template to_vector<uint8_t>(sh);
  };
  auto times3 = [](const aie::vector<uint8_t, 64> &q) {
    return aie::mul(q, (uint8_t)3).template to_vector<uint8_t>(0);
  };
  for (int rb = 0; rb < TG_M / 8; rb++) {
    const uint8_t *__restrict qs = w + rb * SG_BYTES;
    // decode all 128 chunks' digit matrices once
    for (int v = 0; v < 26; v++) {
      const aie::vector<uint8_t, 64> d = aie::load_v<64>(qs + 64 * v);
      const aie::vector<uint8_t, 64> q1 = quot(d, 171, 9), q2 = quot(d, 57, 9);
      const aie::vector<uint8_t, 64> q3 = quot(d, 19, 9), q4 = quot(d, 203, 14);
      aie::store_v(dig[5 * v + 0], aie::sub(d, times3(q1)));
      aie::store_v(dig[5 * v + 1], aie::sub(q1, times3(q2)));
      aie::store_v(dig[5 * v + 2], aie::sub(q2, times3(q3)));
      if (v < 25) {
        aie::store_v(dig[5 * v + 3], aie::sub(q3, times3(q4)));
        aie::store_v(dig[5 * v + 4], q4);
      }
    }
    alignas(64) static uint32_t scbits[64];
    for (int i = 0; i < 4; i++) {
      aie::vector<uint32_t, 16> h = aie::unpack(aie::load_v<16>((const uint16_t *)(qs + 26 * 64) + 16 * i));
      aie::vector<uint32_t, 16> f = aie::bit_or(
          aie::upshift(aie::bit_and((uint32_t)0x8000u, h), 16),
          aie::add(aie::upshift(aie::bit_and((uint32_t)0x7fffu, h), 13), (uint32_t)0x38000000u));
      aie::store_v(scbits + 16 * i, f);
    }
    const float *__restrict scw = (const float *)scbits; // lane n*8+r

    aie::accum<accfloat, 32> yacc = aie::zeros<accfloat, 32>();
    const int8_t *__restrict xa = x;
    for (int g = 0; g < 8; g++) {
      MMUL4 acc;
      acc.mul(aie::load_v<32>(xa), aie::load_v<64>(dig[16 * g]));
      xa += 32;
      for (int c = 1; c < 16; c++) {
        acc.mac(aie::load_v<32>(xa), aie::load_v<64>(dig[16 * g + c]));
        xa += 32;
      }
      aie::vector<int32_t, 32> cv = aie::sub(acc.to_vector<int32_t>(), aie::load_v<32>(corrb[g])); // lane t*8+r
      const aie::vector<float, 8> d8 = aie::load_v<8>(scw + g * 8);
      const aie::vector<float, 32> sv =
          aie::mul(aie::concat(d8, d8, d8, d8), aie::load_v<32>(sxb[g])).template to_vector<float>();
      yacc = aie::mac(yacc, aie::to_float(cv), sv);
    }
    aie::vector<float, 32> yv = yacc.to_vector<float>();
    for (int t = 0; t < NT; t++) {
      aie::vector<float, 8> r8 = yv.extract<8>(t);
      float *yr = yrunb + t * TG_M + rb * 8;
      if (!first) r8 = aie::add(r8, aie::load_v<8>(yr));
      aie::store_v(yr, r8);
      aie::store_v(y + t * TG_M + rb * 8, r8);
    }
  }
}

static int kc_base = 0; // first K chunk of the current launch (K-sliced batch launches)
static int bmode = 0;   // NT: the launch runs the batch path (set by its XLOAD objects)

extern "C" {

// obj: HDR + TG_TILE bytes; y: NT * TG_M floats (single-token tiles use the first TG_M)
void tg_step(const uint8_t *__restrict obj, float *__restrict y) {
  event0();
  const int32_t *hdr = (const int32_t *)obj;
  const uint8_t *payload = obj + HDR;
  if (hdr[0] == KIND_XLOAD) {
    int8_t *__restrict dst = xbuf + hdr[2];
    for (int i = 0; i < hdr[3]; i += 64)
      aie::store_v(dst + i, aie::load_v<64>((const int8_t *)payload + i));
    kc_base = hdr[4];
    bmode = hdr[5];
  } else {
    const int kc = hdr[1] - kc_base;
#ifdef TG_NOP
    y[0] = (float)payload[0];
#else
    if (bmode == NT)
      tile_mv_batch(payload, xbuf + kc * XCHB, y, kc == 0);
    else
      tile_mv(payload, xbuf + kc * XCHUNK, y, kc == 0);
#endif
  }
  event1();
}

} // extern "C"