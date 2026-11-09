// tessera_cuda: the Tessera v2 pixel-student forward pass on an NVIDIA GPU,
// executed in L2-sized chunks of pixels.
//
// Implements the same C API as tessera_cpu.cpp (tc_create / tc_encode /
// tc_profile_dump / tc_destroy), so tessera_encode.cpp links against it
// unchanged: tile loading, binning, sampling, standardisation and int8
// quantisation stay exactly as on the CPU; only the network runs here.
//
// Each tc_encode batch (<= 4096 pixels sharing one (T2, T1)) is copied to the
// device once, then processed in chunks of TC_CHUNK pixels (default 64):
// every chunk runs both backbones and the fusion head before the next chunk
// starts, so its intermediates stay in L2 instead of streaming through DRAM.
// 64 is best on an L4 (50 MB L2, 300 GB/s GDDR6); an A30 (HBM2) prefers
// 512-1024. GEMMs are cuBLASLt with fused epilogues (bias, bias+ReLU, residual via beta*C);
// attention is strided-batched Q.K^T / softmax / P.V per head; LayerNorm,
// input prep (DOY table), and attention pooling are custom kernels.
//
// Numerics follow the CPU path: bf16 weights/activations, f32 accumulation,
// f32 LayerNorm statistics, the DOY sinusoid from the same f32 table. One
// difference: cuBLASLt requires the bias in bf16 when the output is bf16.
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr int64_t D = 640, NH = 4, HD = 160, FF = 2560, NL = 4, RED = 2560, OUT = 128, KPAD = 16;
constexpr float LN_EPS = 1e-5f;
using bf16 = __nv_bfloat16;

void ck(cudaError_t e, const char *what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void cb(cublasStatus_t s, const char *what) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string(int(s)));
}
#define CK(x) ck((x), #x)
#define CB(x) cb((x), #x)

uint16_t f2bf(float f) {                             // round to nearest even (host)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}

// ---------------------------------------------------------------- kernels

// in (M, CIN+1) f32 -> bands (M, KPAD) bf16 zero-padded; pe (M, 640) bf16 from the DOY table
__global__ void k_prep(const float *__restrict__ in, int64_t M, int CIN, bf16 *__restrict__ bands,
                       bf16 *__restrict__ pe, const bf16 *__restrict__ table, const float *__restrict__ div,
                       const int *__restrict__ chunk) {
    const int64_t r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (r >= M) return;
    const float *row = in + (chunk ? int64_t(*chunk) * M * (CIN + 1) : 0) + r * (CIN + 1);
    if (lane < KPAD) bands[r * KPAD + lane] = __float2bfloat16(lane < CIN ? row[lane] : 0.f);
    const float doy = row[CIN];
    const int di = int(doy);
    bf16 *pr = pe + r * D;
    if (float(di) == doy && di >= 0 && di <= 366) {
        const uint4 *src = reinterpret_cast<const uint4 *>(table + int64_t(di) * D);
        uint4 *dst = reinterpret_cast<uint4 *>(pr);
        for (int i = lane; i < D / 8; i += 32) dst[i] = src[i];
    } else {
        for (int i = lane; i < D / 2; i += 32) {
            const float a = doy * div[i];
            pr[2 * i] = __float2bfloat16(sinf(a));
            pr[2 * i + 1] = __float2bfloat16(cosf(a));
        }
    }
}

// LayerNorm over rows of length N (multiple of 32): one warp per row, f32 two-pass
// statistics. Optional affine, ReLU, and f32 output (of the bf16-rounded value).
template <int N>
__global__ void k_layernorm(const bf16 *__restrict__ x, int64_t M, const float *__restrict__ g,
                            const float *__restrict__ b, bf16 *__restrict__ y, float *__restrict__ yf, bool relu,
                            const int *__restrict__ chunk = nullptr) {
    constexpr int PER = N / 32;
    const int64_t r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (r >= M) return;
    const bf16 *xr = x + r * N;
    float v[PER];
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i) { v[i] = __bfloat162float(xr[lane + 32 * i]); s += v[i]; }
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s / N;
    float q = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i) { const float d = v[i] - mean; q += d * d; }
#pragma unroll
    for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    const float rstd = rsqrtf(q / N + LN_EPS);
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int c = lane + 32 * i;
        float o = (v[i] - mean) * rstd;
        if (g) o = o * g[c] + b[c];
        if (relu) o = fmaxf(o, 0.f);
        const bf16 ob = __float2bfloat16(o);
        if (yf) yf[(chunk ? int64_t(*chunk) * M : 0) * N + r * N + c] = __bfloat162float(ob);
        else y[r * N + c] = ob;
    }
}

__global__ void k_next_chunk(int *chunk) { ++*chunk; }

// Fused attention for one (pixel, head) per block, on tensor cores (WMMA bf16,
// f32 accumulate), for T <= 128. The block stages this (pixel, head)'s K and V
// (T x 160 each, zero beyond T) in shared memory once; each warp keeps its 16
// query rows in registers (ten 16x16 fragments) and then
//   S (16 x Tpad) = Q_tile K^T      (per key tile, via an f32 16x16 staging tile)
//   scale + softmax over the T valid keys in f32, P stored as bf16 (0 beyond T)
//   O (16 x 160) = P V              -> att rows < T (bf16)
// S is kept as bf16 between the two products, as on the CPU path.
__global__ void k_attn(const bf16 *__restrict__ qkv, int T, int Tpad, bf16 *__restrict__ att, float scale) {
    using namespace nvcuda;
    extern __shared__ __align__(32) unsigned char smem[];
    const int nw = blockDim.x / 32, warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int p = blockIdx.x / NH, h = blockIdx.x % NH;
    const int64_t LD = 3 * D;
    bf16 *Ks = reinterpret_cast<bf16 *>(smem);                      // Tpad x 160
    bf16 *Vs = Ks + Tpad * HD;                                       // Tpad x 160
    bf16 *Sw = Vs + Tpad * HD + warp * (16 * Tpad);                  // per warp: 16 x Tpad (S, then P)
    float *tmp = reinterpret_cast<float *>(Vs + Tpad * HD + nw * 16 * Tpad) + warp * 256;
    const bf16 *base = qkv + int64_t(p) * T * LD + h * HD;
    // stage K and V (16-byte vectors: 20 per 160-wide row)
    for (int i = threadIdx.x; i < Tpad * (HD / 8); i += blockDim.x) {
        const int t = i / (HD / 8), v = i % (HD / 8);
        uint4 kk = make_uint4(0, 0, 0, 0), vv = kk;
        if (t < T) {
            kk = reinterpret_cast<const uint4 *>(base + int64_t(t) * LD + D)[v];
            vv = reinterpret_cast<const uint4 *>(base + int64_t(t) * LD + 2 * D)[v];
        }
        reinterpret_cast<uint4 *>(Ks + t * HD)[v] = kk;
        reinterpret_cast<uint4 *>(Vs + t * HD)[v] = vv;
    }
    __syncthreads();
    for (int qt = warp; qt * 16 < T; qt += nw) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> qa[HD / 16];
        const bf16 *qrow = base + int64_t(qt) * 16 * LD;
        for (int dt = 0; dt < HD / 16; ++dt) wmma::load_matrix_sync(qa[dt], qrow + dt * 16, LD);
        for (int kt = 0; kt * 16 < Tpad; ++kt) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.f);
            for (int dt = 0; dt < HD / 16; ++dt) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> kb;
                wmma::load_matrix_sync(kb, Ks + kt * 16 * HD + dt * 16, HD);
                wmma::mma_sync(acc, qa[dt], kb, acc);
            }
            wmma::store_matrix_sync(tmp, acc, 16, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane; i < 256; i += 32) Sw[(i / 16) * Tpad + kt * 16 + i % 16] = __float2bfloat16(tmp[i] * scale);
            __syncwarp();
        }
        // softmax over T valid keys: two lanes per row, f32 math, P in place (0 beyond T)
        const int r = lane >> 1, half = lane & 1;
        bf16 *sr = Sw + r * Tpad;
        float mx = -INFINITY;
        for (int j = half; j < T; j += 2) mx = fmaxf(mx, __bfloat162float(sr[j]));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        float z = 0.f;
        for (int j = half; j < T; j += 2) z += expf(__bfloat162float(sr[j]) - mx);
        z += __shfl_xor_sync(0xffffffffu, z, 1);
        const float inv = 1.f / z;
        for (int j = half; j < Tpad; j += 2)
            sr[j] = __float2bfloat16(j < T ? expf(__bfloat162float(sr[j]) - mx) * inv : 0.f);
        __syncwarp();
        for (int dt = 0; dt < HD / 16; ++dt) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.f);
            for (int kt = 0; kt * 16 < Tpad; ++kt) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> pa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> vb;
                wmma::load_matrix_sync(pa, Sw + kt * 16, Tpad);
                wmma::load_matrix_sync(vb, Vs + kt * 16 * HD + dt * 16, HD);
                wmma::mma_sync(acc, pa, vb, acc);
            }
            wmma::store_matrix_sync(tmp, acc, 16, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane; i < 256; i += 32) {
                const int t = qt * 16 + i / 16;
                if (t < T) att[(int64_t(p) * T + t) * D + h * HD + dt * 16 + i % 16] = __float2bfloat16(tmp[i]);
            }
            __syncwarp();
        }
    }
}

// Row softmax in place over (rows, T) bf16, f32 math; one warp per row.
__global__ void k_softmax(bf16 *__restrict__ s, int64_t rows, int T) {
    const int64_t r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (r >= rows) return;
    bf16 *p = s + r * T;
    float mx = -INFINITY;
    for (int i = lane; i < T; i += 32) mx = fmaxf(mx, __bfloat162float(p[i]));
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float z = 0.f;
    for (int i = lane; i < T; i += 32) z += expf(__bfloat162float(p[i]) - mx);
#pragma unroll
    for (int o = 16; o; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
    const float inv = 1.f / z;
    for (int i = lane; i < T; i += 32) p[i] = __float2bfloat16(expf(__bfloat162float(p[i]) - mx) * inv);
}

// Attention pooling: per pixel, w = softmax_t(x_t . q + b); pooled = sum_t w_t x_t
// -> fused[p, col0 : col0+640] bf16. One block (256 threads) per pixel.
__global__ void k_pool(const bf16 *__restrict__ x, int T, const float *__restrict__ qw, float qb,
                       bf16 *__restrict__ fused, int col0) {
    extern __shared__ float sm[];                   // T scores
    const int p = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    const bf16 *xp = x + int64_t(p) * T * D;
    for (int t = warp; t < T; t += nw) {
        float d = 0.f;
        for (int k = lane; k < D; k += 32) d += __bfloat162float(xp[int64_t(t) * D + k]) * qw[k];
#pragma unroll
        for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        if (lane == 0) sm[t] = d + qb;
    }
    __syncthreads();
    if (warp == 0) {
        float mx = -INFINITY;
        for (int t = lane; t < T; t += 32) mx = fmaxf(mx, sm[t]);
#pragma unroll
        for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        float z = 0.f;
        for (int t = lane; t < T; t += 32) { const float e = expf(sm[t] - mx); sm[t] = e; z += e; }
#pragma unroll
        for (int o = 16; o; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
        const float inv = 1.f / z;
        for (int t = lane; t < T; t += 32) sm[t] *= inv;
    }
    __syncthreads();
    for (int k = tid; k < D; k += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < T; ++t) acc += sm[t] * __bfloat162float(xp[int64_t(t) * D + k]);
        fused[int64_t(p) * 2 * D + col0 + k] = __float2bfloat16(acc);
    }
}

// ---------------------------------------------------------------- host side

struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;
};

std::map<std::string, Tensor> load_weights(const char *path) {
    FILE *f = std::fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string("cannot open ") + path);
    char magic[4];
    uint32_t n;
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "TCW1", 4) || std::fread(&n, 4, 1, f) != 1)
        throw std::runtime_error("bad weights file");
    std::map<std::string, Tensor> w;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t len, nd;
        if (std::fread(&len, 4, 1, f) != 1) throw std::runtime_error("truncated weights");
        std::string name(len, '\0');
        if (std::fread(name.data(), 1, len, f) != len || std::fread(&nd, 4, 1, f) != 1)
            throw std::runtime_error("truncated weights");
        Tensor t;
        t.shape.resize(nd);
        if (std::fread(t.shape.data(), 8, nd, f) != nd) throw std::runtime_error("truncated weights");
        int64_t count = 1;
        for (auto s : t.shape) count *= s;
        t.data.resize(count);
        if (std::fread(t.data.data(), 4, count, f) != size_t(count)) throw std::runtime_error("truncated weights");
        w.emplace(std::move(name), std::move(t));
    }
    std::fclose(f);
    return w;
}

template <class T> T *dalloc(size_t count) {
    void *p = nullptr;
    CK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T)));
    return static_cast<T *>(p);
}

struct Lin {                                        // device weights of one Linear
    int64_t K = 0, N = 0;                           // K padded for the embedding input
    bf16 *w = nullptr;                              // N x K row-major (torch layout)
    bf16 *b = nullptr;                              // N (bf16: cuBLASLt bias type follows D)
};

Lin upload_linear(std::map<std::string, Tensor> &W, const std::string &wn, const std::string &bn, int64_t kpad = 0) {
    Tensor &wt = W.at(wn);
    const int64_t N = wt.shape[0], K0 = wt.shape[1], K = kpad ? kpad : K0;
    std::vector<uint16_t> hw(N * K, 0), hb(N);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K0; ++k) hw[n * K + k] = f2bf(wt.data[n * K0 + k]);
    const std::vector<float> &bias = W.at(bn).data;
    for (int64_t n = 0; n < N; ++n) hb[n] = f2bf(bias[n]);
    Lin l;
    l.K = K;
    l.N = N;
    l.w = dalloc<bf16>(N * K);
    l.b = dalloc<bf16>(N);
    CK(cudaMemcpy(l.w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(l.b, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice));
    return l;
}
Lin upload_linear(std::map<std::string, Tensor> &W, const std::string &name, int64_t kpad = 0) {
    return upload_linear(W, name + ".weight", name + ".bias", kpad);
}

float *upload_vec(std::map<std::string, Tensor> &W, const std::string &name) {
    const auto &v = W.at(name).data;
    float *d = dalloc<float>(v.size());
    CK(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

// A cached cuBLASLt matmul plan: y (M x N) = x (M x K) W^T [+ bias] [relu] [+ beta * C]
struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    cublasLtMatmulAlgo_t algo{};
};

struct Backbone {
    int64_t bands = 0;
    Lin emb1, emb2, qkv[NL], out[NL], ffn1[NL], ffn2[NL];
    float *n1s[NL], *n1b[NL], *n2s[NL], *n2b[NL];
    float *pool_w = nullptr;
    float pool_b = 0.f;
};

struct Ctx {
    int64_t C = 32;                                 // pixels per chunk
    cublasLtHandle_t lt = nullptr;
    cudaStream_t st = nullptr;
    void *ws = nullptr;
    size_t wsz = 32u << 20;
    Backbone bb[2];
    Lin red1, red2;
    float *rln_s = nullptr, *rln_b = nullptr;
    bf16 *pe_table = nullptr;
    float *div = nullptr;
    // chunk scratch, sized for Tmax
    int64_t Tcap = 0, Gcap = 0;
    bf16 *bands = nullptr, *pe = nullptr, *x = nullptr, *x1 = nullptr, *y = nullptr, *att = nullptr;
    bf16 *qkvb = nullptr, *h = nullptr, *sc = nullptr, *fused = nullptr, *r1 = nullptr, *r2 = nullptr;
    float *in2 = nullptr, *in1 = nullptr, *outd = nullptr;
    std::map<std::tuple<int64_t, int64_t, int64_t, int, bool>, Plan> plans;       // (M, K, N, epi, resid)
    std::map<std::tuple<int64_t, int, int>, Plan> attn_plans;                     // (T, which, h)
    // profile
    double gpu_s = 0, h2d_s = 0, d2h_s = 0, enqueue_s = 0;
    uint64_t calls = 0, chunks = 0;
    bool use_graph = true;                          // TC_GRAPH=0: launch directly (needed for TC_PROFILE)
    bool fused_attn = false;                        // TC_ATTN=1: fused WMMA kernel (T <= 128); measured no faster
                                                    // than the per-head batched cuBLASLt + softmax path on the L4
    int *chunk_ctr = nullptr;                       // device chunk counter read by graph kernels
    std::map<std::pair<int64_t, int64_t>, cudaGraphExec_t> graphs;   // (T2, T1)
    bool prof = false;                              // TC_PROFILE=1: CUDA event before every op
    std::vector<cudaEvent_t> ev_pool;
    std::vector<int> ev_cat;                        // category of the op that starts at each event
    size_t ev_n = 0;
    double cat_s[16] = {};
};

enum Cat { C_PREP, C_EMB, C_QKV, C_ATT_S, C_SOFTMAX, C_ATT_PV, C_OUT, C_LN, C_FFN1, C_FFN2, C_POOL, C_HEAD, C_N };
const char *cat_name[C_N] = {"input prep", "embedding GEMMs", "QKV GEMM", "attention (fused / Q.K^T)", "attention softmax",
                             "attention P.V", "out-proj GEMM", "LayerNorm", "FFN up GEMM", "FFN down GEMM",
                             "attention pooling", "fusion head"};

inline void mark(Ctx &c, int cat) {                 // record an event marking the start of an op
    if (!c.prof) return;
    if (c.ev_n == c.ev_pool.size()) {
        cudaEvent_t e;
        CK(cudaEventCreate(&e));
        c.ev_pool.push_back(e);
        c.ev_cat.push_back(0);
    }
    c.ev_cat[c.ev_n] = cat;
    CK(cudaEventRecord(c.ev_pool[c.ev_n++], c.st));
}

void harvest(Ctx &c) {                              // after a sync: attribute event gaps to categories
    if (!c.prof || c.ev_n < 2) { c.ev_n = 0; return; }
    for (size_t i = 0; i + 1 < c.ev_n; ++i) {
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, c.ev_pool[i], c.ev_pool[i + 1]));
        c.cat_s[c.ev_cat[i]] += ms * 1e-3;
    }
    c.ev_n = 0;
}

Plan &gemm_plan(Ctx &c, int64_t M, int64_t K, int64_t N, int epi, bool resid) {
    auto key = std::make_tuple(M, K, N, epi, resid);
    auto it = c.plans.find(key);
    if (it != c.plans.end()) return it->second;
    Plan p;
    CB(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    CB(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    CB(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    cublasLtEpilogue_t e = cublasLtEpilogue_t(epi);
    CB(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &e, sizeof(e)));
    CB(cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, K, N, K));   // W (N x K row-major) = K x N col-major
    CB(cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, K, M, K));   // x^T
    CB(cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_16BF, N, M, N));   // y^T
    cublasLtMatmulPreference_t pref;
    CB(cublasLtMatmulPreferenceCreate(&pref));
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &c.wsz, sizeof(c.wsz)));
    cublasLtMatmulHeuristicResult_t h;
    int found = 0;
    CB(cublasLtMatmulAlgoGetHeuristic(c.lt, p.op, p.la, p.lb, p.lc, p.lc, pref, 1, &h, &found));
    cublasLtMatmulPreferenceDestroy(pref);
    if (!found) throw std::runtime_error("no cuBLASLt algorithm for GEMM shape");
    p.algo = h.algo;
    return c.plans.emplace(key, p).first->second;
}

// y = x W^T + b [relu] [+ res]; x (M x K), y (M x N), all row-major bf16.
void gemm(Ctx &c, const Lin &l, int64_t M, const bf16 *x, bf16 *y, bool relu, const bf16 *res = nullptr) {
    const int epi = relu ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_BIAS;
    Plan &p = gemm_plan(c, M, l.K, l.N, epi, res != nullptr);
    CB(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &l.b, sizeof(l.b)));
    const float alpha = 1.f, beta = res ? 1.f : 0.f;
    CB(cublasLtMatmul(c.lt, p.op, &alpha, l.w, p.la, x, p.lb, &beta, res ? res : y, p.lc, y, p.lc, &p.algo,
                      c.ws, c.wsz, c.st));
}

// Attention for one head over a chunk of P pixels (strided batch over pixels).
//   which 0: S = scale Q K^T -> sc (P, T, T) row-major
//   which 1: O = P V         -> att (P*T, 640) at head offset
Plan &attn_plan(Ctx &c, int64_t T, int which) {
    auto key = std::make_tuple(T, which, 0);
    auto it = c.attn_plans.find(key);
    if (it != c.attn_plans.end()) return it->second;
    const int64_t P = c.C;
    Plan p;
    CB(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    int32_t batch = int32_t(P);
    auto batched = [&](cublasLtMatrixLayout_t l, int64_t stride) {
        CB(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        CB(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
    };
    if (which == 0) {
        // col-major: D (T x T) = op_T(K^T stored 160 x T, ld 1920) * (Q^T stored 160 x T, ld 1920)
        // D[a][b] = k_a . q_b; memory of D equals row-major S[b][a]  (query rows, key cols)
        cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        CB(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
        CB(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
        CB(cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, HD, T, 3 * D));
        CB(cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, HD, T, 3 * D));
        CB(cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_16BF, T, T, T));
        batched(p.la, T * 3 * D);
        batched(p.lb, T * 3 * D);
        batched(p.lc, T * T);
    } else {
        // col-major: O^T (160 x T, ld 640) = V^T (160 x T, ld 1920) * P^T (T x T, ld T)
        cublasOperation_t ta = CUBLAS_OP_N, tb = CUBLAS_OP_N;
        CB(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
        CB(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
        CB(cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, HD, T, 3 * D));
        CB(cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, T, T, T));
        CB(cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_16BF, HD, T, D));
        batched(p.la, T * 3 * D);
        batched(p.lb, T * T);
        batched(p.lc, T * D);
    }
    cublasLtMatmulPreference_t pref;
    CB(cublasLtMatmulPreferenceCreate(&pref));
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &c.wsz, sizeof(c.wsz)));
    cublasLtMatmulHeuristicResult_t h;
    int found = 0;
    CB(cublasLtMatmulAlgoGetHeuristic(c.lt, p.op, p.la, p.lb, p.lc, p.lc, pref, 1, &h, &found));
    cublasLtMatmulPreferenceDestroy(pref);
    if (!found) throw std::runtime_error("no cuBLASLt algorithm for attention GEMM");
    p.algo = h.algo;
    return c.attn_plans.emplace(key, p).first->second;
}

void ensure_scratch(Ctx &c, int64_t T, int64_t Gpad, int64_t T2, int64_t T1) {
    if (T > c.Tcap) {
        const int64_t M = c.C * T;
        for (bf16 **b : {&c.bands, &c.pe, &c.x, &c.x1, &c.y, &c.att, &c.qkvb, &c.h, &c.sc}) CK(cudaFree(*b));
        c.bands = dalloc<bf16>(M * KPAD);
        c.pe = dalloc<bf16>(M * D);
        c.x = dalloc<bf16>(M * D);
        c.x1 = dalloc<bf16>(M * D);
        c.y = dalloc<bf16>(M * D);
        c.att = dalloc<bf16>(M * D);
        c.qkvb = dalloc<bf16>((M + 16) * 3 * D);           // +16 zeroed rows: padded WMMA reads stay finite
        CK(cudaMemset(c.qkvb, 0, (M + 16) * 3 * D * sizeof(bf16)));
        c.h = dalloc<bf16>(M * FF);
        c.sc = dalloc<bf16>(c.C * T * T);
        c.Tcap = T;
        for (auto &g : c.graphs) cudaGraphExecDestroy(g.second);
        c.graphs.clear();
    }
    if (Gpad * std::max(T2 * 11, T1 * 3) > c.Gcap) {
        CK(cudaFree(c.in2)); CK(cudaFree(c.in1)); CK(cudaFree(c.outd));
        c.Gcap = Gpad * std::max(T2 * 11, T1 * 3);
        c.in2 = dalloc<float>(c.Gcap);
        c.in1 = dalloc<float>(c.Gcap);
        c.outd = dalloc<float>(std::max<int64_t>(Gpad, 4096) * OUT);
        for (auto &g : c.graphs) cudaGraphExecDestroy(g.second);
        c.graphs.clear();
    }
}

// One backbone on one chunk: in (C, T, bands+1) f32 on device -> pooled into fused[:, col0:]
void run_backbone(Ctx &c, Backbone &bb, const float *in, int64_t T, int col0, const int *chunk) {
    const int64_t M = c.C * T;
    const int rows_per_block = 8;
    const dim3 blk(32 * rows_per_block), grd(unsigned((M + rows_per_block - 1) / rows_per_block));
    mark(c, C_PREP);
    k_prep<<<grd, blk, 0, c.st>>>(in, M, int(bb.bands), c.bands, c.pe, c.pe_table, c.div, chunk);
    mark(c, C_EMB);
    gemm(c, bb.emb1, M, c.bands, c.y, true);                      // relu(bands W1 + b1)
    gemm(c, bb.emb2, M, c.y, c.x, false, c.pe);                   // y W2 + b2 + pe
    const float scale = 1.f / std::sqrt(float(HD)), one = 1.f, zero = 0.f;
    Plan &ps = attn_plan(c, T, 0), &pv = attn_plan(c, T, 1);
    for (int l = 0; l < NL; ++l) {
        mark(c, C_QKV);
        gemm(c, bb.qkv[l], M, c.x, c.qkvb, false);
        if (c.fused_attn && T <= 128) {
            mark(c, C_ATT_S);
            const int Tpad = int((T + 15) / 16 * 16), nw = Tpad <= 112 ? 4 : 3;
            const size_t sm = size_t(2) * Tpad * HD * 2 + size_t(nw) * (16 * Tpad * 2 + 1024);
            k_attn<<<unsigned(c.C * NH), 32 * nw, sm, c.st>>>(c.qkvb, int(T), Tpad, c.att, scale);
        } else
        for (int hh = 0; hh < NH; ++hh) {
            mark(c, C_ATT_S);
            const bf16 *q = c.qkvb + hh * HD, *k = c.qkvb + D + hh * HD, *v = c.qkvb + 2 * D + hh * HD;
            CB(cublasLtMatmul(c.lt, ps.op, &scale, k, ps.la, q, ps.lb, &zero, c.sc, ps.lc, c.sc, ps.lc, &ps.algo,
                              c.ws, c.wsz, c.st));
            const int64_t rows = c.C * T;
            mark(c, C_SOFTMAX);
            k_softmax<<<unsigned((rows + 7) / 8), 256, 0, c.st>>>(c.sc, rows, int(T));
            mark(c, C_ATT_PV);
            CB(cublasLtMatmul(c.lt, pv.op, &one, v, pv.la, c.sc, pv.lb, &zero, c.att + hh * HD, pv.lc,
                              c.att + hh * HD, pv.lc, &pv.algo, c.ws, c.wsz, c.st));
        }
        mark(c, C_OUT);
        gemm(c, bb.out[l], M, c.att, c.y, false, c.x);           // att Wo + bo + x
        mark(c, C_LN);
        k_layernorm<int(D)><<<grd, blk, 0, c.st>>>(c.y, M, bb.n1s[l], bb.n1b[l], c.x1, nullptr, false);
        mark(c, C_FFN1);
        gemm(c, bb.ffn1[l], M, c.x1, c.h, true);                  // relu(x1 W1 + b1)
        mark(c, C_FFN2);
        gemm(c, bb.ffn2[l], M, c.h, c.y, false, c.x1);            // h W2 + b2 + x1
        mark(c, C_LN);
        k_layernorm<int(D)><<<grd, blk, 0, c.st>>>(c.y, M, bb.n2s[l], bb.n2b[l], c.x, nullptr, false);
    }
    mark(c, C_POOL);
    k_pool<<<unsigned(c.C), 256, T * sizeof(float), c.st>>>(c.x, int(T), bb.pool_w, bb.pool_b, c.fused, col0);
}

void init_backbone(std::map<std::string, Tensor> &W, Backbone &bb, const std::string &prefix, int64_t bands) {
    bb.bands = bands;
    bb.emb1 = upload_linear(W, prefix + ".embedding.0", KPAD);
    bb.emb2 = upload_linear(W, prefix + ".embedding.2");
    for (int l = 0; l < NL; ++l) {
        const std::string L = prefix + ".transformer_encoder.layers." + std::to_string(l);
        bb.qkv[l] = upload_linear(W, L + ".self_attn.in_proj_weight", L + ".self_attn.in_proj_bias");
        bb.out[l] = upload_linear(W, L + ".self_attn.out_proj");
        bb.ffn1[l] = upload_linear(W, L + ".linear1");
        bb.ffn2[l] = upload_linear(W, L + ".linear2");
        bb.n1s[l] = upload_vec(W, L + ".norm1.weight");
        bb.n1b[l] = upload_vec(W, L + ".norm1.bias");
        bb.n2s[l] = upload_vec(W, L + ".norm2.weight");
        bb.n2b[l] = upload_vec(W, L + ".norm2.bias");
    }
    bb.pool_w = upload_vec(W, prefix + ".attn_pool.query.weight");
    bb.pool_b = W.at(prefix + ".attn_pool.query.bias").data[0];
}

}  // namespace

extern "C" {

void *tc_create(const char *weights_path, int /*block_pixels*/) {
    try {
        auto W = load_weights(weights_path);
        auto *c = new Ctx();
        const char *fa = std::getenv("TC_ATTN");
        c->fused_attn = fa && *fa == '1';
        CK(cudaFuncSetAttribute(k_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
        const char *gr = std::getenv("TC_GRAPH");
        c->use_graph = !(gr && *gr == '0');
        c->chunk_ctr = dalloc<int>(1);
        const char *pf = std::getenv("TC_PROFILE");
        c->prof = pf && *pf == '1';
        const char *ch = std::getenv("TC_CHUNK");
        c->C = ch && std::atoi(ch) > 0 ? std::atoi(ch) : 64;
        CB(cublasLtCreate(&c->lt));
        CK(cudaStreamCreate(&c->st));
        CK(cudaMalloc(&c->ws, c->wsz));
        init_backbone(W, c->bb[0], "s2_backbone", 10);
        init_backbone(W, c->bb[1], "s1_backbone", 2);
        c->red1 = upload_linear(W, "dim_reducer.0");
        c->red2 = upload_linear(W, "dim_reducer.4");
        c->rln_s = upload_vec(W, "dim_reducer.1.weight");
        c->rln_b = upload_vec(W, "dim_reducer.1.bias");
        // DOY table: the same f32 formula as the CPU path, rounded to bf16 on the host
        std::vector<float> div(D / 2);
        for (int64_t i = 0; i < D / 2; ++i) div[i] = std::exp(float(2 * i) * -(std::log(10000.0f) / float(D)));
        std::vector<uint16_t> table(367 * D);
        for (int d = 0; d <= 366; ++d)
            for (int64_t i = 0; i < D / 2; ++i) {
                const float a = float(d) * div[i];
                table[d * D + 2 * i] = f2bf(std::sin(a));
                table[d * D + 2 * i + 1] = f2bf(std::cos(a));
            }
        c->pe_table = dalloc<bf16>(table.size());
        CK(cudaMemcpy(c->pe_table, table.data(), table.size() * 2, cudaMemcpyHostToDevice));
        c->div = dalloc<float>(div.size());
        CK(cudaMemcpy(c->div, div.data(), div.size() * 4, cudaMemcpyHostToDevice));
        c->fused = dalloc<bf16>(c->C * 2 * D);
        c->r1 = dalloc<bf16>(c->C * RED);
        c->r2 = dalloc<bf16>(c->C * OUT);
        return c;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "tc_create: %s\n", e.what());
        return nullptr;
    }
}

int tc_encode(void *handle, const float *s2, int64_t G, int64_t T2, const float *s1, int64_t T1, float *out) {
    Ctx &c = *static_cast<Ctx *>(handle);
    try {
        using clk = std::chrono::steady_clock;
        const int64_t C = c.C, nch = (G + C - 1) / C, Gpad = nch * C;
        ensure_scratch(c, std::max(T2, T1), Gpad, T2, T1);
        auto t0 = clk::now();
        CK(cudaMemsetAsync(c.in2, 0, Gpad * T2 * 11 * 4, c.st));
        CK(cudaMemsetAsync(c.in1, 0, Gpad * T1 * 3 * 4, c.st));
        CK(cudaMemcpyAsync(c.in2, s2, G * T2 * 11 * 4, cudaMemcpyHostToDevice, c.st));
        CK(cudaMemcpyAsync(c.in1, s1, G * T1 * 3 * 4, cudaMemcpyHostToDevice, c.st));
        CK(cudaStreamSynchronize(c.st));
        auto t1 = clk::now();
        const dim3 hb(32 * 8), hg(unsigned((C + 7) / 8));
        // one chunk; with `chunk` set, input/output offsets come from the device counter
        auto chunk_fwd = [&](const float *in2, const float *in1, float *outp, const int *chunk) {
            run_backbone(c, c.bb[0], in2, T2, 0, chunk);
            run_backbone(c, c.bb[1], in1, T1, int(D), chunk);
            mark(c, C_HEAD);
            gemm(c, c.red1, C, c.fused, c.r1, false);
            k_layernorm<int(RED)><<<hg, hb, 0, c.st>>>(c.r1, C, c.rln_s, c.rln_b, c.r1, nullptr, true);  // LN + relu
            gemm(c, c.red2, C, c.r1, c.r2, false);
            k_layernorm<int(OUT)><<<hg, hb, 0, c.st>>>(c.r2, C, nullptr, nullptr, nullptr, outp, false, chunk);
        };
        if (c.use_graph && !c.prof) {
            auto key = std::make_pair(T2, T1);
            auto it = c.graphs.find(key);
            if (it == c.graphs.end()) {
                cudaGraph_t g;
                CK(cudaStreamBeginCapture(c.st, cudaStreamCaptureModeThreadLocal));
                chunk_fwd(c.in2, c.in1, c.outd, c.chunk_ctr);
                k_next_chunk<<<1, 1, 0, c.st>>>(c.chunk_ctr);
                CK(cudaStreamEndCapture(c.st, &g));
                cudaGraphExec_t ge;
                CK(cudaGraphInstantiate(&ge, g, 0));
                CK(cudaGraphDestroy(g));
                it = c.graphs.emplace(key, ge).first;
            }
            CK(cudaMemsetAsync(c.chunk_ctr, 0, sizeof(int), c.st));
            for (int64_t ci = 0; ci < nch; ++ci) CK(cudaGraphLaunch(it->second, c.st));
        } else {
            for (int64_t ci = 0; ci < nch; ++ci)
                chunk_fwd(c.in2 + ci * C * T2 * 11, c.in1 + ci * C * T1 * 3, c.outd + ci * C * OUT, nullptr);
        }
        mark(c, C_HEAD);                              // closing event
        auto te = clk::now();
        CK(cudaGetLastError());
        CK(cudaStreamSynchronize(c.st));
        auto t2 = clk::now();
        harvest(c);
        c.enqueue_s += std::chrono::duration<double>(te - t1).count();
        CK(cudaMemcpy(out, c.outd, G * OUT * 4, cudaMemcpyDeviceToHost));
        auto t3 = clk::now();
        c.h2d_s += std::chrono::duration<double>(t1 - t0).count();
        c.gpu_s += std::chrono::duration<double>(t2 - t1).count();
        c.d2h_s += std::chrono::duration<double>(t3 - t2).count();
        ++c.calls;
        c.chunks += nch;
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "tc_encode: %s\n", e.what());
        return -1;
    }
}

void tc_profile_dump(void *handle) {
    Ctx &c = *static_cast<Ctx *>(handle);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("TCPROF cuda device=%s chunk_px=%ld calls=%lu chunks=%lu  h2d=%.2fs  gpu=%.2fs  d2h=%.2fs\n",
                prop.name, (long)c.C, (unsigned long)c.calls, (unsigned long)c.chunks, c.h2d_s, c.gpu_s, c.d2h_s);
    std::printf("TCPROF host enqueue time %.2fs (GPU would idle waiting for launches only if this approached gpu time)\n",
                c.enqueue_s);
    if (c.prof) {
        double t = 0;
        for (int k = 0; k < C_N; ++k) t += c.cat_s[k];
        for (int k = 0; k < C_N; ++k)
            std::printf("TCPROF %-20s %7.2fs %5.1f%%\n", cat_name[k], c.cat_s[k], 100 * c.cat_s[k] / t);
        std::printf("TCPROF %-20s %7.2fs (event-measured GPU time; includes inter-kernel gaps)\n", "total", t);
    }
    std::fflush(stdout);
}

void tc_destroy(void *handle) { delete static_cast<Ctx *>(handle); }

}  // extern "C"
