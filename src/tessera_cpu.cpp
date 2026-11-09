// tessera_cpu: the Tessera v2 pixel-student forward pass on CPU,
// thread-per-block.
//
// Instead of many threads cooperating on every operation of a block (a
// barrier after each of ~80 ops), each thread takes its own small block of
// pixels through the whole model - both backbones and the fusion head - with single-threaded oneDNN
// primitives (oneDNN built with ONEDNN_CPU_RUNTIME=SEQ). Pixels are
// independent, so threads never synchronise inside a batch; a thread's
// working set (~2-4 MB at 2 pixels x 112 steps) stays in its core's L2 and
// L3 slice.
//
// Threading: one `omp parallel for` over blocks per tc_encode call. Every
// primitive uses a user-provided scratchpad so the shared primitives can run
// concurrently; scratch buffers and streams are per thread.
#include <dnnl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <chrono>
#include <vector>

#include <omp.h>

using namespace dnnl;
using dt = memory::data_type;
using tag = memory::format_tag;
using dims = memory::dims;

namespace {

constexpr int64_t D = 640, NH = 4, HD = 160, FF = 2560, NL = 4, RED = 2560, OUT = 128;
constexpr float LN_EPS = 1e-5f;

inline uint16_t f2bf(float f) {                     // round to nearest even, like torch
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}
inline float bf2f(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// 64 KB-aligned growable buffer (cache-line and page aligned; small enough
// per thread that 2 MB alignment would only waste memory).
struct Buf {
    void *p = nullptr;
    size_t bytes = 0;
    void reserve(size_t want) {
        if (want <= bytes) return;
        std::free(p);
        bytes = ((want + 65535) / 65536) * 65536;
        p = std::aligned_alloc(65536, bytes);
        if (!p) throw std::runtime_error("aligned_alloc failed");
        std::memset(p, 0, bytes);                   // first touch by the owning thread
    }
    template <class T> T *as() { return static_cast<T *>(p); }
    Buf() = default;
    Buf(const Buf &) = delete;
    Buf &operator=(const Buf &) = delete;
    ~Buf() { std::free(p); }
};

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

struct LinearWeights {                // bf16, torch layout (out x in), packed on demand
    int64_t K = 0, N = 0;
    std::vector<uint16_t> w;
    std::vector<float> bias;
    std::vector<std::pair<memory::desc, memory>> packed;
};

struct Linear {                       // y = x W^T + b [relu] [+ residual]
    matmul prim;
    memory w, b;
    memory::desc spad;
    bool residual = false;
    int res_pos = 0;
};

struct LayerNorm {
    layer_normalization_forward prim;
    memory scale, shift;
    memory::desc spad;
    bool affine = true;
};

struct BackboneT {                    // everything that depends on the sequence length T
    int64_t P = 0, T = 0, M = 0;      // P = pixels per block for this backbone
    Linear emb1, emb2;
    Linear qkv[NL], out[NL], ffn1[NL], ffn2[NL];
    LayerNorm ln1[NL], ln2[NL];
    matmul qk, sv;
    softmax_forward sm;
    memory::desc q_md, kT_md, s_md, v_md, o_md, qk_spad, sm_spad, sv_spad;
};

struct Backbone {
    int64_t bands = 0;
    int64_t P = 0;                    // pixels per block (S2: TC_BLOCK_S2, S1: TC_BLOCK)
    LinearWeights emb1, emb2, qkv[NL], out[NL], ffn1[NL], ffn2[NL];
    std::vector<float> n1s[NL], n1b[NL], n2s[NL], n2b[NL];
    std::vector<float> pool_w;
    float pool_b = 0.f;
    std::map<int64_t, std::unique_ptr<BackboneT>> byT;
};

// ---- optional profiling (TC_PROFILE=1): per-thread time by phase ----
enum Phase { PH_PREP, PH_ARGS, PH_EMB, PH_QKV, PH_ATT, PH_OUT, PH_LN, PH_FFN1, PH_FFN2, PH_POOL, PH_N };
const char *phase_name[PH_N] = {"input prep + DOY", "oneDNN arg setup", "embedding GEMMs", "QKV GEMM",
                                "attention", "out-proj GEMM", "LayerNorm", "FFN up GEMM", "FFN down GEMM",
                                "attention pooling"};
enum Misc { MS_HEAD, MS_PAD, MS_N };
const char *misc_name[MS_N] = {"fusion head", "pad copy"};
const bool g_profile = [] { const char *e = std::getenv("TC_PROFILE"); return e && *e == '1'; }();
using pclock = std::chrono::steady_clock;
inline double secs(pclock::time_point a, pclock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

struct Prof {
    double ph[2][PH_N] = {};          // [backbone][phase] seconds
    double misc[MS_N] = {};
    double busy = 0;                  // seconds spent inside blocks
    uint64_t blocks = 0;
};

struct Scratch {                      // one per thread
    Buf bands, pe, x, x1, y, y2, qkvb, att, hbuf, scores, fused, r1, r2, pad, spad;
    std::unique_ptr<stream> strm;
    Prof prof;
    int cur_bb = 0;                   // backbone being run, for attributing phases
};

struct Ctx {
    engine eng{engine::kind::cpu, 0};
    stream setup_strm{eng};
    int64_t P;                        // pixels per thread block (= S2 block; S1 runs P / P1 sub-blocks)
    int64_t P1;                       // S1 sub-block pixels
    int nthr;
    Backbone bb[2];                   // 0 = S2, 1 = S1
    LinearWeights red1, red2;
    std::vector<float> rln_s, rln_b;
    Linear red1L, red2L;
    LayerNorm rln, oln;
    size_t spad_bytes = 0;            // largest scratchpad any primitive needs
    std::vector<uint16_t> pe_table;   // 367 x 640: DOY 0..366
    std::vector<std::unique_ptr<Scratch>> scratch;
    double region_wall = 0;           // sum over calls of parallel-region wall time
    double call_wall = 0;             // sum over calls of tc_encode wall time
    uint64_t calls = 0;
};

primitive_attr user_scratch_attr(const post_ops &po = post_ops()) {
    primitive_attr a;
    a.set_post_ops(po);
    a.set_scratchpad_mode(scratchpad_mode::user);
    return a;
}

LinearWeights take_linear(std::map<std::string, Tensor> &W, const std::string &wname, const std::string &bname) {
    Tensor &wt = W.at(wname);
    LinearWeights lw;
    lw.N = wt.shape[0];
    lw.K = wt.shape[1];
    lw.w.resize(wt.data.size());
    for (size_t i = 0; i < wt.data.size(); ++i) lw.w[i] = f2bf(wt.data[i]);
    lw.bias = W.at(bname).data;
    return lw;
}
LinearWeights take_linear(std::map<std::string, Tensor> &W, const std::string &name) {
    return take_linear(W, name + ".weight", name + ".bias");
}

Linear make_linear(Ctx &c, LinearWeights &lw, int64_t M, bool relu, bool residual) {
    memory::desc src({M, lw.K}, dt::bf16, tag::ab);
    memory::desc wany({lw.K, lw.N}, dt::bf16, tag::any);
    memory::desc bmd({1, lw.N}, dt::f32, tag::ab);
    memory::desc dst({M, lw.N}, dt::bf16, tag::ab);
    post_ops po;
    if (relu) po.append_eltwise(algorithm::eltwise_relu, 0.f, 0.f);
    if (residual) po.append_binary(algorithm::binary_add, dst);
    matmul::primitive_desc pd(c.eng, src, wany, bmd, dst, user_scratch_attr(po));
    c.spad_bytes = std::max(c.spad_bytes, pd.scratchpad_desc().get_size());
    Linear L;
    L.prim = matmul(pd);
    L.spad = pd.scratchpad_desc();
    L.residual = residual;
    L.res_pos = relu ? 1 : 0;
    for (auto &[md, mem] : lw.packed)
        if (md == pd.weights_desc()) { L.w = mem; break; }
    if (!L.w) {
        memory wp(memory::desc({lw.K, lw.N}, dt::bf16, tag::ba), c.eng, lw.w.data());
        memory wpk(pd.weights_desc(), c.eng);
        reorder(wp, wpk).execute(c.setup_strm, wp, wpk);
        c.setup_strm.wait();
        lw.packed.emplace_back(pd.weights_desc(), wpk);
        L.w = wpk;
    }
    L.b = memory(bmd, c.eng);
    std::memcpy(L.b.get_data_handle(), lw.bias.data(), lw.N * sizeof(float));
    return L;
}

LayerNorm make_ln(Ctx &c, int64_t M, int64_t C, const std::vector<float> *s, const std::vector<float> *b) {
    memory::desc md({M, C}, dt::bf16, tag::ab);
    LayerNorm ln;
    ln.affine = s != nullptr;
    auto flags = ln.affine ? (normalization_flags::use_scale | normalization_flags::use_shift)
                           : normalization_flags::none;
    layer_normalization_forward::primitive_desc pd(c.eng, prop_kind::forward_inference, md, md, LN_EPS,
                                                    flags, user_scratch_attr());
    c.spad_bytes = std::max(c.spad_bytes, pd.scratchpad_desc().get_size());
    ln.prim = layer_normalization_forward(pd);
    ln.spad = pd.scratchpad_desc();
    if (ln.affine) {
        ln.scale = memory({{C}, dt::f32, tag::a}, c.eng);
        ln.shift = memory({{C}, dt::f32, tag::a}, c.eng);
        std::memcpy(ln.scale.get_data_handle(), s->data(), C * sizeof(float));
        std::memcpy(ln.shift.get_data_handle(), b->data(), C * sizeof(float));
    }
    return ln;
}

// ---- execution helpers (all per thread) ----
struct Exec {
    Ctx &c;
    Scratch &s;
    memory spad(const memory::desc &md) const { return memory(md, c.eng, s.spad.p); }
    void linear(const Linear &L, int64_t M, int64_t K, int64_t N, void *src, void *dst, void *res = nullptr,
                int phase = -1) const {
        pclock::time_point t0;
        if (g_profile) t0 = pclock::now();
        std::unordered_map<int, memory> args{
            {DNNL_ARG_SRC, memory({{M, K}, dt::bf16, tag::ab}, c.eng, src)},
            {DNNL_ARG_WEIGHTS, L.w}, {DNNL_ARG_BIAS, L.b},
            {DNNL_ARG_DST, memory({{M, N}, dt::bf16, tag::ab}, c.eng, dst)},
            {DNNL_ARG_SCRATCHPAD, spad(L.spad)}};
        if (L.residual)
            args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(L.res_pos) | DNNL_ARG_SRC_1,
                         memory({{M, N}, dt::bf16, tag::ab}, c.eng, res)});
        pclock::time_point t1;
        if (g_profile) t1 = pclock::now();
        L.prim.execute(*s.strm, args);
        if (g_profile) {
            const auto t2 = pclock::now();
            s.prof.ph[s.cur_bb][PH_ARGS] += secs(t0, t1);
            if (phase >= 0) s.prof.ph[s.cur_bb][phase] += secs(t1, t2);
            else s.prof.misc[MS_HEAD] += secs(t1, t2);
        }
    }
    void ln(const LayerNorm &l, int64_t M, int64_t C, void *src, void *dst, bool in_backbone = true) const {
        pclock::time_point t0;
        if (g_profile) t0 = pclock::now();
        memory::desc md({M, C}, dt::bf16, tag::ab);
        std::unordered_map<int, memory> args{{DNNL_ARG_SRC, memory(md, c.eng, src)},
                                             {DNNL_ARG_DST, memory(md, c.eng, dst)},
                                             {DNNL_ARG_SCRATCHPAD, spad(l.spad)}};
        if (l.affine) {
            args.insert({DNNL_ARG_SCALE, l.scale});
            args.insert({DNNL_ARG_SHIFT, l.shift});
        }
        pclock::time_point t1;
        if (g_profile) t1 = pclock::now();
        l.prim.execute(*s.strm, args);
        if (g_profile) {
            const auto t2 = pclock::now();
            s.prof.ph[s.cur_bb][PH_ARGS] += secs(t0, t1);
            if (in_backbone) s.prof.ph[s.cur_bb][PH_LN] += secs(t1, t2);
            else s.prof.misc[MS_HEAD] += secs(t1, t2);
        }
    }
};

BackboneT &backbone_for(Ctx &c, Backbone &bb, int64_t T) {       // serial: called before the parallel loop
    auto it = bb.byT.find(T);
    if (it != bb.byT.end()) return *it->second;
    auto bt = std::make_unique<BackboneT>();
    const int64_t P = bb.P, M = P * T;
    bt->P = P;
    bt->T = T;
    bt->M = M;
    bt->emb1 = make_linear(c, bb.emb1, M, true, false);
    bt->emb2 = make_linear(c, bb.emb2, M, false, true);
    for (int l = 0; l < NL; ++l) {
        bt->qkv[l] = make_linear(c, bb.qkv[l], M, false, false);
        bt->out[l] = make_linear(c, bb.out[l], M, false, true);
        bt->ffn1[l] = make_linear(c, bb.ffn1[l], M, true, false);
        bt->ffn2[l] = make_linear(c, bb.ffn2[l], M, false, true);
        bt->ln1[l] = make_ln(c, M, D, &bb.n1s[l], &bb.n1b[l]);
        bt->ln2[l] = make_ln(c, M, D, &bb.n2s[l], &bb.n2b[l]);
    }
    bt->q_md = memory::desc({P, NH, T, HD}, dt::bf16, dims{T * 3 * D, HD, 3 * D, 1});
    bt->kT_md = memory::desc({P, NH, HD, T}, dt::bf16, dims{T * 3 * D, HD, 1, 3 * D});
    bt->s_md = memory::desc({P, NH, T, T}, dt::bf16, tag::abcd);
    bt->v_md = memory::desc({P, NH, T, HD}, dt::bf16, dims{T * 3 * D, HD, 3 * D, 1});
    bt->o_md = memory::desc({P, NH, T, HD}, dt::bf16, dims{T * D, HD, D, 1});
    post_ops po;
    po.append_eltwise(algorithm::eltwise_linear, 1.f / std::sqrt(float(HD)), 0.f);
    matmul::primitive_desc qk_pd(c.eng, bt->q_md, bt->kT_md, bt->s_md, user_scratch_attr(po));
    softmax_forward::primitive_desc sm_pd(c.eng, prop_kind::forward_inference, algorithm::softmax_accurate,
                                          bt->s_md, bt->s_md, 3, user_scratch_attr());
    matmul::primitive_desc sv_pd(c.eng, bt->s_md, bt->v_md, bt->o_md, user_scratch_attr());
    for (size_t sz : {qk_pd.scratchpad_desc().get_size(), sm_pd.scratchpad_desc().get_size(),
                      sv_pd.scratchpad_desc().get_size()})
        c.spad_bytes = std::max(c.spad_bytes, sz);
    bt->qk = matmul(qk_pd);
    bt->sm = softmax_forward(sm_pd);
    bt->sv = matmul(sv_pd);
    bt->qk_spad = qk_pd.scratchpad_desc();
    bt->sm_spad = sm_pd.scratchpad_desc();
    bt->sv_spad = sv_pd.scratchpad_desc();
    return *bb.byT.emplace(T, std::move(bt)).first->second;
}

void reserve_scratch(Ctx &c, Scratch &s, int64_t T2, int64_t T1) {  // run by the owning thread
    const int64_t P = c.P, M = std::max(c.bb[0].P * T2, c.bb[1].P * T1);
    const int64_t SC = std::max(c.bb[0].P * T2 * T2, c.bb[1].P * T1 * T1);
    s.bands.reserve(M * 16 * 2);
    s.pe.reserve(M * D * 2);
    for (Buf *b : {&s.x, &s.x1, &s.y, &s.y2, &s.att}) b->reserve(M * D * 2);
    s.qkvb.reserve(M * 3 * D * 2);
    s.hbuf.reserve(M * FF * 2);
    s.scores.reserve(NH * SC * 2);
    s.fused.reserve(P * 2 * D * 2);
    s.r1.reserve(P * RED * 2);
    s.r2.reserve(P * OUT * 2);
    s.pad.reserve(P * std::max(T2 * 11, T1 * 3) * 4);
    s.spad.reserve(std::max<size_t>(c.spad_bytes, 64));
    if (!s.strm) s.strm = std::make_unique<stream>(c.eng);
}

// One backbone on one thread block: in (P, T, bands+1) f32 -> pooled (P, 640)
// bf16 into fused[:, col0 : col0+640].
void run_backbone(const Exec &e, Backbone &bb, BackboneT &bt, const float *in, uint16_t *fused, int64_t col0) {
    Ctx &c = e.c;
    Scratch &s = e.s;
    s.cur_bb = col0 == 0 ? 0 : 1;
    pclock::time_point tp0;
    if (g_profile) tp0 = pclock::now();
    const int64_t P = bt.P, T = bt.T, M = bt.M, C = bb.bands, IN = C + 1;
    uint16_t *bands = s.bands.as<uint16_t>(), *pe = s.pe.as<uint16_t>();
    uint16_t *x = s.x.as<uint16_t>(), *x1 = s.x1.as<uint16_t>(), *y = s.y.as<uint16_t>();
    uint16_t *y2 = s.y2.as<uint16_t>(), *qkv = s.qkvb.as<uint16_t>(), *att = s.att.as<uint16_t>();
    uint16_t *h = s.hbuf.as<uint16_t>(), *sc = s.scores.as<uint16_t>();

    for (int64_t r = 0; r < M; ++r) {
        const float *row = in + r * IN;
        for (int64_t k = 0; k < C; ++k) bands[r * C + k] = f2bf(row[k]);
        const float doy = row[C];
        const int di = int(doy);
        if (float(di) == doy && di >= 0 && di <= 366) {
            std::memcpy(pe + r * D, c.pe_table.data() + di * D, D * sizeof(uint16_t));
        } else {
            for (int64_t i = 0; i < D / 2; ++i) {
                const float a = doy * std::exp(float(2 * i) * -(std::log(10000.0f) / float(D)));
                pe[r * D + 2 * i] = f2bf(std::sin(a));
                pe[r * D + 2 * i + 1] = f2bf(std::cos(a));
            }
        }
    }

    if (g_profile) s.prof.ph[s.cur_bb][PH_PREP] += secs(tp0, pclock::now());
    e.linear(bt.emb1, M, C, D, bands, y, nullptr, PH_EMB);
    e.linear(bt.emb2, M, D, D, y, x, pe, PH_EMB);
    for (int l = 0; l < NL; ++l) {
        e.linear(bt.qkv[l], M, D, 3 * D, x, qkv, nullptr, PH_QKV);
        pclock::time_point ta0, ta1;
        if (g_profile) ta0 = pclock::now();
        memory mq(bt.q_md, c.eng, qkv), mk(bt.kT_md, c.eng, qkv + D), ms(bt.s_md, c.eng, sc);
        memory mv(bt.v_md, c.eng, qkv + 2 * D), mo(bt.o_md, c.eng, att);
        memory sp_qk = e.spad(bt.qk_spad), sp_sm = e.spad(bt.sm_spad), sp_sv = e.spad(bt.sv_spad);
        if (g_profile) ta1 = pclock::now();
        bt.qk.execute(*s.strm, {{DNNL_ARG_SRC, mq}, {DNNL_ARG_WEIGHTS, mk}, {DNNL_ARG_DST, ms},
                                {DNNL_ARG_SCRATCHPAD, sp_qk}});
        bt.sm.execute(*s.strm, {{DNNL_ARG_SRC, ms}, {DNNL_ARG_DST, ms}, {DNNL_ARG_SCRATCHPAD, sp_sm}});
        bt.sv.execute(*s.strm, {{DNNL_ARG_SRC, ms}, {DNNL_ARG_WEIGHTS, mv}, {DNNL_ARG_DST, mo},
                                {DNNL_ARG_SCRATCHPAD, sp_sv}});
        if (g_profile) {
            const auto ta2 = pclock::now();
            s.prof.ph[s.cur_bb][PH_ARGS] += secs(ta0, ta1);
            s.prof.ph[s.cur_bb][PH_ATT] += secs(ta1, ta2);
        }
        e.linear(bt.out[l], M, D, D, att, y, x, PH_OUT);
        e.ln(bt.ln1[l], M, D, y, x1);
        e.linear(bt.ffn1[l], M, D, FF, x1, h, nullptr, PH_FFN1);
        e.linear(bt.ffn2[l], M, FF, D, h, y2, x1, PH_FFN2);
        e.ln(bt.ln2[l], M, D, y2, x);
    }
    s.strm->wait();
    pclock::time_point tpool;
    if (g_profile) tpool = pclock::now();

    const float *qw = bb.pool_w.data();
    float sbuf[256];
    std::vector<float> acc(D);
    for (int64_t p = 0; p < P; ++p) {
        const uint16_t *xp = x + p * T * D;
        float mx = -INFINITY;
        for (int64_t t = 0; t < T; ++t) {
            float d = bb.pool_b;
            for (int64_t k = 0; k < D; ++k) d += bf2f(xp[t * D + k]) * qw[k];
            sbuf[t] = d;
            mx = std::max(mx, d);
        }
        float z = 0.f;
        for (int64_t t = 0; t < T; ++t) z += (sbuf[t] = std::exp(sbuf[t] - mx));
        std::fill(acc.begin(), acc.end(), 0.f);
        for (int64_t t = 0; t < T; ++t) {
            const float w = sbuf[t] / z;
            for (int64_t k = 0; k < D; ++k) acc[k] += w * bf2f(xp[t * D + k]);
        }
        uint16_t *fo = fused + p * 2 * D + col0;
        for (int64_t k = 0; k < D; ++k) fo[k] = f2bf(acc[k]);
    }
    if (g_profile) s.prof.ph[s.cur_bb][PH_POOL] += secs(tpool, pclock::now());
}

void init_backbone(std::map<std::string, Tensor> &W, Backbone &bb, const std::string &prefix, int64_t bands) {
    bb.bands = bands;
    bb.emb1 = take_linear(W, prefix + ".embedding.0");
    bb.emb2 = take_linear(W, prefix + ".embedding.2");
    for (int l = 0; l < NL; ++l) {
        const std::string L = prefix + ".transformer_encoder.layers." + std::to_string(l);
        bb.qkv[l] = take_linear(W, L + ".self_attn.in_proj_weight", L + ".self_attn.in_proj_bias");
        bb.out[l] = take_linear(W, L + ".self_attn.out_proj");
        bb.ffn1[l] = take_linear(W, L + ".linear1");
        bb.ffn2[l] = take_linear(W, L + ".linear2");
        bb.n1s[l] = W.at(L + ".norm1.weight").data;
        bb.n1b[l] = W.at(L + ".norm1.bias").data;
        bb.n2s[l] = W.at(L + ".norm2.weight").data;
        bb.n2b[l] = W.at(L + ".norm2.bias").data;
    }
    bb.pool_w = W.at(prefix + ".attn_pool.query.weight").data;
    bb.pool_b = W.at(prefix + ".attn_pool.query.bias").data[0];
}

}  // namespace

extern "C" {

void *tc_create(const char *weights_path, int block_pixels) {
    try {
        auto W = load_weights(weights_path);
        auto *c = new Ctx();
        c->P1 = block_pixels > 0 ? block_pixels : 4;
        const char *b2 = std::getenv("TC_BLOCK_S2");
        c->P = b2 && std::atoi(b2) > 0 ? std::atoi(b2) : c->P1;
        if (c->P % c->P1) throw std::runtime_error("TC_BLOCK_S2 must be a multiple of TC_BLOCK");
        c->nthr = omp_get_max_threads();
        init_backbone(W, c->bb[0], "s2_backbone", 10);
        init_backbone(W, c->bb[1], "s1_backbone", 2);
        c->bb[0].P = c->P;
        c->bb[1].P = c->P1;
        c->red1 = take_linear(W, "dim_reducer.0");
        c->red2 = take_linear(W, "dim_reducer.4");
        c->rln_s = W.at("dim_reducer.1.weight").data;
        c->rln_b = W.at("dim_reducer.1.bias").data;
        c->red1L = make_linear(*c, c->red1, c->P, false, false);
        c->red2L = make_linear(*c, c->red2, c->P, false, false);
        c->rln = make_ln(*c, c->P, RED, &c->rln_s, &c->rln_b);
        c->oln = make_ln(*c, c->P, OUT, nullptr, nullptr);
        c->pe_table.resize(367 * D);
        for (int d = 0; d <= 366; ++d)
            for (int64_t i = 0; i < D / 2; ++i) {
                const float a = float(d) * std::exp(float(2 * i) * -(std::log(10000.0f) / float(D)));
                c->pe_table[d * D + 2 * i] = f2bf(std::sin(a));
                c->pe_table[d * D + 2 * i + 1] = f2bf(std::cos(a));
            }
        for (int t = 0; t < c->nthr; ++t) c->scratch.push_back(std::make_unique<Scratch>());
        return c;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "tc_create: %s\n", e.what());
        return nullptr;
    }
}

int tc_encode(void *handle, const float *s2, int64_t G, int64_t T2, const float *s1, int64_t T1, float *out) {
    Ctx &c = *static_cast<Ctx *>(handle);
    const auto tcall = pclock::now();
    try {
        BackboneT &b2 = backbone_for(c, c.bb[0], T2);           // serial setup for this batch's shapes
        BackboneT &b1 = backbone_for(c, c.bb[1], T1);
        const int64_t P = c.P, P1 = c.P1, nblk = (G + P - 1) / P;
        int err = 0;
        pclock::time_point tr0;
#pragma omp parallel num_threads(c.nthr)
        {
            Scratch &s = *c.scratch[omp_get_thread_num()];
            try {
                reserve_scratch(c, s, T2, T1);
            } catch (...) {
#pragma omp atomic write
                err = 1;
            }
#pragma omp barrier
#pragma omp master
            tr0 = pclock::now();
            Exec e{c, s};
#pragma omp for schedule(dynamic, 1)
            for (int64_t bi = 0; bi < nblk; ++bi) {
                if (err) continue;
                try {
                    pclock::time_point tb0;
                    if (g_profile) tb0 = pclock::now();
                    const int64_t p0 = bi * P, n = std::min(P, G - p0);
                    const float *in2 = s2 + p0 * T2 * 11, *in1 = s1 + p0 * T1 * 3;
                    uint16_t *fused = s.fused.as<uint16_t>(), *r1 = s.r1.as<uint16_t>(), *r2 = s.r2.as<uint16_t>();
                    float *pad = s.pad.as<float>();
                    if (n < P) {
                        auto tq = pclock::now();
                        std::memset(pad, 0, P * T2 * 11 * 4);
                        std::memcpy(pad, in2, n * T2 * 11 * 4);
                        if (g_profile) s.prof.misc[MS_PAD] += secs(tq, pclock::now());
                        run_backbone(e, c.bb[0], b2, pad, fused, 0);
                        tq = pclock::now();
                        std::memset(pad, 0, P * T1 * 3 * 4);
                        std::memcpy(pad, in1, n * T1 * 3 * 4);
                        if (g_profile) s.prof.misc[MS_PAD] += secs(tq, pclock::now());
                        for (int64_t q = 0; q < P; q += P1)           // S1 in sub-blocks of P1
                            run_backbone(e, c.bb[1], b1, pad + q * T1 * 3, fused + q * 2 * D, D);
                    } else {
                        run_backbone(e, c.bb[0], b2, in2, fused, 0);
                        for (int64_t q = 0; q < P; q += P1)
                            run_backbone(e, c.bb[1], b1, in1 + q * T1 * 3, fused + q * 2 * D, D);
                    }
                    e.linear(c.red1L, P, 2 * D, RED, fused, r1);
                    e.ln(c.rln, P, RED, r1, r1, false);
                    s.strm->wait();
                    for (int64_t i = 0; i < P * RED; ++i)
                        if (r1[i] & 0x8000u) r1[i] = 0;             // relu on bf16
                    e.linear(c.red2L, P, RED, OUT, r1, r2);
                    e.ln(c.oln, P, OUT, r2, r2, false);
                    s.strm->wait();
                    for (int64_t i = 0; i < n * OUT; ++i) out[p0 * OUT + i] = bf2f(r2[i]);
                    if (g_profile) {
                        s.prof.busy += secs(tb0, pclock::now());
                        ++s.prof.blocks;
                    }
                } catch (const std::exception &ex) {
                    std::fprintf(stderr, "tc_encode block %ld: %s\n", (long)bi, ex.what());
#pragma omp atomic write
                    err = 1;
                }
            }
        }
        if (g_profile) {
            const auto tend = pclock::now();
            c.region_wall += secs(tr0, tend);
            c.call_wall += secs(tcall, tend);
            ++c.calls;
        }
        return err ? -1 : 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "tc_encode: %s\n", e.what());
        return -1;
    }
}

// Print the profile: per phase, thread-seconds and share of all thread time
// inside parallel regions (nthr x region wall). Idle = region time with no
// block to run (load imbalance / tail of small batches).
void tc_profile_dump(void *handle) {
    Ctx &c = *static_cast<Ctx *>(handle);
    if (!g_profile) { std::printf("TCPROF disabled (set TC_PROFILE=1)\n"); return; }
    Prof tot;
    for (auto &sp : c.scratch) {
        const Prof &p = sp->prof;
        for (int b = 0; b < 2; ++b)
            for (int k = 0; k < PH_N; ++k) tot.ph[b][k] += p.ph[b][k];
        for (int k = 0; k < MS_N; ++k) tot.misc[k] += p.misc[k];
        tot.busy += p.busy;
        tot.blocks += p.blocks;
    }
    const double avail = c.region_wall * c.nthr;
    std::printf("TCPROF calls=%lu blocks=%lu threads=%d block_px S2=%ld S1=%ld  tc_encode wall %.2f s  "
                "(setup outside region %.2f s)\n",
                (unsigned long)c.calls, (unsigned long)tot.blocks, c.nthr, (long)c.P, (long)c.P1, c.call_wall,
                c.call_wall - c.region_wall);
    std::printf("TCPROF %-22s %10s %10s %10s %7s\n", "phase", "S2 thr-s", "S1 thr-s", "total", "share");
    double accounted = 0;
    for (int k = 0; k < PH_N; ++k) {
        const double t = tot.ph[0][k] + tot.ph[1][k];
        accounted += t;
        std::printf("TCPROF %-22s %10.2f %10.2f %10.2f %6.1f%%\n", phase_name[k], tot.ph[0][k], tot.ph[1][k], t,
                    100 * t / avail);
    }
    for (int k = 0; k < MS_N; ++k) {
        accounted += tot.misc[k];
        std::printf("TCPROF %-22s %10s %10s %10.2f %6.1f%%\n", misc_name[k], "", "", tot.misc[k], 100 * tot.misc[k] / avail);
    }
    const double other = tot.busy - accounted, idle = avail - tot.busy;
    std::printf("TCPROF %-22s %10s %10s %10.2f %6.1f%%\n", "other in-block", "", "", other, 100 * other / avail);
    std::printf("TCPROF %-22s %10s %10s %10.2f %6.1f%%\n", "idle (no block)", "", "", idle, 100 * idle / avail);
    std::printf("TCPROF %-22s %10s %10s %10.2f %6.1f%%\n", "total thread time", "", "", avail, 100.0);
    std::fflush(stdout);
}

void tc_destroy(void *handle) { delete static_cast<Ctx *>(handle); }

}  // extern "C"
