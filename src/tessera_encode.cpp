// tessera_encode: standalone Tessera v2 tile encoder (no Python).
//
//   tessera_encode TILE_DIR OUT_DIR [WEIGHTS]
//
// Reads a staged DPixel tile (dpixel.save layout: s2/{bands,masks,doys}.npy,
// s1/sar_{ascending,descending}{,_doy}.npy, or the same files flat), runs the
// v2 pixel student through the forward pass in tessera_cpu.cpp (or tessera_cuda.cu),
// and writes OUT_DIR/<grid>.npy (H, W, 128) int8 and <grid>_scales.npy (H, W)
// f32 - the same products as the Python path.
//
// Everything the Python path did around the network is reproduced here
// exactly (tessera_v2_student/infer.py and quantize.py):
//   - per-pixel valid observations: S2 mask != 0; S1 any band != 0 (raw)
//   - bin = next multiple of 8 above the count, capped at 256 (0 if none)
//   - pixels grouped by (s2_bin, s1_bin) in ascending key order, each group
//     cut into batches of TC_BATCH (default 4096, as Python) in ascending pixel order
//   - valid time steps first (stable), then padded/subsampled to the bin with
//     _pad_pattern (numpy linspace floor / array_split middle / modulo fill)
//   - z-score in f32 with the model's constants: S2 after gathering, S1
//     ascending and descending separately before merging (asc then desc)
//   - int8 quantisation: scale = max(|e|, 1e-12) / 127, code = round-half-even
//     (e / scale) clipped to [-127, 127]
// Batch preparation runs on the same OpenMP threads as the network, in
// parallel over the batch's pixels, immediately before each batch.
//
// Env: TC_BATCH (pixels per network call, default 4096; larger cuts the idle tail
//      when one process uses many threads), TC_BLOCK (S1 px per thread block,
//      default 4), TC_BLOCK_S2 (default 8),
//      OMP_NUM_THREADS, TC_PROFILE=1 for the library's phase profile.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <omp.h>

extern "C" {
void *tc_create(const char *weights_path, int block_pixels);
int tc_encode(void *h, const float *s2, int64_t G, int64_t T2, const float *s1, int64_t T1, float *out);
void tc_profile_dump(void *h);
void tc_destroy(void *h);
}

namespace {

using clk = std::chrono::steady_clock;
double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

// ---- constants from tessera_v2_student/model.py ----
const float S2_MEAN[10] = {1633.0042f, 1341.1090f, 1539.5536f, 3054.8269f, 3117.4658f,
                           2004.1648f, 2694.7275f, 2945.1504f, 2266.6079f, 1657.3094f};
const float S2_STD[10] = {1999.4603f, 2014.7549f, 1929.2201f, 1754.2493f, 1649.9807f,
                          1936.8988f, 1748.6041f, 1708.6991f, 1207.5250f, 1108.6046f};
const float S1A_MEAN[2] = {5909.3921f, 3405.0322f}, S1A_STD[2] = {1507.1750f, 1531.2615f};
const float S1D_MEAN[2] = {5816.1382f, 3277.7576f}, S1D_STD[2] = {1554.6475f, 1546.4733f};
constexpr int64_t EMB = 128;
constexpr float EPS = 1e-9f;

// ---- minimal .npy I/O ----
struct Npy {
    std::string descr;
    std::vector<int64_t> shape;
    std::vector<char> data;
    int64_t count() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
};

Npy read_npy(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6)) throw std::runtime_error("not a .npy file: " + path);
    unsigned char ver[2];
    f.read(reinterpret_cast<char *>(ver), 2);
    uint32_t hlen = 0;
    if (ver[0] == 1) { uint16_t h; f.read(reinterpret_cast<char *>(&h), 2); hlen = h; }
    else { f.read(reinterpret_cast<char *>(&hlen), 4); }
    std::string hdr(hlen, '\0');
    f.read(hdr.data(), hlen);
    Npy a;
    auto field = [&](const std::string &key) {
        auto k = hdr.find("'" + key + "'");
        if (k == std::string::npos) throw std::runtime_error("bad npy header in " + path);
        return hdr.find(':', k) + 1;
    };
    auto d = hdr.find('\'', field("descr"));
    a.descr = hdr.substr(d + 1, hdr.find('\'', d + 1) - d - 1);
    if (hdr.find("True", field("fortran_order")) < hdr.find(',', field("fortran_order")))
        throw std::runtime_error("fortran-order npy not supported: " + path);
    auto s0 = hdr.find('(', field("shape")), s1 = hdr.find(')', s0);
    std::stringstream ss(hdr.substr(s0 + 1, s1 - s0 - 1));
    for (std::string tok; std::getline(ss, tok, ',');)
        if (tok.find_first_not_of(" ") != std::string::npos) a.shape.push_back(std::stoll(tok));
    const int64_t esz = std::stoi(a.descr.substr(2));
    a.data.resize(a.count() * esz);
    f.read(a.data.data(), a.data.size());
    if (!f) throw std::runtime_error("truncated npy: " + path);
    return a;
}

template <class T> const T *as(const Npy &a, const char *descr) {
    if (a.descr != descr) throw std::runtime_error(std::string("expected dtype ") + descr + ", got " + a.descr);
    return reinterpret_cast<const T *>(a.data.data());
}

void write_npy(const std::string &path, const char *descr, const std::vector<int64_t> &shape,
               const void *data, size_t bytes) {
    std::string sh = "(";                  // numpy style: "(a, b, c)" or "(n,)"
    for (size_t i = 0; i < shape.size(); ++i) sh += (i ? ", " : "") + std::to_string(shape[i]);
    sh += shape.size() == 1 ? ",)" : ")";
    std::string hdr = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': " + sh + ", }";
    const size_t unpadded = 10 + hdr.size() + 1;            // magic+version+len, header, '\n'
    hdr.append((64 - unpadded % 64) % 64, ' ');
    hdr += '\n';
    std::ofstream f(path, std::ios::binary);
    f.write("\x93NUMPY\x01\x00", 8);
    const uint16_t hl = uint16_t(hdr.size());
    f.write(reinterpret_cast<const char *>(&hl), 2);
    f.write(hdr.data(), hdr.size());
    f.write(static_cast<const char *>(data), bytes);
    if (!f) throw std::runtime_error("write failed: " + path);
}

bool exists(const std::string &p) { struct stat st; return stat(p.c_str(), &st) == 0; }

void mkdirs(const std::string &path) {             // mkdir -p
    for (size_t i = 1; i <= path.size(); ++i)
        if (i == path.size() || path[i] == '/') {
            const std::string part = path.substr(0, i);
            if (!exists(part) && mkdir(part.c_str(), 0755) != 0 && !exists(part))
                throw std::runtime_error("cannot create directory " + part);
        }
}

// ---- binning and padding, as tessera_v2_student/infer.py ----
int bin_size(int64_t n) {
    if (n <= 0) return 0;
    for (int b = 8; b <= 256; b += 8)
        if (n <= b) return b;
    return 256;
}

// _pad_pattern(n, B): (B,) indices into [0, n)
void pad_pattern(int64_t n, int64_t B, std::vector<int64_t> &out) {
    out.assign(B, 0);
    if (n == 0) return;
    if (n >= B) {                          // np.linspace(0, n-1, B, dtype=int64): float64, floor, exact end
        if (B == 1) { out[0] = 0; return; }
        const double delta = double(n - 1), div = double(B - 1), step = delta / div;
        for (int64_t i = 0; i < B; ++i) {
            double y = step == 0 ? (double(i) / div) * delta : double(i) * step;
            out[i] = int64_t(std::floor(y));
        }
        out[B - 1] = n - 1;
        return;
    }
    for (int64_t i = 0; i < n; ++i) out[i] = i;
    const int64_t remain = B - n;
    if (remain <= n) {                     // np.array_split(arange(n), remain): middle of each group
        const int64_t q = n / remain, r = n % remain;
        int64_t start = 0;
        for (int64_t g = 0; g < remain; ++g) {
            const int64_t len = q + (g < r ? 1 : 0);
            out[n + g] = start + len / 2;
            start += len;
        }
    } else {
        for (int64_t i = 0; i < remain; ++i) out[n + i] = i % n;
    }
}

struct Tile {
    std::string grid;
    int64_t H = 0, W = 0, N = 0, T2 = 0, Ta = 0, Td = 0;
    Npy s2b, s2m, s2d, s1a, s1ad, s1d, s1dd;
};

Tile load_tile(const std::string &dir) {
    Tile t;
    std::string d = dir;
    while (d.size() > 1 && d.back() == '/') d.pop_back();
    t.grid = d.substr(d.rfind('/') + 1);
    const bool sub = exists(d + "/s2");
    const std::string s2 = sub ? d + "/s2/" : d + "/", s1 = sub ? d + "/s1/" : d + "/";
    t.s2b = read_npy(s2 + "bands.npy");
    t.s2m = read_npy(s2 + "masks.npy");
    t.s2d = read_npy(s2 + "doys.npy");
    t.s1a = read_npy(s1 + "sar_ascending.npy");
    t.s1ad = read_npy(s1 + "sar_ascending_doy.npy");
    t.s1d = read_npy(s1 + "sar_descending.npy");
    t.s1dd = read_npy(s1 + "sar_descending_doy.npy");
    t.T2 = t.s2b.shape[0];
    t.H = t.s2b.shape[1];
    t.W = t.s2b.shape[2];
    t.N = t.H * t.W;
    t.Ta = t.s1a.count() ? t.s1a.shape[0] : 0;
    t.Td = t.s1d.count() ? t.s1d.shape[0] : 0;
    return t;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s TILE_DIR OUT_DIR [WEIGHTS]\n", argv[0]);
        return 2;
    }
    const std::string tile_dir = argv[1], out_dir = argv[2];
    std::string weights = argc > 3 ? argv[3] : "";
    if (weights.empty()) {
        const std::string self = argv[0];
        const auto slash = self.rfind('/');
        weights = (slash == std::string::npos ? std::string(".") : self.substr(0, slash)) + "/../student_large.tcw";
    }
    const char *bat = std::getenv("TC_BATCH");
    const int64_t BATCH = bat && std::atoll(bat) > 0 ? std::atoll(bat) : 4096;
    const char *blk = std::getenv("TC_BLOCK");
    const int block = blk ? std::atoi(blk) : 4;
    if (!std::getenv("TC_BLOCK_S2")) setenv("TC_BLOCK_S2", "8", 0);

    try {
        auto t_all = clk::now();
        auto t0 = clk::now();
        Tile t = load_tile(tile_dir);
        const double t_load = since(t0);
        const int64_t N = t.N, T2 = t.T2, Ta = t.Ta, Td = t.Td, T1 = Ta + Td;
        const uint16_t *s2b = as<uint16_t>(t.s2b, "<u2");
        const uint8_t *s2m = as<uint8_t>(t.s2m, "|u1");
        const uint16_t *s2d = as<uint16_t>(t.s2d, "<u2");
        const int16_t *s1a = Ta ? as<int16_t>(t.s1a, "<i2") : nullptr;
        const int16_t *s1d = Td ? as<int16_t>(t.s1d, "<i2") : nullptr;
        const int16_t *s1ad = Ta ? as<int16_t>(t.s1ad, "<i2") : nullptr;
        const int16_t *s1dd = Td ? as<int16_t>(t.s1dd, "<i2") : nullptr;

        // ---- per-pixel valid counts and bins ----
        t0 = clk::now();
        std::vector<int32_t> n2(N), n1(N), key(N);
#pragma omp parallel for schedule(static)
        for (int64_t p = 0; p < N; ++p) {
            int c2 = 0, c1 = 0;
            for (int64_t s = 0; s < T2; ++s) c2 += s2m[s * N + p] != 0;
            for (int64_t s = 0; s < Ta; ++s) c1 += (s1a[(s * N + p) * 2] != 0) | (s1a[(s * N + p) * 2 + 1] != 0);
            for (int64_t s = 0; s < Td; ++s) c1 += (s1d[(s * N + p) * 2] != 0) | (s1d[(s * N + p) * 2 + 1] != 0);
            n2[p] = c2;
            n1[p] = c1;
            key[p] = bin_size(c2) * 1000 + bin_size(c1);
        }
        std::vector<int64_t> order(N);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return key[a] < key[b]; });
        const double t_bin = since(t0);

        // ---- network ----
        void *h = tc_create(weights.c_str(), block);
        if (!h) throw std::runtime_error("tc_create failed");
        std::vector<float> emb(N * EMB, 0.f);        // pixels with no observations stay zero
        std::vector<float> s2in, s1in, out;
        double t_prep = 0, t_net = 0;
        int64_t batches = 0;
        for (int64_t g0 = 0; g0 < N;) {
            int64_t g1 = g0;
            while (g1 < N && key[order[g1]] == key[order[g0]]) ++g1;
            const int b2 = key[order[g0]] / 1000, b1 = key[order[g0]] % 1000;
            if (b2 == 0 && b1 == 0) { g0 = g1; continue; }
            const int64_t B2 = std::max(b2, 1), B1 = std::max(b1, 1);
            for (int64_t c0 = g0; c0 < g1; c0 += BATCH) {
                const int64_t G = std::min(BATCH, g1 - c0);
                auto tp = clk::now();
                s2in.assign(G * B2 * 11, 0.f);
                s1in.assign(G * B1 * 3, 0.f);
#pragma omp parallel
                {
                    std::vector<int64_t> pos2, pos1, pat;
#pragma omp for schedule(static)
                    for (int64_t i = 0; i < G; ++i) {
                        const int64_t p = order[c0 + i];
                        if (b2 > 0) {                  // S2: valid steps first, pad to B2, z-score
                            pos2.clear();
                            for (int64_t s = 0; s < T2; ++s) if (s2m[s * N + p]) pos2.push_back(s);
                            const int64_t nv = pos2.size();
                            for (int64_t s = 0; s < T2; ++s) if (!s2m[s * N + p]) pos2.push_back(s);
                            pad_pattern(nv, B2, pat);
                            float *row = s2in.data() + i * B2 * 11;
                            for (int64_t k = 0; k < B2; ++k) {
                                const int64_t s = pos2[pat[k]];
                                const uint16_t *src = s2b + (s * N + p) * 10;
                                for (int c = 0; c < 10; ++c)
                                    row[k * 11 + c] = (float(src[c]) - S2_MEAN[c]) / (S2_STD[c] + EPS);
                                row[k * 11 + 10] = float(s2d[s]);
                            }
                        }
                        if (b1 > 0) {                  // S1: asc then desc merged, each z-scored with its stats
                            pos1.clear();
                            auto valid1 = [&](int64_t s) {
                                const int16_t *v = s < Ta ? s1a + (s * N + p) * 2 : s1d + ((s - Ta) * N + p) * 2;
                                return v[0] != 0 || v[1] != 0;
                            };
                            for (int64_t s = 0; s < T1; ++s) if (valid1(s)) pos1.push_back(s);
                            const int64_t nv = pos1.size();
                            for (int64_t s = 0; s < T1; ++s) if (!valid1(s)) pos1.push_back(s);
                            pad_pattern(nv, B1, pat);
                            float *row = s1in.data() + i * B1 * 3;
                            for (int64_t k = 0; k < B1; ++k) {
                                const int64_t s = pos1[pat[k]];
                                if (s < Ta) {
                                    const int16_t *v = s1a + (s * N + p) * 2;
                                    for (int c = 0; c < 2; ++c)
                                        row[k * 3 + c] = (float(v[c]) - S1A_MEAN[c]) / (S1A_STD[c] + EPS);
                                    row[k * 3 + 2] = float(s1ad[s]);
                                } else {
                                    const int16_t *v = s1d + ((s - Ta) * N + p) * 2;
                                    for (int c = 0; c < 2; ++c)
                                        row[k * 3 + c] = (float(v[c]) - S1D_MEAN[c]) / (S1D_STD[c] + EPS);
                                    row[k * 3 + 2] = float(s1dd[s - Ta]);
                                }
                            }
                        }
                    }
                }
                t_prep += since(tp);
                tp = clk::now();
                out.resize(G * EMB);
                if (tc_encode(h, s2in.data(), G, B2, s1in.data(), B1, out.data()) != 0)
                    throw std::runtime_error("tc_encode failed");
                t_net += since(tp);
#pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < G; ++i)
                    std::memcpy(emb.data() + order[c0 + i] * EMB, out.data() + i * EMB, EMB * sizeof(float));
                ++batches;
            }
            g0 = g1;
        }

        // ---- int8 quantisation (quantize.py) and output ----
        t0 = clk::now();
        std::vector<int8_t> code(N * EMB);
        std::vector<float> scale(N);
#pragma omp parallel for schedule(static)
        for (int64_t p = 0; p < N; ++p) {
            const float *e = emb.data() + p * EMB;
            float am = 0.f;
            for (int64_t k = 0; k < EMB; ++k) am = std::max(am, std::fabs(e[k]));
            am = std::max(am, 1e-12f);
            const float sc = am / 127.0f;
            scale[p] = sc;
            for (int64_t k = 0; k < EMB; ++k) {
                float q = std::nearbyint(e[k] / sc);   // round half to even, as np.round
                q = std::min(127.f, std::max(-127.f, q));
                code[p * EMB + k] = int8_t(q);
            }
        }
        mkdirs(out_dir);
        write_npy(out_dir + "/" + t.grid + ".npy", "|i1", {t.H, t.W, EMB}, code.data(), code.size());
        write_npy(out_dir + "/" + t.grid + "_scales.npy", "<f4", {t.H, t.W}, scale.data(), scale.size() * 4);
        const double t_out = since(t0);
        std::printf("ENCODE grid=%s px=%ld batch_px=%ld batches=%ld threads=%d  load=%.1fs bin=%.1fs prep=%.1fs net=%.1fs "
                    "quant+write=%.1fs total=%.1fs\n",
                    t.grid.c_str(), (long)N, (long)BATCH, (long)batches, omp_get_max_threads(), t_load, t_bin, t_prep, t_net,
                    t_out, since(t_all));
        std::fflush(stdout);
        tc_profile_dump(h);
        tc_destroy(h);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "tessera_encode: %s\n", e.what());
        return 1;
    }
    return 0;
}
