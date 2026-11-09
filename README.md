# tessera-inference

A standalone C++ encoder for the [Tessera](https://github.com/ucam-eo/tessera) v2 pixel student (Large). It turns one staged 0.1-degree tile of Sentinel-2 and Sentinel-1 observations into 128-dimensional int8 embeddings, with no Python or PyTorch at run time.

There are two back ends behind the same small C API:

- `src/tessera_cpu.cpp` uses oneDNN. Each thread takes its own small block of pixels through the whole model, so the working set stays in that core's cache. It runs best on Xeons with AMX. It also runs on AVX-512 CPUs such as AMD EPYC (Zen 4 and later), but CPUs without AVX-512 are not supported, because oneDNN has no bf16 matrix multiply for them.
- `src/tessera_cuda.cu` uses cuBLASLt and a few custom kernels. It processes L2-sized chunks of pixels.

`src/tessera_encode.cpp` loads the tile, bins and pads each pixel's time series, standardises the inputs and quantises the output. It does all of this exactly as the reference Python code does, and on the CPU back end its output files are byte-for-byte identical to the Python path.

The write-up of how this code came about, with all the measurements, is at <https://www.tunbury.org/2026/11/09/tessera-on-endeavour/>.

| Machine | One tile (Manchester, 772,875 px) | Whole node |
|---|---|---|
| 2 x Xeon 6980P, CPU back end | 32 s | 136.5 tiles per hour |
| NVIDIA L4, CUDA back end | 117 s | |
| NVIDIA L4, eager PyTorch, for comparison | 227 s | |

## Build

### CPU

The CPU back end needs oneDNN built with the **sequential** runtime, so that every call runs on the thread that made it. The code was developed against oneDNN v3.13.3.

```bash
git clone --branch v3.13.3 https://github.com/uxlfoundation/oneDNN
cmake -S oneDNN -B onednn-build -DCMAKE_BUILD_TYPE=Release \
      -DONEDNN_CPU_RUNTIME=SEQ -DONEDNN_BUILD_GRAPH=OFF \
      -DONEDNN_BUILD_TESTS=OFF -DONEDNN_BUILD_EXAMPLES=OFF \
      -DCMAKE_INSTALL_PREFIX=$PWD/onednn-seq
cmake --build onednn-build -j && cmake --install onednn-build

make ONEDNN=$PWD/onednn-seq
```

If you build on a different machine from the one you run on, set `MARCH` (for example `MARCH=sapphirerapids`).

### CUDA

```bash
make cuda CUDA=/usr/local/cuda CUDA_ARCH=sm_89     # sm_89 for an L4, sm_80 for an A30 or A100
```

## Weights

The encoder runs the v2 Large pixel student, [geotessera/TESSERA-V-2.0-2B-L](https://huggingface.co/geotessera/TESSERA-V-2.0-2B-L) on Hugging Face (CC0). Download the checkpoint and convert it to the flat `.tcw` file the encoder reads. The converter only needs `torch`.

```bash
curl -LO https://huggingface.co/geotessera/TESSERA-V-2.0-2B-L/resolve/main/ckpt/student_large.pt
python tools/export_weights.py student_large.pt student_large.tcw
```

The encoder is built for the Large architecture only, and `export_weights.py` refuses a checkpoint with any other shape.

## Input

A tile is a directory named after the grid cell (for example `grid_-1.65_53.35`). It contains the raw, un-normalised arrays written by Tessera's d-pixel stage:

| File | dtype | Shape |
|---|---|---|
| `s2/bands.npy` | uint16 | (T2, H, W, 10), bands B04 B02 B03 B08 B8A B05 B06 B07 B11 B12 |
| `s2/masks.npy` | uint8 | (T2, H, W), 1 = valid |
| `s2/doys.npy` | uint16 | (T2,), day of year |
| `s1/sar_ascending.npy` | int16 | (Ta, H, W, 2), VV and VH |
| `s1/sar_ascending_doy.npy` | int16 | (Ta,) |
| `s1/sar_descending.npy` | int16 | (Td, H, W, 2) |
| `s1/sar_descending_doy.npy` | int16 | (Td,) |

The same seven files may also sit directly in the tile directory without the `s2/` and `s1/` subdirectories.

## Run

```bash
export OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_WAIT_POLICY=ACTIVE
./tessera_encode TILE_DIR OUT_DIR student_large.tcw
./tessera_encode_cuda TILE_DIR OUT_DIR student_large.tcw
```

This writes `OUT_DIR/<grid>.npy`, which is (H, W, 128) int8, and `OUT_DIR/<grid>_scales.npy`, which is (H, W) float32. Multiply the int8 codes by the per-pixel scale to recover the embedding. Pixels with no valid observations come out as all-zero codes.

To check a result against a reference tile:

```bash
python tools/compare.py REF_DIR OUT_DIR grid_-1.65_53.35
```

### Tuning

| Variable | Default | Meaning |
|---|---|---|
| `OMP_NUM_THREADS` | all CPUs | Threads per process. On a large node, several processes of about 16 cores each, one per socket slice, give the best throughput. |
| `OMP_WAIT_POLICY` | | Set it to `ACTIVE`. `PASSIVE` is several times slower. |
| `TC_BLOCK` | 4 | Sentinel-1 pixels per thread block (CPU) |
| `TC_BLOCK_S2` | 8 | Sentinel-2 pixels per thread block (CPU) |
| `TC_BATCH` | 4096 | Pixels per call into the back end. Raise it when one process has many threads. |
| `TC_CHUNK` | 64 | Pixels per chunk on the GPU. 64 suits an L4; an A30 prefers 512 to 1024. |
| `TC_PROFILE` | 0 | Set it to 1 to print a per-phase timing profile. |

Hyperthreads do not help on the CPU: both siblings share one AMX unit and one L2, so give each process one thread per physical core.

## Licence

ISC. See `LICENSE`.
