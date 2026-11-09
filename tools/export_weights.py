"""Convert a Tessera v2 pixel-student checkpoint to the flat .tcw file the encoder reads.

usage: python export_weights.py CHECKPOINT.pt OUT.tcw

Format: b"TCW1", u32 tensor count, then per tensor: u32 name_len, name,
u32 ndim, i64 dims[ndim], f32 data (row-major, little-endian).

Only torch is needed; the model code is not imported.
"""
import os
import struct
import sys

import torch

if len(sys.argv) != 3:
    sys.exit(__doc__.strip())
src, out = sys.argv[1], sys.argv[2]

payload = torch.load(src, map_location="cpu", weights_only=False)
sd = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
sd = {k: v.detach().float().contiguous() for k, v in sd.items()}

args = payload.get("args", {}) if isinstance(payload, dict) else {}
expect = {"latent_dim": 160, "num_layers": 4, "nhead": 4, "dim_feedforward": 2560, "repr_dim": 128,
          "enable_qk_norm": False}
for k, v in expect.items():
    if k in args and type(v)(args[k]) != v:
        sys.exit(f"{src}: {k}={args[k]}, but the encoder is built for {k}={v} (the v2 Large student)")

with open(out, "wb") as f:
    f.write(b"TCW1")
    f.write(struct.pack("<I", len(sd)))
    for k, v in sd.items():
        f.write(struct.pack("<I", len(k)))
        f.write(k.encode())
        f.write(struct.pack("<I", v.dim()))
        f.write(struct.pack(f"<{v.dim()}q", *v.shape))
        f.write(v.numpy().tobytes())
print(f"wrote {out}: {len(sd)} tensors, {os.path.getsize(out) / 1e6:.1f} MB")
