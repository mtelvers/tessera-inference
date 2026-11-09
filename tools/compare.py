"""compare.py REF_DIR TEST_DIR [GRID] - int8 agreement + per-pixel cosine."""
import sys
import numpy as np

ref_dir, test_dir = sys.argv[1], sys.argv[2]
g = sys.argv[3] if len(sys.argv) > 3 else "grid_-1.65_53.35"
a = np.load(f"{ref_dir}/{g}.npy").astype(np.int32)
b = np.load(f"{test_dir}/{g}.npy").astype(np.int32)
sa = np.load(f"{ref_dir}/{g}_scales.npy")
sb = np.load(f"{test_dir}/{g}_scales.npy")
assert a.shape == b.shape, (a.shape, b.shape)
d = np.abs(a - b)
valid = (sa > 0) & (sb > 0) & np.isfinite(sa) & np.isfinite(sb)
na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
ok = valid & (na > 0) & (nb > 0)
cos = (a * b).sum(-1)[ok] / (na[ok] * nb[ok])
print(f"shape={a.shape} valid_px={ok.sum()}/{ok.size}")
print(f"int8 exact={np.mean(d == 0):.1%} off1={np.mean(d == 1):.1%} off2+={np.mean(d >= 2):.1%} max={d.max()}")
print(f"cosine mean={cos.mean():.5f} p1={np.percentile(cos, 1):.5f} min={cos.min():.5f} "
      f"frac<0.99={np.mean(cos < 0.99):.3%}")
rs = np.abs(sa[ok] - sb[ok]) / np.maximum(np.abs(sa[ok]), 1e-12)
print(f"scales rel diff median={np.median(rs):.2e} p99={np.percentile(rs, 99):.2e}")
