"""Code statistics of PQ2_0 weights: code frequencies, entropy, per-block zero rates."""

import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "third_party" / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402

path = sys.argv[1]
r = gguf.GGUFReader(path)
tot = np.zeros(4, np.int64)
blk_zero_hist = np.zeros(129, np.int64)  # zeros per 128-weight block
n_t = 0
for t in r.tensors:
    if t.tensor_type.name != "PQ2_0" or "token_embd" in t.name:
        continue
    raw = np.asarray(t.data).view(np.uint8).reshape(-1, 34)
    # sample up to 200k blocks per tensor
    if raw.shape[0] > 200_000:
        raw = raw[np.random.default_rng(n_t).choice(raw.shape[0], 200_000, replace=False)]
    qs = raw[:, 2:]
    codes = np.stack([(qs >> (2 * i)) & 3 for i in range(4)], axis=-1).reshape(qs.shape[0], 128)
    tot += np.bincount(codes.ravel(), minlength=4)
    blk_zero_hist += np.bincount((codes == 1).sum(axis=1), minlength=129)
    n_t += 1
p = tot / tot.sum()
H = -sum(x * np.log2(x) for x in p if x > 0)
print(f"tensors {n_t}; code freq (-1,0,+1,+2): {np.round(p, 4)}; entropy {H:.3f} bits/weight")
zb = np.arange(129)
print(f"per-block zero fraction: mean {(blk_zero_hist * zb).sum() / blk_zero_hist.sum() / 128:.3f}")
cum = np.cumsum(blk_zero_hist) / blk_zero_hist.sum()
for thr in (0.25, 0.35, 0.45, 0.55):
    print(f"  blocks with zero fraction >= {thr}: {1 - cum[int(thr * 128) - 1]:.3f}")
