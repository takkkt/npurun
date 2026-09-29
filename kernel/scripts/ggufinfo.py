"""Print GGUF metadata (architecture keys) and unique tensor shapes/types."""

import collections
import re
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "third_party" / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402

KEYS = ("block_count", "embedding_length", "feed_forward", "head_count", "ssm", "full_attention",
        "rope", "vocab_size", "hadamard", "prism", "context_length", "key_length", "value_length",
        "general.architecture", "linear", "interval")

for p in sys.argv[1:]:
    r = gguf.GGUFReader(p)
    print("==", p)
    for k, v in r.fields.items():
        if any(s in k for s in KEYS):
            try:
                val = v.contents()
            except Exception:
                val = "?"
            if isinstance(val, list) and len(val) > 8:
                val = f"{val[:8]}... (len {len(val)})"
            print("  ", k, "=", val)
    seen = collections.OrderedDict()
    for t in r.tensors:
        key = (re.sub(r"blk\.\d+\.", "blk.N.", t.name), t.tensor_type.name, tuple(int(x) for x in t.shape))
        seen[key] = seen.get(key, 0) + 1
    for (k, ty, sh), n in seen.items():
        print(f"   {k:42s} {ty:7s} {str(sh):24s} x{n}")
