"""Decode an NPU transaction stream (insts.bin) into readable ops; optionally diff two."""

import sys

import numpy as np

OPS = {0: "WRITE", 1: "BLOCKWRITE", 2: "BLOCKSET", 3: "MASKWRITE", 4: "MASKPOLL", 5: "NOOP",
       6: "PREEMPT", 128: "TCT(sync)", 129: "DDR_PATCH"}


def decode(words):
    w = np.asarray(words, np.uint32)
    out = [f"header {[hex(x) for x in w[:4]]} ops={w[2]} bytes={w[3]}"]
    i = 4
    while i < len(w):
        op = int(w[i]) & 0xFF
        name = OPS.get(op, f"op{op}")
        if op == 0:
            n = 6
            out.append(f"{i:5d} WRITE addr={w[i+2]:#x} val={w[i+4]:#x}")
        elif op == 1:
            n = int(w[i + 3]) // 4
            data = " ".join(f"{x:#x}" for x in w[i + 4 : i + n])
            out.append(f"{i:5d} BLOCKWRITE col/row={w[i+1]:#x} addr={w[i+2]:#x} data=[{data}]")
        elif op in (3, 4):
            n = 7
            out.append(f"{i:5d} {name} addr={w[i+2]:#x} val={w[i+4]:#x} mask={w[i+5]:#x}")
        elif op == 128:
            n = int(w[i + 1]) // 4
            out.append(f"{i:5d} SYNC {w[i+2]:#x} {w[i+3]:#x}")
        elif op == 129:
            n = int(w[i + 1]) // 4
            plus = int(w[i + 10]) | (int(w[i + 11]) << 32)
            out.append(f"{i:5d} DDR_PATCH reg={w[i+6]:#x} arg={w[i+8]} plus={plus:#x}")
        else:
            n = int(w[i + 1]) // 4 if w[i + 1] else 1
            out.append(f"{i:5d} {name} raw={[hex(x) for x in w[i:i+4]]}")
        i += max(n, 1)
    return out


if __name__ == "__main__":
    a = decode(np.fromfile(sys.argv[1], np.uint32))
    if len(sys.argv) > 2:
        b = decode(np.fromfile(sys.argv[2], np.uint32))
        for x, y in zip(a, b):
            if x != y:
                print("A:", x)
                print("B:", y)
        if len(a) != len(b):
            print(f"length differs: {len(a)} vs {len(b)}")
    else:
        print("\n".join(a))
