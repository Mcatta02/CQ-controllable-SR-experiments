#!/usr/bin/env python3
"""
Show what is inside a checkpoint: which encoder, how many generator blocks,
and how many parameters each part has.

  python inspect_checkpoint.py ~/CQ-orig/base.pth

Read the output like this:
  1. "Config stored in the checkpoint" -> encoder name, number of layers, T_max (if saved).
  2. "Parameters per top-level module"  -> the encoder (EDSR) is large (millions),
     the recurrent predictor is small (thousands). The predictor count is the number
     to put in the paper (compare with the 6,751 in the current draft).
"""
import sys
import collections
import torch

path = sys.argv[1]
ck = torch.load(path, map_location="cpu", weights_only=False)
print("file:", path)
print("top-level type:", type(ck).__name__)
if isinstance(ck, dict):
    print("top-level keys:", list(ck.keys())[:30])


def find_state_dict(o):
    """largest dict whose values are all tensors"""
    best, stack = None, [o]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            if x and all(torch.is_tensor(v) for v in x.values()):
                if best is None or len(x) > len(best):
                    best = x
            else:
                stack.extend(x.values())
    return best


def show_config(o, depth=0, key=""):
    """print small non-tensor entries (names, ints, strings) that describe the model"""
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                print("  " * depth + f"{key}/{k}: {v}")
            elif isinstance(v, (list, tuple)) and len(v) < 10 and all(isinstance(i, (str, int, float)) for i in v):
                print("  " * depth + f"{key}/{k}: {v}")
            elif isinstance(v, dict) and not (v and all(torch.is_tensor(i) for i in v.values())):
                show_config(v, depth, f"{key}/{k}")


print("\nConfig stored in the checkpoint:")
show_config(ck)

sd = find_state_dict(ck)
if sd is None:
    sys.exit("no state dict found")
total = sum(v.numel() for v in sd.values())
print(f"\nTotal parameters in state dict: {total:,}")

for depth in (1, 2):
    c = collections.Counter()
    for k, v in sd.items():
        c[".".join(k.split(".")[:depth])] += v.numel()
    print(f"\nParameters per module (first {depth} name part(s)):")
    for k, n in sorted(c.items(), key=lambda kv: -kv[1])[:25]:
        print(f"  {k:50s} {n:>12,}")
