"""
Prints a full Python traceback the first N times a GPU->CPU sync happens
(torch.Tensor.item / __bool__ / __index__ / .tolist()), then stops patching.

Call install_sync_tracer() right before the code you actually want to
inspect (NOT at module import time) — otherwise unrelated syncs from
DataLoader startup / dataset preprocessing / your own print statements
will eat up the hit budget before the interesting code ever runs.

Pass a `only_if` substring (e.g. "CQ-controllable-SR-experiments/models")
to ignore syncs whose traceback doesn't pass through your model code at
all (e.g. torchvision's dataset-side normalize running in a DataLoader
worker) — this filters out noise from files outside your project.
"""
import traceback
import torch

_orig_item = torch.Tensor.item
_orig_tolist = torch.Tensor.tolist
_orig_bool = torch.Tensor.__bool__
_state = {"hits": 0, "max_hits": 5, "only_if": None}


def _traced(orig_fn, label):
    def wrapper(self, *args, **kwargs):
        if _state["hits"] < _state["max_hits"]:
            stack = traceback.format_stack(limit=20)
            if _state["only_if"] is None or any(_state["only_if"] in line for line in stack):
                _state["hits"] += 1
                print(f"\n=== sync #{_state['hits']} via {label} (shape={tuple(self.shape)}) ===")
                print("".join(stack))
        return orig_fn(self, *args, **kwargs)
    return wrapper


def install_sync_tracer(max_hits=5, only_if=None):
    """Call this immediately before the code block you want to inspect."""
    _state["hits"] = 0
    _state["max_hits"] = max_hits
    _state["only_if"] = only_if
    torch.Tensor.item = _traced(_orig_item, "Tensor.item()")
    torch.Tensor.tolist = _traced(_orig_tolist, "Tensor.tolist()")
    torch.Tensor.__bool__ = _traced(_orig_bool, "Tensor.__bool__ (implicit if/while on a tensor)")


def uninstall_sync_tracer():
    torch.Tensor.item = _orig_item
    torch.Tensor.tolist = _orig_tolist
    torch.Tensor.__bool__ = _orig_bool