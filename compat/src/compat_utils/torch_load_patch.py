"""PyTorch >= 2.6 made weights_only=True the default; many baseline checkpoints
serialize Python objects and fail to load. Monkey-patch to restore old behavior."""
import torch as _torch

_original_load = _torch.load
_PATCHED = False


def _patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)


def apply():
    global _PATCHED
    if _PATCHED:
        return
    _torch.load = _patched_load
    _PATCHED = True
