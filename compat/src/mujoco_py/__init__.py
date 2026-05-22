"""Stub for `import mujoco_py` used by old-API repos for rendering only.

We never invoke MuJoCo via mujoco_py — Gymnasium-v5 uses the official `mujoco`
bindings. This stub keeps top-level imports happy; if anything actually tries to
*call* into mujoco_py the attribute access raises a clear error.
"""
import sys as _sys
import types as _types


def _raise_if_called(*args, **kwargs):
    raise RuntimeError(
        "mujoco_py stub: rendering / direct-mujoco-py paths are not supported. "
        "Use Gymnasium-v5 MuJoCo via the compat gym.make() path."
    )


class GlfwContext:
    """Stub for `from mujoco_py import GlfwContext` (used by LDCQ eval scripts
    we do not call)."""

    def __init__(self, *args, **kwargs):
        pass

    def make_context_current(self):
        pass


def load_model_from_path(*args, **kwargs):
    _raise_if_called(*args, **kwargs)


def MjSim(*args, **kwargs):  # noqa: N802
    _raise_if_called(*args, **kwargs)


def MjViewer(*args, **kwargs):  # noqa: N802
    _raise_if_called(*args, **kwargs)


# Register submodules expected by some imports
for _sub in ("builder", "cymj", "version"):
    _sys.modules[f"mujoco_py.{_sub}"] = _types.ModuleType(f"mujoco_py.{_sub}")


__version__ = "2.1.2.14-stub"
