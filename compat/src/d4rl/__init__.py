"""Fake `d4rl` shim. Implements only what the baselines actually call."""
from compat_utils.d4rl_dict_converter import (
    qlearning_dataset,
    get_dataset,
    get_normalized_score,
)

import sys as _sys
import types as _types

# Some repos try `import d4rl.kitchen_envs` etc. — register no-op submodules.
for _sub in ("locomotion", "kitchen_envs", "infos"):
    _mod = _types.ModuleType(f"d4rl.{_sub}")
    _sys.modules[f"d4rl.{_sub}"] = _mod


def _register():
    """No-op stand-in for d4rl's gym env registration."""
    pass


_register()
