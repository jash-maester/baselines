"""TensorBoard SummaryWriter -> MLflow shim.

QGPO/CEP (`Offline_RL_2D/train_{behavior,critic}.py`) and Diffusion-QL
(`agents/ql_diffusion.py`) log training/eval scalars through
`torch.utils.tensorboard.SummaryWriter.add_scalar/add_scalars`. Those land in
local TB event files and never reach MLflow. We inject a drop-in SummaryWriter
that forwards scalars to `mlflow.log_metric` (buffered via MlflowClient.log_batch,
mirroring `wandb_stub`) on the run the launcher already opened with
`mlflow_start(...)`. Telemetry only — no training math is touched.

Install BEFORE the repo imports tensorboard (the launcher calls
`install()` right after `install_wandb_stub()`):

    from compat_utils import tb_shim
    tb_shim.install()
"""
import sys
import time
import types

try:
    import mlflow as _mlflow
    from mlflow.tracking import MlflowClient as _MlflowClient
    from mlflow.entities import Metric as _Metric
except Exception:  # pragma: no cover
    _mlflow = None
    _MlflowClient = None
    _Metric = None

_client = None
_buffer: list = []
_last_flush = 0.0
FLUSH_EVERY_N = 500
FLUSH_EVERY_SEC = 2.0

# Reference scores for the active run (set by the launcher via set_refs). When a
# `d4rl_normalized_score*` scalar is forwarded we also emit the inverted
# `raw_return*` so raw + normalized are both logged.
_ref_min = None
_ref_max = None


def set_refs(ref_min, ref_max) -> None:
    global _ref_min, _ref_max
    _ref_min = float(ref_min) if ref_min is not None else None
    _ref_max = float(ref_max) if ref_max is not None else None


def _get_client():
    global _client
    if _client is None and _MlflowClient is not None:
        _client = _MlflowClient()
    return _client


def _active_run_id():
    if _mlflow is None:
        return None
    run = _mlflow.active_run()
    return run.info.run_id if run else None


def _flush():
    global _buffer, _last_flush
    _last_flush = time.time()
    if not _buffer:
        return
    run_id = _active_run_id()
    client = _get_client()
    if run_id is None or client is None:
        _buffer = []
        return
    pending, _buffer = _buffer, []
    try:
        for i in range(0, len(pending), 1000):
            client.log_batch(run_id=run_id, metrics=pending[i:i + 1000])
    except Exception:
        # MLflow occasionally rejects a single bad point; never crash training.
        pass


def _record(tag, value, step):
    if _Metric is None:
        return
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return  # non-scalar (tensor/ndarray/None) — skip
    key = str(tag).replace("/", ".")
    ts_ms = int(time.time() * 1000)
    step_i = int(step) if step is not None else 0
    _buffer.append(_Metric(key, fv, ts_ms, step_i))
    # also emit the raw return alongside any normalized score
    if ("d4rl_normalized_score" in key and _ref_min is not None
            and _ref_max is not None and _ref_max != _ref_min):
        raw = fv / 100.0 * (_ref_max - _ref_min) + _ref_min
        _buffer.append(_Metric(key.replace("d4rl_normalized_score", "raw_return"),
                               raw, ts_ms, step_i))
    if len(_buffer) >= FLUSH_EVERY_N or (time.time() - _last_flush) > FLUSH_EVERY_SEC:
        _flush()


class SummaryWriter:
    """Drop-in for torch.utils.tensorboard.SummaryWriter; forwards to MLflow."""

    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, tag, scalar_value, global_step=None, *a, **k):
        _record(tag, scalar_value, global_step)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, *a, **k):
        if isinstance(tag_scalar_dict, dict):
            for k2, v in tag_scalar_dict.items():
                _record(f"{main_tag}/{k2}", v, global_step)

    # Everything else TB exposes is a no-op (histograms/images/graph/text/hparams).
    def add_histogram(self, *a, **k): pass
    def add_image(self, *a, **k): pass
    def add_images(self, *a, **k): pass
    def add_figure(self, *a, **k): pass
    def add_text(self, *a, **k): pass
    def add_graph(self, *a, **k): pass
    def add_hparams(self, *a, **k): pass
    def add_pr_curve(self, *a, **k): pass

    def flush(self, *a, **k): _flush()
    def close(self, *a, **k): _flush()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        _flush()


def install():
    """Register the shim at `torch.utils.tensorboard` (and `tensorboardX`).

    `from torch.utils.tensorboard import SummaryWriter` resolves via sys.modules,
    so seeding it here before the repo import makes the import return our shim.
    """
    for name in ("torch.utils.tensorboard", "tensorboardX"):
        existing = sys.modules.get(name)
        if existing is not None and getattr(existing, "_is_compat_stub", False):
            continue
        mod = types.ModuleType(name)
        mod._is_compat_stub = True
        mod.SummaryWriter = SummaryWriter
        sys.modules[name] = mod
    # Make `torch.utils.tensorboard` reachable as an attribute too, so code that
    # does `import torch; torch.utils.tensorboard.SummaryWriter` resolves.
    try:
        import torch.utils as _tu
        _tu.tensorboard = sys.modules["torch.utils.tensorboard"]
    except Exception:
        pass
