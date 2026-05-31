"""wandb shim: intercept upstream `wandb.log(...)` calls and forward to MLflow.

CORL / EDA / DQL / Diffuser / LDCQ all import wandb and call wandb.log() in
their hot training loops. We install this no-op module under the name `wandb`
in sys.modules before those imports run; calls land here instead.

We *do not* use the real wandb package — nothing leaves the box. All telemetry
ends up in the MLflow experiment opened by the launcher's `mlflow_start(...)`.

Performance note: forwarding each `wandb.log({...})` to `mlflow.log_metric()`
synchronously round-trips to the Postgres-backed MLflow server. With 8 cells
running concurrently and BC calling `wandb.log(...)` every update step, that
serialized ~100 REST calls/sec through one gunicorn worker — capping BC at
~12 update-steps/sec per cell despite a near-idle GPU. Fix: buffer metric
points and flush in batches via `MlflowClient.log_batch` (≤1000 metrics per
call), every `FLUSH_EVERY_N` points or `FLUSH_EVERY_SEC` seconds, whichever
comes first. `finish()` flushes the tail.
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

_run_active = False
_current_run = None

_client = None
_buffer: list = []
_last_flush = 0.0
FLUSH_EVERY_N = 500
FLUSH_EVERY_SEC = 2.0

# Reference scores for the active run (set by the launcher via set_refs). When a
# `d4rl_normalized_score*` metric is forwarded we also emit the inverted
# `raw_return*` so raw + normalized are both logged. (raw = norm/100*(max-min)+min)
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
        # No active MLflow run — drop the buffer rather than letting it grow.
        _buffer = []
        return
    pending, _buffer = _buffer, []
    try:
        for i in range(0, len(pending), 1000):
            client.log_batch(run_id=run_id, metrics=pending[i:i + 1000])
    except Exception:
        # MLflow occasionally rejects a single bad point; don't crash training.
        pass


def init(*args, **kwargs):
    global _run_active, _current_run, _last_flush
    _run_active = True
    _last_flush = time.time()
    run = _StubRun()
    _current_run = run
    mod = sys.modules.get("wandb")
    if mod is not None:
        mod.run = run
    return run


class _StubRun:
    def __init__(self):
        self.id = "stub"
        self.name = "stub"
        self.dir = "."
        self.config = _ConfigStub()

    def log(self, metrics, step=None, **kwargs):
        # Forward run.log(...) to the module-level MLflow forwarder. Without
        # this, __getattr__ returns a no-op and silently drops every metric
        # from repos that log through the run object (e.g. EDA's
        # `args.run.log({...})`, and `wandb.run.log(...)`) rather than the
        # module-level `wandb.log()` — yielding empty MLflow runs. `log` here
        # resolves to the module-level function defined below at call time.
        log(metrics, step=step, **kwargs)

    def __getattr__(self, name):
        return lambda *a, **kw: None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def finish(self, *args, **kwargs):
        global _run_active
        _flush()
        _run_active = False


class _ConfigStub(dict):
    """Mimics wandb.config — dict-ish with attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return None

    def __setattr__(self, name, value):
        self[name] = value

    def update(self, *args, **kwargs):
        new_args = []
        for a in args:
            if hasattr(a, "__dict__") and not isinstance(a, dict):
                new_args.append(vars(a))
            else:
                new_args.append(a)
        super().update(*new_args, **kwargs)


def log(metrics, step=None, **kwargs):
    """Buffer wandb-style metric dicts; flush in batches to MLflow."""
    if not _run_active or _Metric is None:
        return
    if not isinstance(metrics, dict):
        return
    ts_ms = int(time.time() * 1000)
    step_int = int(step) if step is not None else 0
    for k, v in metrics.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue  # non-scalar (image, histogram, ndarray) — skip
        key = str(k).replace("/", ".")
        _buffer.append(_Metric(key, fv, ts_ms, step_int))
        # also emit the raw return alongside any normalized score
        if ("d4rl_normalized_score" in key and _ref_min is not None
                and _ref_max is not None and _ref_max != _ref_min):
            raw = fv / 100.0 * (_ref_max - _ref_min) + _ref_min
            _buffer.append(_Metric(key.replace("d4rl_normalized_score", "raw_return"),
                                   raw, ts_ms, step_int))
    if len(_buffer) >= FLUSH_EVERY_N or (time.time() - _last_flush) > FLUSH_EVERY_SEC:
        _flush()


def finish(*args, **kwargs):
    global _run_active, _current_run
    _flush()
    _run_active = False
    _current_run = None
    mod = sys.modules.get("wandb")
    if mod is not None:
        mod.run = None


def login(*args, **kwargs):
    return True


def watch(*args, **kwargs):
    return None


def save(*args, **kwargs):
    return None


def define_metric(*args, **kwargs):
    return None


class Image:
    def __init__(self, *args, **kwargs):
        pass


class Video:
    def __init__(self, *args, **kwargs):
        pass


class Histogram:
    def __init__(self, *args, **kwargs):
        pass


def install():
    """Install as `wandb` in sys.modules so `import wandb` resolves here."""
    if "wandb" in sys.modules and getattr(sys.modules["wandb"], "_is_compat_stub", False):
        return
    mod = types.ModuleType("wandb")
    mod._is_compat_stub = True
    mod.init = init
    mod.log = log
    mod.finish = finish
    mod.login = login
    mod.watch = watch
    mod.save = save
    mod.define_metric = define_metric
    mod.Image = Image
    mod.Video = Video
    mod.Histogram = Histogram
    mod.config = _ConfigStub()
    mod.run = None
    sys.modules["wandb"] = mod
