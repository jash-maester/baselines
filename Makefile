# Baselines reproduction — operator shortcuts.
#
# Conventions:
#   - All targets are no-GPU unless suffixed `_gpu`.
#   - The MLflow container at localhost:5555 is a shared service; we never
#     bring it up/down from here. We only set MLFLOW_TRACKING_URI.
#   - Experiments: latent_cep_baselines (matrix), latent_cep_baselines_smoke
#     (autoresearch).

VENV         ?= .venv
PY           ?= $(VENV)/bin/python
DATASET_OPT  ?= C
MLFLOW_URI   ?= http://localhost:5555

export MLFLOW_TRACKING_URI := $(MLFLOW_URI)
export DATASET_OPTION       := $(DATASET_OPT)

.PHONY: help status dashboard validate plan \
        smoke smoke-status smoke-one smoke-reset \
        matrix matrix-plan matrix-status matrix-resume \
        ldcq-data notify-test logs-tail \
        clean-state

help:
	@echo "Targets:"
	@echo "  status              - quick read-only dashboard"
	@echo "  dashboard           - live watch dashboard (30s refresh)"
	@echo "  validate            - Minari ID download validation (Option C)"
	@echo "  ldcq-data ENV=<>    - build the LDCQ pickle for one env"
	@echo "  smoke               - run all autoresearch smokes (halts on crash)"
	@echo "  smoke-status        - just print the smoke state table"
	@echo "  smoke-one KEY=<key> - retry a single smoke method"
	@echo "  smoke-reset KEY=<>  - reset a single method to pending (or all)"
	@echo "  smoke-footprints    - per-method vRAM/RAM/GPU% table (after smokes)"
	@echo "  matrix-plan         - print the full long-run manifest"
	@echo "  matrix-status       - print matrix state"
	@echo "  matrix [POOL=,CONCURRENCY=] - launch long_runner (nohup, bin-packing)"
	@echo "  matrix-resume       - resume long_runner where it left off (foreground)"
	@echo "  matrix-reassign PATTERN= POOL= - move cells matching PATTERN into POOL"
	@echo "  matrix-export   POOL= OUT=     - export pool's cells to a portable manifest"
	@echo "  notify-test         - send a test Discord ping"
	@echo "  clean-state         - reset state/*.json (with confirm prompt)"

status: ; @$(PY) scripts/progress_dashboard.py

dashboard: ; @$(PY) scripts/progress_dashboard.py --watch 30

validate: ; @$(PY) scripts/validate_minari.py

ldcq-data:
	@test -n "$(ENV)" || { echo "ENV=<env_d4rl_name> required"; exit 2; }
	$(PY) scripts/prepare_data_ldcq.py --env_d4rl_name $(ENV)

smoke:
	@echo "[smoke] starting autoresearch loop; halts on first crash (rc=42)."
	$(PY) scripts/autoresearch_smoke.py --execute

smoke-status: ; @$(PY) scripts/autoresearch_smoke.py --status

smoke-footprints: ; @$(PY) scripts/autoresearch_smoke.py --footprint_table

smoke-one:
	@test -n "$(KEY)" || { echo "KEY=<algo>:<env>:<stage> required"; exit 2; }
	$(PY) scripts/autoresearch_smoke.py --execute --only "$(KEY)"

smoke-reset:
	@if [ -z "$(KEY)" ]; then \
	  echo "[reset] resetting ALL smoke methods to pending"; \
	  $(PY) scripts/autoresearch_smoke.py --reset; \
	else \
	  $(PY) scripts/autoresearch_smoke.py --reset --only "$(KEY)"; \
	fi

matrix-plan:   ; @$(PY) scripts/long_runner.py --plan
matrix-status: ; @$(PY) scripts/long_runner.py --status

# Launch the overnight matrix. Easy-first ordering by default, footprint-driven
# bin-packing concurrency on (28 GB vRAM, 45 GB system RAM). Discord pings on
# failure only.
#   make matrix                 -> default pool, easy-first
#   make matrix POOL=foo        -> only cells with pool=foo
#   make matrix CONCURRENCY=4   -> override concurrency cap
matrix:
	@test -d logs/matrix || mkdir -p logs/matrix
	@test -d state || mkdir -p state
	@if [ -f state/long_runner.pid ] && kill -0 $$(cat state/long_runner.pid) 2>/dev/null; then \
	  echo "[matrix] long_runner already running with pid $$(cat state/long_runner.pid)"; \
	  exit 1; \
	fi
	@CONCURRENCY=$${CONCURRENCY:-12}; POOL=$${POOL:-default}; \
	 nohup env PYTHONUNBUFFERED=1 $(PY) scripts/long_runner.py --execute --pool "$$POOL" --max_concurrency "$$CONCURRENCY" \
	  > logs/matrix_runner.log 2>&1 & echo $$! > state/long_runner.pid; \
	 echo "[matrix] launched pid=$$(cat state/long_runner.pid) pool=$$POOL concurrency=$$CONCURRENCY"; \
	 echo "         tail: tail -f logs/matrix_runner.log    dashboard: make dashboard"

matrix-resume:
	$(PY) scripts/long_runner.py --execute

# Multi-server split helpers
#   make matrix-reassign PATTERN=qgpo POOL=server2
#   make matrix-export POOL=server2 OUT=state/matrix-server2.json
#   (scp state/matrix-server2.json to the other box; run there with
#    `python scripts/long_runner.py --execute --pool server2` against its
#    copy of state/matrix.json).
matrix-reassign:
	@test -n "$(PATTERN)" || { echo "PATTERN=<substr> required"; exit 2; }
	@test -n "$(POOL)"    || { echo "POOL=<name> required"; exit 2; }
	$(PY) scripts/long_runner.py --reassign "$(PATTERN)" --to_pool "$(POOL)"

matrix-export:
	@test -n "$(POOL)" || { echo "POOL=<name> required"; exit 2; }
	@test -n "$(OUT)"  || { echo "OUT=<path> required"; exit 2; }
	$(PY) scripts/long_runner.py --export_pool --pool "$(POOL)" --out "$(OUT)"

notify-test:
	$(PY) scripts/notify.py --level info --task "baselines/test" --msg "make notify-test ping ($$(date -Is))"

logs-tail:
	@ls -t logs/matrix/*.log 2>/dev/null | head -1 | xargs -I {} sh -c 'echo "==> {} <=="; tail -n 80 {}'

clean-state:
	@echo "About to remove state/*.json — current state:" ; ls state 2>/dev/null
	@read -p "Type yes to confirm: " a && [ "$$a" = "yes" ] && rm -fv state/*.json || echo "aborted"
