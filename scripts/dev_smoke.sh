#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -x "$REPO_ROOT/build/visionforge" ]]; then
  cmake -S "$REPO_ROOT" -B "$REPO_ROOT/build" -DCMAKE_BUILD_TYPE=Release -DVISIONFORGE_OMP=ON
fi

cmake --build "$REPO_ROOT/build" -j

TMP="$(mktemp -d)"
export TMP
trap 'rm -rf "$TMP"' EXIT

SMOKE_DS="${TMP}/smoke_dataset"
mkdir -p "$SMOKE_DS"
SMOKE_DS="$(cd "$SMOKE_DS" && pwd -P)"

export REPO_ROOT
export SMOKE_DS
export VF_DEV_SMOKE_OUT="$TMP/world_smoke.json"

python3 - <<'PY'
import json
import os

repo_root = os.environ["REPO_ROOT"]
smoke_ds = os.environ["SMOKE_DS"]
out_path = os.environ["VF_DEV_SMOKE_OUT"]

with open(os.path.join(repo_root, "world.json"), "r", encoding="utf-8") as f:
    obj = json.load(f)

obj["dataset"]["root"] = smoke_ds

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2)
    f.write("\n")
PY

"$REPO_ROOT/build/visionforge" forge --config "$TMP/world_smoke.json" --frames 3

python3 scripts/validate_dataset.py --dataset-root "$SMOKE_DS" --check-meta

(
  cd "$REPO_ROOT/python/visionforge_loader"
  if ! PYTHONPATH=. python3 -m pytest -q; then
    if ! python3 -c "import pytest" 2>/dev/null; then
      echo "Install test deps: pip install ./python/visionforge_loader[dev] (see python/visionforge_loader/README.md)." >&2
    fi
    exit 1
  fi
)

if [[ "${VF_SMOKE_SCENARIO:-0}" == "1" ]]; then
  python3 - <<'PY'
import json, os
repo = os.environ["REPO_ROOT"]
tmp  = os.environ["TMP"]
ds   = tmp + "/smoke_scenario"
os.makedirs(ds, exist_ok=True)
with open(repo + "/examples/scenarios/stack_test.json") as f:
    cfg = json.load(f)
cfg["dataset"]["root"] = ds
with open(tmp + "/smoke_scenario.json", "w") as f:
    json.dump(cfg, f, indent=2)
PY
  "$REPO_ROOT/build/visionforge" scenario \
    --config "$TMP/smoke_scenario.json" \
    --name "stack_test" --frames 2
  python3 scripts/validate_dataset.py \
    --dataset-root "$TMP/smoke_scenario" \
    --check-meta \
    --coco "$TMP/smoke_scenario/scenario_coco.json"
  echo "scenario smoke OK"
fi

echo "dev smoke OK"
