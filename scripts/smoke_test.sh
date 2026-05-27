#!/usr/bin/env bash
# smoke_test.sh — end-to-end health check for the VisionForge pipeline.
#
# Usage:
#   bash scripts/smoke_test.sh
#
# Runs from the repo root regardless of where it is invoked from.
# Requires: cmake, python3, pytest.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
STEP=0
step() {
    STEP=$((STEP + 1))
    echo ""
    echo "══════════════════════════════════════════════════════════════════"
    printf  "  Step %d: %s\n" "$STEP" "$*"
    echo "══════════════════════════════════════════════════════════════════"
}

fail() {
    echo "" >&2
    echo "✗  SMOKE TEST FAILED: $*" >&2
    exit 1
}

# Temp dir for patched configs and transient output — cleaned up on exit
SMOKE_OUT="$(mktemp -d)"
export SMOKE_OUT
trap 'rm -rf "$SMOKE_OUT"' EXIT

# --------------------------------------------------------------------------- #
# (a) Build
# --------------------------------------------------------------------------- #
step "Building VisionForge"
cmake --build build --parallel \
    || fail "cmake build failed"

# --------------------------------------------------------------------------- #
# (b) C++ unit tests
# --------------------------------------------------------------------------- #
step "Running C++ unit tests"
for t in test_world_config test_meta_pose test_dataset_manifest test_flow_math; do
    exe="./build/$t"
    if [[ ! -x "$exe" ]]; then
        fail "C++ test binary not found: $exe  (was it added to CMakeLists.txt?)"
    fi
    echo "  → $t"
    "$exe" || fail "$t failed"
done

# --------------------------------------------------------------------------- #
# (c) Forge smoke dataset — patch dataset.root to $SMOKE_OUT/forge
# --------------------------------------------------------------------------- #
step "Generating forge smoke dataset (4 frames, 64x64)"

python3 - <<'PY'
import json, os
s = os.environ["SMOKE_OUT"]
cfg = json.load(open("tests/fixtures/smoke_world.json"))
cfg["dataset"]["root"] = s + "/forge"
json.dump(cfg, open(s + "/smoke_world.json", "w"), indent=2)
PY

./build/visionforge forge \
    --config  "${SMOKE_OUT}/smoke_world.json" \
    --frames  4    \
    --width   64   \
    --height  64   \
    --spp     2    \
    --verbose \
    || fail "forge mode failed"

# --------------------------------------------------------------------------- #
# (d) Scenario smoke — 2-frame camera trajectory, patch dataset.root
# --------------------------------------------------------------------------- #
step "Generating scenario smoke dataset (2 frames, trajectory)"

python3 - <<'PY'
import json, os
s = os.environ["SMOKE_OUT"]
cfg = json.load(open("tests/fixtures/smoke_scenario_trajectory.json"))
cfg["dataset"]["root"] = s + "/scenario"
json.dump(cfg, open(s + "/smoke_scenario.json", "w"), indent=2)
PY

./build/visionforge scenario \
    --config "${SMOKE_OUT}/smoke_scenario.json" \
    --name   "SmokeTrajectory"                  \
    --frames 2    \
    --width  64   \
    --height 64   \
    --spp    2    \
    || fail "scenario mode failed"

# --------------------------------------------------------------------------- #
# (e) Validate forge output
# --------------------------------------------------------------------------- #
step "Validating forge dataset"
python3 scripts/validate_dataset.py \
    --dataset-root "${SMOKE_OUT}/forge" \
    --check-meta \
    || fail "forge dataset validation failed"

# --------------------------------------------------------------------------- #
# (f) Python test suite (includes test_flow_e2e.py)
#     Run with PYTHONPATH=. from the package root so the package resolves
#     without needing a system-wide pip install (mirrors dev_smoke.sh).
# --------------------------------------------------------------------------- #
step "Running Python test suite"
(
    cd python/visionforge_loader
    PYTHONPATH=. python3 -m pytest tests/ -v
) || fail "Python test suite failed"

# --------------------------------------------------------------------------- #
# Done
# --------------------------------------------------------------------------- #
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  ALL SMOKE TESTS PASSED"
echo "══════════════════════════════════════════════════════════════════"
