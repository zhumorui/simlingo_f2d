#!/usr/bin/env bash
# Source this script to configure paths for simlingo_f2d.
# Usage:  source env_vars.sh
#
# Works in bash and zsh.

LOCAL="$(realpath "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")")"
CARLA="$LOCAL/f2d_carla"      # symlink created by setup (see README)

export WORK_DIR=$LOCAL
export CARLA_ROOT=$CARLA
export LEADERBOARD_ROOT=$LOCAL/leaderboard
export SCENARIO_RUNNER_ROOT=$LOCAL/scenario_runner
export PYTHONPATH=$CARLA/PythonAPI/carla:$LOCAL/leaderboard:$LOCAL/scenario_runner
# Prevent UE4 Signal 11 segfault when multiple GPU drivers coexist
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# Also persist into the conda env for future activations
conda env config vars set WORK_DIR=$LOCAL              -n simlingo_f2d 2>/dev/null || true
conda env config vars set CARLA_ROOT=$CARLA            -n simlingo_f2d 2>/dev/null || true
conda env config vars set LEADERBOARD_ROOT=$LOCAL/leaderboard    -n simlingo_f2d 2>/dev/null || true
conda env config vars set SCENARIO_RUNNER_ROOT=$LOCAL/scenario_runner -n simlingo_f2d 2>/dev/null || true
conda env config vars set PYTHONPATH=$CARLA/PythonAPI/carla:$LOCAL/leaderboard:$LOCAL/scenario_runner -n simlingo_f2d 2>/dev/null || true
conda env config vars set VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json -n simlingo_f2d 2>/dev/null || true

echo "simlingo_f2d environment ready."
echo "  CARLA_ROOT            = $CARLA_ROOT"
echo "  LEADERBOARD_ROOT      = $LEADERBOARD_ROOT"
echo "  SCENARIO_RUNNER_ROOT  = $SCENARIO_RUNNER_ROOT"
echo "  WORK_DIR              = $WORK_DIR"
