#!/bin/bash
# `--gres=gpu:1` reserves one GPU for this job.
# `--exclusive` is stronger: it asks Slurm not to co-schedule other jobs on the allocated node.
#SBATCH --job-name=entropy-suite
#SBATCH --output=%x-%j.out
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: sbatch [slurm options] suite/run_experiment_suite.sbatch <config.json> [runner args...]"
  exit 2
fi

CONFIG_PATH="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

python suite/run_experiment_suite.py --config "$CONFIG_PATH" "$@"
