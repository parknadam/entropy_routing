# Experiment Suite

This directory contains a small orchestration layer for long-running, repeated
multi-model experiments.

What it gives us:

- One JSON config can describe `experiments x models x seeds`.
- Optional long-lived servers such as vLLM stay up across related tasks.
- Logs and suite state are written under one run root, so large sweeps can resume.
- When launched inside a Slurm allocation, GPU ownership stays with your job even
  if a server process is restarted between tasks.

## Recommended workflow

1. Reserve GPUs with Slurm, preferably exclusive:

```bash
sbatch --partition=<partition> --gres=gpu:<N> --time=48:00:00 --exclusive \
  suite/run_experiment_suite.sbatch suite/template_experiment_suite.json --dry-run
```

2. Replace `--dry-run` with the real run once the resolved plan looks correct.

3. Resume an interrupted sweep with the same `--run-root`:

```bash
python suite/run_experiment_suite.py \
  --config suite/template_experiment_suite.json \
  --run-root /path/to/existing/run_root \
  --resume
```

## Config shape

The runner reads JSON only, so it stays dependency-free inside the repo's venv.

Top-level keys:

- `name`: suite name for logs/run roots
- `defaults`: shared `cwd`, `env`, and seed policy
- `models`: model-specific variables, env, and optional server assignment
- `servers`: optional server lifecycle blocks, usually for vLLM
- `experiments`: commands to run for each model/seed pair

Template variables are rendered with Python `str.format_map()`. Useful built-ins:

- `{repo_root}`
- `{run_root}`
- `{suite_name}`
- `{model_id}`
- `{experiment_id}`
- `{seed}`
- `{repeat_index}`
- Any keys inside `defaults.vars`, `models.<id>.vars`, or `experiments[].vars`

## Fair repeated runs

The training entrypoints in this repo now accept `--seed` and use it for:

- Python RNG
- NumPy RNG when available
- Torch RNG
- Dataset shuffle seed
- Hugging Face `TrainingArguments.seed` and `data_seed`

That means the suite can safely expand seeds such as `0..4` for repeated runs
without all five jobs silently using the same implicit randomness.

## Notes on vLLM stability

If a task uses a `server` block, the runner keeps that server process alive until
the next task needs a different server instance. This reduces repeated cold-starts.

More importantly, if the suite itself runs under `sbatch` or `salloc`, the GPU
reservation belongs to the Slurm job, not to the vLLM process. So even if the
runner stops and restarts vLLM between experiments, other users cannot jump onto
those GPUs while your allocation is still active.
