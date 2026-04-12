#!/usr/bin/env python3
"""
Config-driven experiment suite runner.

This script expands experiments x models x seeds from a JSON config, keeps
optional long-lived servers alive across related tasks, and writes enough state
to resume a large sweep safely inside a Slurm GPU allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import urlopen


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def slugify(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-") or "suite"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def render_value(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        try:
            return value.format_map(context)
        except KeyError as exc:
            missing = exc.args[0]
            raise KeyError(f"Missing template variable '{missing}' while rendering '{value}'") from exc
    if isinstance(value, list):
        return [render_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: render_value(item, context) for key, item in value.items()}
    return value


def merge_env(context: Dict[str, Any], *env_layers: Optional[Dict[str, Any]]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for layer in env_layers:
        if not layer:
            continue
        rendered = render_value(layer, context)
        for key, value in rendered.items():
            merged[str(key)] = str(value)
    return merged


def comma_set(raw: Optional[str]) -> Optional[set]:
    if not raw:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def resolve_seeds(defaults: Dict[str, Any], experiment: Dict[str, Any]) -> List[int]:
    if "seeds" in experiment:
        return [int(seed) for seed in experiment["seeds"]]
    if "seeds" in defaults:
        return [int(seed) for seed in defaults["seeds"]]

    repeats = int(experiment.get("repeats", defaults.get("repeats", 1)))
    seed_start = int(experiment.get("seed_start", defaults.get("seed_start", 0)))
    return list(range(seed_start, seed_start + repeats))


def healthcheck_url(url: str, timeout_sec: float) -> bool:
    try:
        with urlopen(url, timeout=timeout_sec) as response:
            return 200 <= getattr(response, "status", 200) < 300
    except URLError:
        return False
    except Exception:
        return False


def write_log_header(log_path: Path, title: str, command: str) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{now_iso()}] {title}\n")
        handle.write(f"$ {command}\n")


def terminate_process_group(proc: subprocess.Popen, grace_sec: float = 15.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.time() + grace_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.5)

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait(timeout=5)


def run_foreground_command(
    command: str,
    cwd: str,
    env: Dict[str, str],
    log_path: Path,
    timeout_sec: Optional[float],
) -> Dict[str, Any]:
    write_log_header(log_path, "task", command)
    with log_path.open("a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            cwd=cwd,
            env={**os.environ, **env},
            stdout=handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        timed_out = False
        started = time.time()
        try:
            return_code = proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_group(proc)
            return_code = proc.returncode if proc.returncode is not None else -9
        duration = time.time() - started
    return {
        "return_code": int(return_code),
        "timed_out": timed_out,
        "duration_sec": round(duration, 2),
    }


class ManagedServer:
    def __init__(self, spec: Dict[str, Any], log_path: Path):
        self.name = spec["name"]
        self.instance_key = spec["instance_key"]
        self.command = spec["start_command"]
        self.stop_command = spec.get("stop_command")
        self.healthcheck_url = spec.get("healthcheck_url")
        self.healthcheck_command = spec.get("healthcheck_command")
        self.cwd = spec["cwd"]
        self.env = spec["env"]
        self.startup_timeout_sec = float(spec.get("startup_timeout_sec", 600))
        self.startup_interval_sec = float(spec.get("startup_interval_sec", 2))
        self.startup_grace_sec = float(spec.get("startup_grace_sec", 5))
        self.shutdown_timeout_sec = float(spec.get("shutdown_timeout_sec", 20))
        self.log_path = log_path
        self.process: Optional[subprocess.Popen] = None

    def start(self) -> None:
        write_log_header(self.log_path, f"server:{self.name}", self.command)
        handle = self.log_path.open("a", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                ["/bin/bash", "-lc", self.command],
                cwd=self.cwd,
                env={**os.environ, **self.env},
                stdout=handle,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
        finally:
            handle.close()

        if not self.healthcheck_url and not self.healthcheck_command:
            time.sleep(self.startup_grace_sec)
            if self.process.poll() is not None:
                raise RuntimeError(f"server '{self.name}' exited before becoming ready")
            return

        deadline = time.time() + self.startup_timeout_sec
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"server '{self.name}' exited before passing health checks")

            if self.healthcheck_url and healthcheck_url(self.healthcheck_url, timeout_sec=2.0):
                return

            if self.healthcheck_command:
                result = run_foreground_command(
                    self.healthcheck_command,
                    cwd=self.cwd,
                    env=self.env,
                    log_path=self.log_path,
                    timeout_sec=15,
                )
                if result["return_code"] == 0:
                    return

            time.sleep(self.startup_interval_sec)

        raise RuntimeError(f"server '{self.name}' did not become ready within {self.startup_timeout_sec}s")

    def stop(self) -> None:
        if self.stop_command:
            run_foreground_command(
                self.stop_command,
                cwd=self.cwd,
                env=self.env,
                log_path=self.log_path,
                timeout_sec=60,
            )

        if self.process is not None and self.process.poll() is None:
            terminate_process_group(self.process, grace_sec=self.shutdown_timeout_sec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a config-driven multi-model experiment suite.")
    parser.add_argument("--config", required=True, help="Path to the suite JSON config.")
    parser.add_argument("--run-root", default=None, help="Directory for logs/state. Defaults to suite_runs/<timestamp>.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve tasks and print them without executing.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing run-root/state file.")
    parser.add_argument("--only-models", default=None, help="Comma-separated subset of model ids to run.")
    parser.add_argument("--only-experiments", default=None, help="Comma-separated subset of experiment ids to run.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed task.")
    return parser.parse_args()


def build_context(
    suite_name: str,
    repo_root: Path,
    run_root: Path,
    defaults: Dict[str, Any],
    model_id: str,
    model_cfg: Dict[str, Any],
    experiment: Dict[str, Any],
    seed: int,
    repeat_index: int,
) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    context.update(defaults.get("vars", {}))
    context.update(model_cfg.get("vars", {}))
    context.update(experiment.get("vars", {}))
    context.update(
        {
            "suite_name": suite_name,
            "repo_root": str(repo_root),
            "run_root": str(run_root),
            "model_id": model_id,
            "experiment_id": experiment["id"],
            "seed": seed,
            "repeat_index": repeat_index,
        }
    )
    return context


def resolve_server_spec(
    config: Dict[str, Any],
    defaults: Dict[str, Any],
    model_cfg: Dict[str, Any],
    experiment: Dict[str, Any],
    context: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    server_name = experiment.get("server") or model_cfg.get("server")
    if not server_name:
        return None

    servers = config.get("servers", {})
    if server_name not in servers:
        raise KeyError(f"Experiment references missing server '{server_name}'")

    raw_server = servers[server_name]
    cwd = render_value(raw_server.get("cwd", experiment.get("cwd", model_cfg.get("cwd", defaults.get("cwd", "{repo_root}")))), context)
    env = merge_env(context, defaults.get("env"), model_cfg.get("env"), raw_server.get("env"))
    start_command = render_value(raw_server["start_command"], context)
    stop_command = render_value(raw_server.get("stop_command"), context) if raw_server.get("stop_command") else None
    healthcheck_url_value = render_value(raw_server.get("healthcheck_url"), context) if raw_server.get("healthcheck_url") else None
    healthcheck_cmd_value = (
        render_value(raw_server.get("healthcheck_command"), context)
        if raw_server.get("healthcheck_command")
        else None
    )

    identity_payload = {
        "name": server_name,
        "cwd": cwd,
        "env": env,
        "start_command": start_command,
        "stop_command": stop_command,
        "healthcheck_url": healthcheck_url_value,
        "healthcheck_command": healthcheck_cmd_value,
    }
    instance_key = hashlib.sha1(json.dumps(identity_payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    resolved = dict(raw_server)
    resolved.update(
        {
            "name": server_name,
            "cwd": cwd,
            "env": env,
            "start_command": start_command,
            "stop_command": stop_command,
            "healthcheck_url": healthcheck_url_value,
            "healthcheck_command": healthcheck_cmd_value,
            "instance_key": instance_key,
        }
    )
    return resolved


def resolve_tasks(
    config: Dict[str, Any],
    repo_root: Path,
    run_root: Path,
    selected_models: Optional[set],
    selected_experiments: Optional[set],
) -> List[Dict[str, Any]]:
    defaults = config.get("defaults", {})
    models = config.get("models", {})
    experiments = config.get("experiments", [])
    suite_name = slugify(config.get("name", Path(config.get("__config_path__", "suite")).stem))

    if not models:
        raise ValueError("Config must define at least one model in 'models'")
    if not experiments:
        raise ValueError("Config must define at least one experiment in 'experiments'")

    tasks: List[Dict[str, Any]] = []
    all_model_ids = list(models.keys())

    for exp_index, experiment in enumerate(experiments):
        exp_id = experiment["id"]
        if selected_experiments and exp_id not in selected_experiments:
            continue
        if experiment.get("disabled"):
            continue

        model_ids = experiment.get("models", all_model_ids)
        seeds = resolve_seeds(defaults, experiment)

        for model_id in model_ids:
            if selected_models and model_id not in selected_models:
                continue
            if model_id not in models:
                raise KeyError(f"Experiment '{exp_id}' references unknown model '{model_id}'")

            model_cfg = models[model_id]
            for repeat_index, seed in enumerate(seeds):
                context = build_context(
                    suite_name=suite_name,
                    repo_root=repo_root,
                    run_root=run_root,
                    defaults=defaults,
                    model_id=model_id,
                    model_cfg=model_cfg,
                    experiment=experiment,
                    seed=seed,
                    repeat_index=repeat_index,
                )
                task_id = render_value(
                    experiment.get("task_id", "{experiment_id}__{model_id}__seed{seed}"),
                    context,
                )
                task_id = slugify(task_id)
                context["task_id"] = task_id

                cwd = render_value(
                    experiment.get("cwd", model_cfg.get("cwd", defaults.get("cwd", "{repo_root}"))),
                    context,
                )
                env = merge_env(context, defaults.get("env"), model_cfg.get("env"), experiment.get("env"))
                command = render_value(experiment["command"], context)
                done_file = (
                    Path(render_value(experiment["done_file"], context)).resolve()
                    if experiment.get("done_file")
                    else None
                )
                timeout_sec = experiment.get("timeout_sec", defaults.get("timeout_sec"))
                server_spec = resolve_server_spec(config, defaults, model_cfg, experiment, context)

                task = {
                    "task_id": task_id,
                    "suite_name": suite_name,
                    "experiment_id": exp_id,
                    "model_id": model_id,
                    "seed": seed,
                    "repeat_index": repeat_index,
                    "cwd": cwd,
                    "env": env,
                    "command": command,
                    "done_file": str(done_file) if done_file else None,
                    "timeout_sec": float(timeout_sec) if timeout_sec is not None else None,
                    "server_spec": server_spec,
                    "sort_key": (
                        model_id,
                        exp_index,
                        seed,
                    ),
                }
                tasks.append(task)

    tasks.sort(key=lambda item: item["sort_key"])
    return tasks


def load_state(state_path: Path, run_root: Path, config_path: Path) -> Dict[str, Any]:
    if state_path.is_file():
        state = load_json(state_path)
    else:
        state = {
            "created_at": now_iso(),
            "run_root": str(run_root),
            "config_path": str(config_path),
            "tasks": {},
        }
    state.setdefault("tasks", {})
    return state


def mark_task(state: Dict[str, Any], state_path: Path, task: Dict[str, Any], **fields: Any) -> None:
    record = state["tasks"].setdefault(task["task_id"], {})
    record.update(fields)
    dump_json(state_path, state)


def print_plan(tasks: List[Dict[str, Any]]) -> None:
    print(f"[suite] resolved {len(tasks)} task(s)")
    for task in tasks:
        server = task["server_spec"]["name"] if task["server_spec"] else "-"
        print(
            f"  - {task['task_id']}: model={task['model_id']} exp={task['experiment_id']} "
            f"seed={task['seed']} server={server}"
        )


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    config["__config_path__"] = str(config_path)

    suite_name = slugify(config.get("name", config_path.stem))
    default_run_root = repo_root / "suite_runs" / f"{suite_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = Path(args.run_root).resolve() if args.run_root else default_run_root
    ensure_dir(run_root)

    state_path = run_root / "suite_state.json"
    state = load_state(state_path, run_root, config_path)

    selected_models = comma_set(args.only_models)
    selected_experiments = comma_set(args.only_experiments)
    tasks = resolve_tasks(config, repo_root, run_root, selected_models, selected_experiments)
    dump_json(run_root / "resolved_tasks.json", {"tasks": tasks, "created_at": now_iso()})
    print_plan(tasks)

    if args.dry_run:
        print(f"[suite] dry run only. Resolved plan written to {run_root / 'resolved_tasks.json'}")
        return 0

    logs_dir = ensure_dir(run_root / "logs")
    task_logs_dir = ensure_dir(logs_dir / "tasks")
    server_logs_dir = ensure_dir(logs_dir / "servers")

    failures = 0
    current_server: Optional[ManagedServer] = None
    current_server_key: Optional[str] = None

    try:
        for task in tasks:
            existing = state["tasks"].get(task["task_id"], {})
            if args.resume and existing.get("status") == "success":
                print(f"[skip] {task['task_id']} already completed successfully")
                continue

            if args.resume and task["done_file"] and Path(task["done_file"]).exists():
                print(f"[skip] {task['task_id']} done_file already exists")
                mark_task(
                    state,
                    state_path,
                    task,
                    status="success",
                    skipped=True,
                    started_at=existing.get("started_at"),
                    finished_at=now_iso(),
                    command=task["command"],
                    done_file=task["done_file"],
                )
                continue

            desired_server = task["server_spec"]
            desired_server_key = desired_server["instance_key"] if desired_server else None

            if current_server_key != desired_server_key:
                if current_server is not None:
                    print(f"[server] stopping {current_server.name}")
                    current_server.stop()
                current_server = None
                current_server_key = None

                if desired_server is not None:
                    server_log_path = server_logs_dir / f"{desired_server['name']}-{desired_server_key}.log"
                    current_server = ManagedServer(desired_server, server_log_path)
                    print(f"[server] starting {current_server.name}")
                    current_server.start()
                    current_server_key = desired_server_key

            log_path = task_logs_dir / f"{task['task_id']}.log"
            print(f"[run] {task['task_id']}")
            print(f"      cwd={task['cwd']}")
            print(f"      log={log_path}")

            mark_task(
                state,
                state_path,
                task,
                status="running",
                started_at=now_iso(),
                command=task["command"],
                cwd=task["cwd"],
                log_path=str(log_path),
                model_id=task["model_id"],
                experiment_id=task["experiment_id"],
                seed=task["seed"],
                done_file=task["done_file"],
            )

            result = run_foreground_command(
                command=task["command"],
                cwd=task["cwd"],
                env=task["env"],
                log_path=log_path,
                timeout_sec=task["timeout_sec"],
            )

            status = "success"
            failure_reason = None
            if result["return_code"] != 0:
                status = "failed"
                failure_reason = f"command exited with {result['return_code']}"
            elif task["done_file"] and not Path(task["done_file"]).exists():
                status = "failed"
                failure_reason = f"expected done_file missing: {task['done_file']}"

            mark_task(
                state,
                state_path,
                task,
                status=status,
                finished_at=now_iso(),
                return_code=result["return_code"],
                timed_out=result["timed_out"],
                duration_sec=result["duration_sec"],
                failure_reason=failure_reason,
            )

            if status == "failed":
                failures += 1
                print(f"[fail] {task['task_id']}: {failure_reason}")
                if args.fail_fast:
                    break
            else:
                print(f"[ok] {task['task_id']} ({result['duration_sec']}s)")
    finally:
        if current_server is not None:
            print(f"[server] stopping {current_server.name}")
            current_server.stop()

    total = len(tasks)
    completed = sum(1 for rec in state["tasks"].values() if rec.get("status") == "success")
    failed = sum(1 for rec in state["tasks"].values() if rec.get("status") == "failed")
    print(f"[suite] finished: total={total} success={completed} failed={failed} run_root={run_root}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
