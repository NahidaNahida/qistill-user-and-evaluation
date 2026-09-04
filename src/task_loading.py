"""Load configured Quanbench tasks and call the outer Skill-user interface."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

import yaml

from pipeline import WorkflowResult, run_skill_user_workflow


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_CONFIG_PATH = os.path.join(PROJECT_ROOT, "src", "run_config.yaml")
benchmark_root = os.path.join(PROJECT_ROOT, "quanbench")


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    prompt: str


def load_run_config(config_path: str = RUN_CONFIG_PATH) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    required = {"llm", "random_seeds", "task_file"}
    if not isinstance(document, dict) or set(document) != required:
        raise ValueError(f"run_config.yaml must contain exactly {sorted(required)}.")
    if not isinstance(document["task_file"], list) or not document["task_file"]:
        raise ValueError("run_config.yaml task_file must be a non-empty list.")
    return document


def resolve_task_file(task_file: str, benchmark_root: str = benchmark_root) -> str:
    """Resolve a filename directly under Quanbench and reject path traversal."""
    if not isinstance(task_file, str) or not task_file.strip():
        raise ValueError("task_file must be a non-empty filename.")
    if os.path.basename(task_file) != task_file:
        raise ValueError("task_file must be a filename, not a path.")
    root = os.path.realpath(benchmark_root)
    task_path = os.path.realpath(os.path.join(root, task_file))
    if os.path.dirname(task_path) != root:
        raise ValueError("task_file must resolve directly under quanbench/.")
    if not os.path.isfile(task_path):
        raise FileNotFoundError(f"Configured task file does not exist: {task_path}")
    return task_path


def load_tasks(task_file: str, benchmark_root: str = benchmark_root) -> list[BenchmarkTask]:
    """Read only task_id and Prompt; never expose solutions or tests."""
    tasks: list[BenchmarkTask] = []
    with open(resolve_task_file(task_file, benchmark_root), "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = record.get("task_id")
            prompt = record.get("prompt", record.get("complete_prompt"))
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(f"Task line {line_number} has no valid task_id.")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Task {task_id!r} has no valid prompt.")
            tasks.append(BenchmarkTask(task_id, prompt))
    return tasks


def run_task_file(task_file: str, llm: str, random_seed: int,
                  skill_path: str | None = None,
                  output_root: str = "data/generated_code",
                  runtime_root: str = "data/runtime",
                  environment_file: str | None = None,
                  model_call: Callable[..., Any] | None = None,
                  benchmark_root: str = benchmark_root,
                  execution_mode: str = "sequential",
                  max_workers: int = 4) -> list[WorkflowResult]:
    """Run every record in one configured task file."""
    if execution_mode not in {"sequential", "parallel"}:
        raise ValueError("execution_mode must be 'sequential' or 'parallel'.")
    if not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers <= 0:
        raise ValueError("max_workers must be a positive integer.")

    workflow_arguments: dict[str, Any] = {
        "skill_path": skill_path,
        "llm": llm,
        "random_seed": random_seed,
        "output_root": output_root,
        "runtime_root": runtime_root,
        "model_call": model_call,
    }
    if environment_file is not None:
        workflow_arguments["environment_file"] = environment_file
    tasks = load_tasks(task_file, benchmark_root)

    def run_task(task: BenchmarkTask) -> WorkflowResult:
        return run_skill_user_workflow(
            task_prompt=task.prompt,
            task_id=task.task_id,
            **workflow_arguments,
        )

    if execution_mode == "sequential":
        return [run_task(task) for task in tasks]

    # Executor.map preserves input order while independently scheduling each
    # task workflow across the bounded worker pool.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(run_task, tasks))


def run_configured_tasks(config_path: str = RUN_CONFIG_PATH,
                         skill_path: str | None = None,
                         model_call: Callable[..., Any] | None = None) -> list[WorkflowResult]:
    """Run the configured task-file/model/seed Cartesian product."""
    config = load_run_config(config_path)
    results: list[WorkflowResult] = []
    for task_file in config["task_file"]:
        for llm in config["llm"]:
            for random_seed in config["random_seeds"]:
                results.extend(run_task_file(
                    task_file, llm, random_seed,
                    skill_path=skill_path,
                    model_call=model_call,
                ))
    return results


__all__ = ["BenchmarkTask", "load_run_config", "load_tasks", "resolve_task_file",
           "run_configured_tasks", "run_task_file"]
