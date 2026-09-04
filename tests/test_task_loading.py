from __future__ import annotations

import json
import os
import sys
import threading
import time

import pytest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from task_loading import load_tasks, resolve_task_file, run_task_file


def _write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(content)


def _write_small_task_file(path: str) -> None:
    records = [
        {"task_id": "01", "complete_prompt": "def first():\n    pass", "canonical_solution": "hidden"},
        {"task_id": "02", "complete_prompt": "def second():\n    pass", "test": "hidden"},
    ]
    _write_text(path, "".join(json.dumps(record) + "\n" for record in records))


def test_loader_exposes_only_task_id_and_prompt(tmp_path) -> None:
    temporary_root = str(tmp_path)
    _write_small_task_file(os.path.join(temporary_root, "tasks.jsonl"))
    tasks = load_tasks("tasks.jsonl", benchmark_root=temporary_root)
    assert [(task.task_id, task.prompt) for task in tasks] == [
        ("01", "def first():\n    pass"),
        ("02", "def second():\n    pass"),
    ]


def test_task_file_rejects_path_traversal(tmp_path) -> None:
    try:
        resolve_task_file("../secret.jsonl", benchmark_root=str(tmp_path))
    except ValueError as error:
        assert "filename" in str(error)
    else:
        raise AssertionError("Path traversal must be rejected.")


def test_outer_interface_generates_small_task_file_without_skill(tmp_path) -> None:
    temporary_root = str(tmp_path)
    benchmark_root = os.path.join(temporary_root, "quanbench")
    task_file = os.path.join(benchmark_root, "tasks.jsonl")
    environment_file = os.path.join(benchmark_root, "environment.yml")
    _write_small_task_file(task_file)
    _write_text(environment_file, "name: Quanbench\ndependencies:\n  - python>=3.10\n")
    prompts: list[str] = []
    tool_inputs: list[object] = []

    def fake_model_call(**kwargs: object) -> dict[str, object]:
        prompt = kwargs["messages"][0]["content"]
        prompts.append(prompt)
        tool_inputs.append(kwargs["tools"])
        function_name = "first" if "first" in prompt else "second"
        return {
            "choices": [{"message": {"content": f"def {function_name}():\n    return 1\n"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
        }

    results = run_task_file(
        task_file="tasks.jsonl",
        llm="qwen/qwen3.5-35b-a3b",
        random_seed=42,
        skill_path=None,
        output_root=os.path.join(temporary_root, "generated_code"),
        runtime_root=os.path.join(temporary_root, "runtime"),
        environment_file=environment_file,
        model_call=fake_model_call,
        benchmark_root=benchmark_root,
    )

    assert [result.code_task for result in results] == ["01", "02"]
    assert ["def first():\n    pass" in prompt for prompt in prompts] == [True, False]
    assert ["def second():\n    pass" in prompt for prompt in prompts] == [False, True]
    assert [
        [tool["function"]["name"] for tool in tools]
        for tools in tool_inputs
    ] == [["submit_code"], ["submit_code"]]
    assert all(os.path.isfile(result.artifact_path) for result in results)
    assert [os.path.basename(result.artifact_path) for result in results] == ["01.py", "02.py"]
    assert all(os.path.isfile(result.runtime_path) for result in results)
    assert not os.path.exists(os.path.join(temporary_root, "sandbox"))
    for result in results:
        with open(result.runtime_path, "r", encoding="utf-8") as stream:
            runtime = json.load(stream)
        assert runtime["task_id"] == result.code_task
        assert runtime["skill_path"] is None
        assert runtime["total_tokens"] == 9


def test_task_file_parallel_mode_runs_independent_tasks_in_input_order(tmp_path) -> None:
    temporary_root = str(tmp_path)
    benchmark_root = os.path.join(temporary_root, "quanbench")
    task_file = os.path.join(benchmark_root, "tasks.jsonl")
    environment_file = os.path.join(benchmark_root, "environment.yml")
    _write_small_task_file(task_file)
    _write_text(environment_file, "name: Quanbench\n")
    active_calls = 0
    maximum_active_calls = 0
    state_lock = threading.Lock()

    def fake_model_call(**kwargs: object) -> dict[str, object]:
        nonlocal active_calls, maximum_active_calls
        prompt = kwargs["messages"][0]["content"]
        with state_lock:
            active_calls += 1
            maximum_active_calls = max(maximum_active_calls, active_calls)
        try:
            time.sleep(0.05)
            function_name = "first" if "first" in prompt else "second"
            return {
                "choices": [{"message": {"content": f"def {function_name}():\n    return 1\n"}}],
                "usage": {"total_tokens": 9},
            }
        finally:
            with state_lock:
                active_calls -= 1

    results = run_task_file(
        task_file="tasks.jsonl",
        llm="qwen/qwen3.5-35b-a3b",
        random_seed=42,
        output_root=os.path.join(temporary_root, "generated_code"),
        runtime_root=os.path.join(temporary_root, "runtime"),
        environment_file=environment_file,
        model_call=fake_model_call,
        benchmark_root=benchmark_root,
        execution_mode="parallel",
        max_workers=2,
    )

    assert [result.code_task for result in results] == ["01", "02"]
    assert maximum_active_calls == 2


def test_task_file_rejects_unknown_execution_mode(tmp_path) -> None:
    with pytest.raises(ValueError, match="execution_mode"):
        run_task_file(
            task_file="tasks.jsonl",
            llm="qwen/qwen3.5-35b-a3b",
            random_seed=42,
            benchmark_root=str(tmp_path),
            execution_mode="batch",
        )
