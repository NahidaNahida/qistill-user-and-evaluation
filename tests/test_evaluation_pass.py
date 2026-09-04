from __future__ import annotations

import importlib.util
import os
import subprocess

import pytest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALUATION_PATH = os.path.join(PROJECT_ROOT, "src", "evaluation", "pass.py")


def _load_evaluation_module():
    spec = importlib.util.spec_from_file_location("evaluation_pass", EVALUATION_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load evaluation module from {EVALUATION_PATH}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(content)


@pytest.mark.parametrize("task_id", ["01", "02", "06"])
def test_quanbench_correctness_runs_real_quanbench44_assertions(task_id: str) -> None:
    evaluation = _load_evaluation_module()
    reference_path = os.path.join(PROJECT_ROOT, "quanbench", "QuanBench44.jsonl")
    generated_path = os.path.join(
        PROJECT_ROOT, "tests", "tests_generated_code", f"{task_id}.py"
    )

    # The committed generated-code fixtures are copies of the corresponding
    # canonical solutions. Together, tasks 01, 02, and 06 exercise QuanBench44's
    # measurement-result, phase, and gate-set assertions through the real JSONL.
    assert evaluation.quanbench_correctness_pass(
        generated_path,
        reference_path,
    ) is True


def test_quanbench_correctness_returns_generated_artifact_result(
    tmp_path, monkeypatch
) -> None:
    evaluation = _load_evaluation_module()
    generated_path = os.path.join(str(tmp_path), "01.py")
    reference_path = os.path.join(str(tmp_path), "tasks.jsonl")
    generated_code = "def solve():\n    return 1\n"
    problem = {
        "task_id": "01",
        "entry_point": "solve",
        "canonical_solution": generated_code,
        "test": "",
    }
    _write_text(generated_path, generated_code)
    _write_text(reference_path, "{}\n")
    captured: dict[str, object] = {}

    def fake_read_problems(path: str) -> dict[str, dict[str, str]]:
        captured["reference_path"] = path
        return {"01": problem}

    def fake_check_correctness(
        selected_problem: dict[str, str], completion: str, timeout: float
    ) -> dict[str, object]:
        captured.update(
            {"problem": selected_problem, "completion": completion, "timeout": timeout}
        )
        return {"passed": True}

    import quanbench.Quanbench_eval.data as quanbench_data
    import quanbench.Quanbench_eval.execution as quanbench_execution

    monkeypatch.setattr(quanbench_data, "read_problems", fake_read_problems)
    monkeypatch.setattr(
        quanbench_execution, "check_correctness", fake_check_correctness
    )

    assert evaluation.quanbench_correctness_pass(generated_path, reference_path) is True
    assert captured == {
        "reference_path": reference_path,
        "problem": problem,
        "completion": generated_code,
        "timeout": 50.0,
    }


def test_quanbench_correctness_returns_false_when_assertions_fail(
    tmp_path, monkeypatch
) -> None:
    evaluation = _load_evaluation_module()
    generated_path = os.path.join(str(tmp_path), "02.py")
    reference_path = os.path.join(str(tmp_path), "tasks.jsonl")
    _write_text(generated_path, "def solve():\n    return 0\n")
    _write_text(reference_path, "{}\n")

    import quanbench.Quanbench_eval.data as quanbench_data
    import quanbench.Quanbench_eval.execution as quanbench_execution

    monkeypatch.setattr(
        quanbench_data,
        "read_problems",
        lambda path: {"02": {"task_id": "02"}},
    )
    monkeypatch.setattr(
        quanbench_execution,
        "check_correctness",
        lambda problem, completion, timeout: {"passed": False},
    )

    assert evaluation.quanbench_correctness_pass(generated_path, reference_path) is False


def test_quanbench_correctness_forwards_custom_timeout(tmp_path, monkeypatch) -> None:
    evaluation = _load_evaluation_module()
    generated_path = os.path.join(str(tmp_path), "03.py")
    reference_path = os.path.join(str(tmp_path), "tasks.jsonl")
    _write_text(generated_path, "def solve():\n    return 1\n")
    _write_text(reference_path, "{}\n")
    captured: dict[str, float] = {}

    import quanbench.Quanbench_eval.data as quanbench_data
    import quanbench.Quanbench_eval.execution as quanbench_execution

    monkeypatch.setattr(
        quanbench_data,
        "read_problems",
        lambda path: {"03": {"task_id": "03"}},
    )

    def fake_check_correctness(problem, completion, timeout):
        captured["timeout"] = timeout
        return {"passed": True}

    monkeypatch.setattr(
        quanbench_execution, "check_correctness", fake_check_correctness
    )

    assert evaluation.quanbench_correctness_pass(
        generated_path, reference_path, timeout=75
    ) is True
    assert captured["timeout"] == 75.0


def test_quanbench_correctness_raises_for_task_loading_failures(tmp_path) -> None:
    evaluation = _load_evaluation_module()
    generated_path = os.path.join(str(tmp_path), "missing-task.py")
    reference_path = os.path.join(str(tmp_path), "tasks.jsonl")
    _write_text(generated_path, "def solve():\n    return 1\n")
    _write_text(reference_path, '{"task_id": "01"}\n')

    with pytest.raises(KeyError, match="missing-task"):
        evaluation.quanbench_correctness_pass(generated_path, reference_path)

    with pytest.raises(FileNotFoundError):
        evaluation.quanbench_correctness_pass(
            generated_path, os.path.join(str(tmp_path), "missing.jsonl")
        )


def test_compilation_pass_reports_only_generated_code_quality(tmp_path) -> None:
    evaluation = _load_evaluation_module()
    valid_path = os.path.join(str(tmp_path), "valid.py")
    invalid_path = os.path.join(str(tmp_path), "invalid.py")
    _write_text(valid_path, "value = 1\n")
    _write_text(invalid_path, "def broken(:\n")

    assert evaluation.compilation_pass(valid_path) is True
    assert evaluation.compilation_pass(invalid_path) is False

    with pytest.raises(FileNotFoundError):
        evaluation.compilation_pass(os.path.join(str(tmp_path), "missing.py"))


@pytest.mark.parametrize("returncode, expected", [(0, True), (1, False)])
def test_execution_pass_uses_generated_process_exit_status(
    tmp_path, monkeypatch, returncode: int, expected: bool
) -> None:
    evaluation = _load_evaluation_module()
    generated_path = os.path.join(str(tmp_path), "generated.py")
    _write_text(generated_path, "value = 1\n")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=returncode)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert evaluation.execution_pass(generated_path) is expected


def test_execution_pass_distinguishes_timeout_from_launch_failure(
    tmp_path, monkeypatch
) -> None:
    evaluation = _load_evaluation_module()
    generated_path = os.path.join(str(tmp_path), "generated.py")
    _write_text(generated_path, "value = 1\n")

    def timeout_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=120)

    monkeypatch.setattr(subprocess, "run", timeout_run)
    assert evaluation.execution_pass(generated_path) is False

    def failed_launch(*args, **kwargs):
        raise OSError("conda could not be launched")

    monkeypatch.setattr(subprocess, "run", failed_launch)
    with pytest.raises(OSError, match="conda could not be launched"):
        evaluation.execution_pass(generated_path)

    with pytest.raises(FileNotFoundError):
        evaluation.execution_pass(os.path.join(str(tmp_path), "missing.py"))
