from __future__ import annotations

import json
import os
import sys

import pytest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pipeline import QistillUser
from pipeline.workflow import build_agent_prompt, load_run_config, load_skill
from sandbox import SandboxPolicy


def _write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(content)


def test_workflow_loads_skill_path_and_writes_artifacts(tmp_path) -> None:
    temporary_root = str(tmp_path)
    skill_path = os.path.join(temporary_root, "quantum")
    _write_text(os.path.join(skill_path, "SKILL.md"), "Generate readable Python.")
    _write_text(os.path.join(skill_path, "references", "api.txt"), "Use QuantumCircuit.")
    environment_file = os.path.join(temporary_root, "environment.yml")
    _write_text(environment_file, "name: Quanbench\n")
    captured: dict[str, object] = {}

    def fake_model_call(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "choices": [{"message": {"content": "def bell_state():\n    return 1\n"}}],
            "usage": {"total_tokens": 12},
        }

    result = QistillUser(
        output_root=os.path.join(temporary_root, "generated_code"),
        model_call=fake_model_call,
        runtime_root=os.path.join(temporary_root, "runtime"),
        environment_file=environment_file,
    ).run(skill_path, "implement task", "qwen/qwen3.5-35b-a3b", 42, task_id="01")

    messages = captured["messages"]
    assert "Skill bundle is available" in messages[0]["content"]
    assert "Generate readable Python." not in messages[0]["content"]
    assert "Use QuantumCircuit." not in messages[0]["content"]
    assert {tool["function"]["name"] for tool in captured["tools"]} == {
        "list_skill_resources", "read_skill_resource", "run_skill_resource", "submit_code"
    }
    assert os.path.isfile(result.artifact_path)
    assert result.artifact_path.endswith(
        os.path.join("qwen-qwen3.5-35b-a3b__seed-42", "01.py")
    )
    assert os.path.isfile(result.runtime_path)
    assert result.code_task == "01"
    with open(result.runtime_path, "r", encoding="utf-8") as stream:
        runtime = json.load(stream)
    assert runtime["skill_path"] == os.path.realpath(skill_path)
    assert runtime["total_tokens"] == 12
    policy = SandboxPolicy(environment_file, has_skill=True).as_dict()
    assert policy["external_input"] == "skill_bundle_only"
    assert policy["filesystem"]["readable_inputs"] == ["skill_bundle"]
    assert policy["conda_environment"] == "qistill"
    assert policy["environment_visible_to_agent"] is True


def test_none_skill_is_exact_baseline_prompt(tmp_path) -> None:
    temporary_root = str(tmp_path)
    environment_file = os.path.join(temporary_root, "environment.yml")
    _write_text(environment_file, "name: Quanbench\n")
    captured: dict[str, object] = {}

    def fake_model_call(**kwargs: object) -> str:
        captured.update(kwargs)
        return "generated code"

    result = QistillUser(
        output_root=os.path.join(temporary_root, "generated"),
        runtime_root=os.path.join(temporary_root, "runtime"),
        environment_file=environment_file,
        model_call=fake_model_call,
    ).run(None, "original prompt", "qwen/qwen3.5-35b-a3b", 42, task_id="02")

    assert "qistill Conda environment" in captured["messages"][0]["content"]
    assert "Python 3.10" in captured["messages"][0]["content"]
    assert "qiskit 0.46.0" in captured["messages"][0]["content"]
    assert [tool["function"]["name"] for tool in captured["tools"]] == ["submit_code"]
    policy = SandboxPolicy(environment_file, has_skill=False).as_dict()
    assert policy["external_input"] == "none"
    assert policy["filesystem"]["readable_inputs"] == []
    assert policy["network"] == "denied"
    assert policy["agent_tools"] == ["submit_code"]
    assert policy["environment_visible_to_agent"] is False


def test_agent_autonomously_reads_and_runs_skill_resources(tmp_path) -> None:
    temporary_root = str(tmp_path)
    skill_path = os.path.join(temporary_root, "defensive-skill")
    environment_file = os.path.join(temporary_root, "environment.yml")
    _write_text(
        os.path.join(skill_path, "SKILL.md"),
        "Use references and defensive scripts when they are useful.",
    )
    _write_text(
        os.path.join(skill_path, "references", "rule.txt"),
        "Generated code must return a positive integer.",
    )
    _write_text(
        os.path.join(skill_path, "scripts", "guard.py"),
        "import sys\nvalue = int(sys.stdin.read())\nprint('safe' if value > 0 else 'unsafe')\n",
    )
    _write_text(environment_file, "name: Quanbench\n")
    calls: list[dict[str, object]] = []

    def tool_call(name: str, arguments: dict[str, object], call_id: str) -> dict[str, object]:
        return {
            "choices": [{"message": {"tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }]}}],
            "usage": {"total_tokens": 2},
        }

    def fake_model_call(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        step = len(calls)
        if step == 1:
            return tool_call("list_skill_resources", {}, "list-1")
        if step == 2:
            assert "references/rule.txt" in kwargs["messages"][-1]["content"]
            return tool_call("read_skill_resource", {"path": "references/rule.txt"}, "read-1")
        if step == 3:
            assert "positive integer" in kwargs["messages"][-1]["content"]
            return tool_call(
                "run_skill_resource",
                {"path": "scripts/guard.py", "stdin": "1"},
                "run-1",
            )
        assert 'safe' in kwargs["messages"][-1]["content"]
        return tool_call(
            "submit_code",
            {"code": "def guarded_value():\n    return 1\n"},
            "submit-1",
        )

    result = QistillUser(
        output_root=os.path.join(temporary_root, "generated"),
        runtime_root=os.path.join(temporary_root, "runtime"),
        environment_file=environment_file,
        model_call=fake_model_call,
    ).run(skill_path, "Implement guarded_value().", "qwen/qwen3.5-35b-a3b", 42, task_id="03")

    assert len(calls) == 4
    assert result.response_text == "def guarded_value():\n    return 1\n"
    assert result.usage["total_tokens"] == 8
    with open(result.artifact_path, "r", encoding="utf-8") as stream:
        assert stream.read() == result.response_text
    with open(result.runtime_path, "r", encoding="utf-8") as stream:
        runtime = json.load(stream)
    assert [step.get("tool_name") for step in runtime["agent_steps"]] == [
        "list_skill_resources",
        "read_skill_resource",
        "run_skill_resource",
        "submit_code",
    ]
    assert runtime["termination_reason"] == "submitted_code"


def test_missing_skill_path_is_rejected(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_skill(os.path.join(str(tmp_path), "missing"))


def test_prompt_wrapping() -> None:
    prompt_without_skill = build_agent_prompt(False, "task")
    assert "qistill Conda environment" in prompt_without_skill
    assert "Python 3.10" in prompt_without_skill
    assert "qiskit 0.46.0" in prompt_without_skill
    assert "Skill bundle is available" in build_agent_prompt(True, "task")
    assert "<skill>" not in build_agent_prompt(True, "task")


def test_run_config_model_and_seed_values() -> None:
    models, seeds = load_run_config()
    assert "qwen/qwen3.5-35b-a3b" in models
    assert seeds == (42, 888, 2026)
