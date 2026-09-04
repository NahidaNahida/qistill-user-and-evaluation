"""Load an optional named Skill, execute a benchmark Prompt, and save code."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import yaml

from .agent_runtime import AgentRequest, SingleAgentRuntime
from .openrouter import OpenRouterClient
from sandbox import SandboxPolicy
from tool import SkillToolExecutor, agent_tools


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN_CONFIG_PATH = os.path.join(PROJECT_ROOT, "src", "run_config.yaml")
DEFAULT_ENVIRONMENT_FILE = os.path.join(PROJECT_ROOT, "environment-qistill.yml")
DEFAULT_CONDA_ENVIRONMENT = "qistill"
PROMPT_TEMPLATE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt")
PROMPT_TEMPLATE_NAME = "user_workflow.jinja"


@dataclass(frozen=True)
class WorkflowResult:
    artifact_path: str
    runtime_path: str
    llm_task: str
    code_task: str
    response_text: str
    usage: dict[str, Any]
    walltime_seconds: float
    task_id: str | None
    skill_path: str | None
    skill_condition: str


def _safe_component(value: str, fallback: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(".-")
    return (component or fallback)[:96]


def _code_task_name(task_prompt: str, task_id: str | None = None) -> str:
    if task_id is not None:
        return _safe_component(task_id, "code-task")
    prefix = _safe_component(task_prompt, "code-task")
    digest = hashlib.sha256(task_prompt.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}__{digest}"


def skill_condition(skill_path: str | None) -> str:
    """Return the stable experiment label for the Skill condition."""
    if skill_path is None:
        return "no_skill"
    return _safe_component(os.path.basename(os.path.normpath(skill_path)), "skill")


def load_run_config(config_path: str = RUN_CONFIG_PATH) -> tuple[tuple[str, ...], tuple[int, ...]]:
    with open(config_path, "r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not {"llm", "random_seeds"}.issubset(document):
        raise ValueError("run_config.yaml must contain llm and random_seeds.")
    models = document["llm"]
    seeds = document["random_seeds"]
    if not isinstance(models, list) or not models or not all(isinstance(item, str) for item in models):
        raise ValueError("run_config.yaml llm must be a non-empty string list.")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(item, int) for item in seeds):
        raise ValueError("run_config.yaml random_seeds must be a non-empty integer list.")
    return tuple(models), tuple(seeds)


def resolve_skill_path(skill_path: str | None) -> str | None:
    """Resolve the explicitly supplied Skill directory path."""
    if skill_path is None:
        return None
    if not isinstance(skill_path, str) or not skill_path.strip():
        raise ValueError("skill_path must be None or a non-empty directory path.")
    resolved_path = os.path.realpath(skill_path)
    if not os.path.isdir(resolved_path):
        raise FileNotFoundError(f"Skill directory does not exist: {resolved_path}")
    return resolved_path


def load_skill(skill_path: str | None) -> str | None:
    """Load only the Skill entry instructions; resources remain tool-mediated."""
    skill_path = resolve_skill_path(skill_path)
    if skill_path is None:
        return None
    entry_path = os.path.join(skill_path, "SKILL.md")
    if not os.path.isfile(entry_path):
        raise FileNotFoundError(f"Skill entry file does not exist: {entry_path}")
    with open(entry_path, "r", encoding="utf-8") as stream:
        return stream.read()


def build_agent_prompt(has_skill: bool, task_prompt: str) -> str:
    # Keep the Agent-facing contract in a version-controlled template so that
    # prompt changes are reviewable independently from workflow orchestration.
    template_path = os.path.join(PROMPT_TEMPLATE_ROOT, PROMPT_TEMPLATE_NAME)
    with open(template_path, "r", encoding="utf-8") as stream:
        template = stream.read()
    skill_block = ""
    if has_skill:
        skill_block = (
            "<skill_bundle>\n"
            "A Skill bundle is available through the Skill resource tools. Inspect it with\n"
            "list_skill_resources and read or run only the resources exposed by those tools.\n"
            "The Skill bundle is a directory of resources, not prompt text.\n"
            "</skill_bundle>\n"
        )
    return template.replace("{{ skill_bundle }}", skill_block).replace(
        "{{ task_prompt }}", task_prompt
    ).strip()


def _write_runtime_record(runtime_path: str, record: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(runtime_path), exist_ok=True)
    with open(runtime_path, "w", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


class QistillUser:
    """Execute one model/seed/task run with an optional named Skill."""

    def __init__(self, output_root: str = "data/generated_code",
                 model_call: Callable[..., Any] | None = None,
                 runtime_root: str = "data/runtime",
                 environment_file: str = DEFAULT_ENVIRONMENT_FILE) -> None:
        self.output_root = os.path.abspath(output_root)
        self.model_call = model_call
        self.runtime_root = os.path.abspath(runtime_root)
        self.environment_file = os.path.abspath(environment_file)

    def run(self, skill_path: str | None, task_prompt: str, llm: str,
            random_seed: int, task_id: str | None = None) -> WorkflowResult:
        allowed_llms, allowed_random_seeds = load_run_config()
        if llm not in allowed_llms:
            raise ValueError(f"Unsupported llm: {llm!r}")
        if random_seed not in allowed_random_seeds:
            raise ValueError(f"Unsupported random_seed: {random_seed!r}")
        if not isinstance(task_prompt, str) or not task_prompt.strip():
            raise ValueError("task_prompt must be a non-empty string.")

        llm_task = f"{_safe_component(llm, 'llm')}__seed-{random_seed}"
        code_task = _code_task_name(task_prompt, task_id)
        skill_condition_name = skill_condition(skill_path)
        # Sandbox is a runtime contract, not an output location. Validation is
        # performed without creating per-task files under src/pipeline.
        sandbox_policy = SandboxPolicy(
            self.environment_file,
            has_skill=skill_path is not None,
            conda_environment=DEFAULT_CONDA_ENVIRONMENT,
        )
        sandbox_policy.validate()
        runtime_path = os.path.join(
            self.runtime_root, skill_condition_name, llm_task, code_task, "runtime.json"
        )
        model_call = self.model_call or OpenRouterClient(llm, random_seed).complete
        skill_path = resolve_skill_path(skill_path)
        skill_tool_executor = SkillToolExecutor(
            skill_path, conda_environment=DEFAULT_CONDA_ENVIRONMENT
        )

        started = time.perf_counter()
        try:
            response = SingleAgentRuntime(model_call).run(AgentRequest(
                prompt=build_agent_prompt(skill_path is not None, task_prompt),
                tools=agent_tools(has_skill=skill_path is not None),
                max_steps=8,
                tool_executor=skill_tool_executor.execute,
            ))
        except Exception as error:
            walltime_seconds = time.perf_counter() - started
            _write_runtime_record(runtime_path, {
                "llm": llm, "random_seed": random_seed, "llm_task": llm_task,
                "code_task": code_task, "task_id": task_id, "skill_path": skill_path,
                "skill_condition": skill_condition_name,
                "status": "failed", "walltime_seconds": walltime_seconds,
                "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
                "usage": {}, "error_type": type(error).__name__, "error_message": str(error),
            })
            raise
        walltime_seconds = time.perf_counter() - started

        output_directory = os.path.join(self.output_root, skill_condition_name, llm_task)
        os.makedirs(output_directory, exist_ok=True)
        # One benchmark task maps directly to one Python source artifact.
        artifact_path = os.path.join(output_directory, f"{code_task}.py")
        with open(artifact_path, "w", encoding="utf-8") as stream:
            stream.write(response.text)
        _write_runtime_record(runtime_path, {
            "llm": llm, "random_seed": random_seed, "llm_task": llm_task,
            "code_task": code_task, "task_id": task_id, "skill_path": skill_path,
            "skill_condition": skill_condition_name,
            "status": "completed", "walltime_seconds": walltime_seconds,
            "prompt_tokens": response.usage.get("prompt_tokens"),
            "completion_tokens": response.usage.get("completion_tokens"),
            "total_tokens": response.usage.get("total_tokens"), "usage": response.usage,
            "agent_steps": response.steps,
            "termination_reason": response.termination_reason,
            "error_type": None, "error_message": None,
        })
        return WorkflowResult(
            artifact_path, runtime_path, llm_task, code_task,
            response.text, response.usage, walltime_seconds, task_id, skill_path, skill_condition_name,
        )

def run_skill_user_workflow(skill_path: str | None, task_prompt: str, llm: str,
                            random_seed: int, task_id: str | None = None,
                            output_root: str = "data/generated_code",
                            runtime_root: str = "data/runtime",
                            environment_file: str = DEFAULT_ENVIRONMENT_FILE,
                            model_call: Callable[..., Any] | None = None) -> WorkflowResult:
    workflow = QistillUser(
        output_root, model_call, runtime_root, environment_file
    )
    return workflow.run(skill_path, task_prompt, llm, random_seed, task_id=task_id)


__all__ = ["QistillUser", "WorkflowResult", "agent_tools", "build_agent_prompt", "load_run_config",
           "load_skill", "resolve_skill_path", "run_skill_user_workflow"]
