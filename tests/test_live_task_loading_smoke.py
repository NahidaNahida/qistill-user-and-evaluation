"""Opt-in live OpenRouter smoke test for the task-loading workflow.

Run from the qistill environment with ``QISTILL_LIVE_SMOKE=1``.  The test is
skipped by default so ordinary test runs never make a provider request.
"""

from __future__ import annotations

import json
import os
import sys

import pytest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from task_loading import run_task_file


LIVE_SMOKE_ENABLED = os.environ.get("QISTILL_LIVE_SMOKE") == "1"
LIVE_MODEL = "qwen/qwen3.5-35b-a3b"
LIVE_SEED = 42


def _write_sample(source_path: str, target_path: str, sample_size: int) -> list[dict[str, str]]:
    """Materialize a small prompt-only sample from the real benchmark file."""
    records: list[dict[str, str]] = []
    with open(source_path, "r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            source_record = json.loads(line)
            records.append({
                "task_id": source_record["task_id"],
                "complete_prompt": source_record["complete_prompt"],
            })
            if len(records) == sample_size:
                break
    if len(records) != sample_size:
        raise AssertionError(f"Expected {sample_size} benchmark records.")

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


@pytest.mark.skipif(
    not LIVE_SMOKE_ENABLED,
    reason="Set QISTILL_LIVE_SMOKE=1 to enable the live OpenRouter smoke test.",
)
def test_live_openrouter_task_loading_smoke(tmp_path) -> None:
    """Prove that task_loading reaches the real OpenRouter Agent backend."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.fail("QISTILL_LIVE_SMOKE=1 requires OPENROUTER_API_KEY to be set.")

    quanbench_root = os.path.join(str(tmp_path), "quanbench")
    source_path = os.path.join(PROJECT_ROOT, "quanbench", "Quanbench117.jsonl")
    sample_path = os.path.join(quanbench_root, "Quanbench117-live-smoke.jsonl")
    environment_file = os.path.join(quanbench_root, "environment.yml")
    os.makedirs(quanbench_root, exist_ok=True)
    with open(environment_file, "w", encoding="utf-8") as stream:
        stream.write("name: Quanbench\ndependencies:\n  - python>=3.10\n")

    records = _write_sample(source_path, sample_path, sample_size=1)
    results = run_task_file(
        task_file=os.path.basename(sample_path),
        llm=LIVE_MODEL,
        random_seed=LIVE_SEED,
        skill_path=None,
        output_root=os.path.join(str(tmp_path), "generated_code"),
        runtime_root=os.path.join(str(tmp_path), "runtime"),
        environment_file=environment_file,
        quanbench_root=quanbench_root,
    )

    assert len(results) == 1
    result = results[0]
    assert result.code_task == records[0]["task_id"]
    assert result.skill_path is None
    assert result.skill_condition == "no_skill"
    assert result.llm_task == "qwen-qwen3.5-35b-a3b__seed-42"
    assert os.path.isfile(result.artifact_path)
    assert os.path.isfile(result.runtime_path)
    with open(result.runtime_path, "r", encoding="utf-8") as stream:
        runtime = json.load(stream)
    assert runtime["status"] == "completed"
    assert runtime["llm"] == LIVE_MODEL
    assert runtime["skill_path"] is None
    assert runtime["termination_reason"] in {"submitted_code", "final_response"}
