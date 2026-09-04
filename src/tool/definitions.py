"""OpenAI-compatible tool schemas for the Skill User Agent."""

from __future__ import annotations

from typing import Any


def _tool_schema(name: str, description: str, properties: dict[str, Any],
                 required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def agent_tools(has_skill: bool) -> list[dict[str, Any]]:
    """Return the tool whitelist for the selected Skill condition."""
    tools: list[dict[str, Any]] = []
    if has_skill:
        tools.extend([
            _tool_schema(
                "list_skill_resources",
                "List resources contained in the selected Skill bundle.",
                {},
                [],
            ),
            _tool_schema(
                "read_skill_resource",
                "Read one relative resource from the selected Skill bundle.",
                {"path": {"type": "string"}},
                ["path"],
            ),
            _tool_schema(
                "run_skill_resource",
                "Run one Python resource from the selected Skill bundle and observe its result.",
                {
                    "path": {"type": "string"},
                    "arguments": {"type": "array", "items": {"type": "string"}},
                    "stdin": {"type": "string"},
                },
                ["path"],
            ),
        ])
    tools.append(_tool_schema(
        "submit_code",
        "Submit the complete final Python source code for this task.",
        {"code": {"type": "string"}},
        ["code"],
    ))
    return tools
