"""Bounded OpenAI-compatible Agent loop used by the Skill User workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class AgentRequest:
    prompt: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    max_steps: int = 8
    tool_executor: Callable[[str, dict[str, Any]], Any] | None = None


@dataclass(frozen=True)
class AgentResponse:
    text: str
    steps: list[dict[str, Any]]
    termination_reason: str
    usage: dict[str, Any]


def _merge_usage(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + value
        elif key not in target:
            target[key] = value


class SingleAgentRuntime:
    """Run bounded model/tool turns while preserving the author Agent API."""

    def __init__(self, model_call: Callable[..., Any], tool_executor: Callable[[str, dict[str, Any]], Any] | None = None) -> None:
        self.model_call = model_call
        self.tool_executor = tool_executor

    def run(self, request: AgentRequest) -> AgentResponse:
        executor = request.tool_executor or self.tool_executor
        messages: list[dict[str, Any]] = [{"role": "user", "content": request.prompt}]
        steps: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}
        for index in range(request.max_steps):
            raw = self.model_call(messages=messages, tools=request.tools)
            normalized = self._normalize(raw)
            _merge_usage(usage, normalized.get("usage", {}))
            step = {"step": index + 1, "kind": normalized["kind"]}
            if normalized["kind"] == "final":
                steps.append(step)
                return AgentResponse(normalized["content"], steps, "final_response", usage)
            if executor is None:
                raise RuntimeError("Agent requested a tool but no tool executor is configured.")
            tool_name = normalized["tool_name"]
            arguments = normalized["arguments"]
            call_id = normalized.get("tool_call_id") or f"call-{index + 1}"
            if tool_name == "submit_code":
                code = arguments.get("code")
                if not isinstance(code, str):
                    raise TypeError("submit_code requires a string code argument.")
                step.update({"tool_name": tool_name, "tool_call_id": call_id, "observation_status": "accepted"})
                steps.append(step)
                return AgentResponse(code, steps, "submitted_code", usage)
            try:
                observation = executor(tool_name, arguments)
                status = "passed"
            except Exception as error:
                observation = {"status": "error", "error_type": type(error).__name__, "message": str(error)}
                status = "error"
            step.update({"tool_name": tool_name, "tool_call_id": call_id, "observation_status": status})
            steps.append(step)
            messages.extend([
                {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": tool_name, "arguments": json.dumps(arguments)}}]},
                {"role": "tool", "tool_call_id": call_id, "name": tool_name, "content": json.dumps(observation, ensure_ascii=False)},
            ])
        raise RuntimeError("Agent step budget exhausted.")

    @staticmethod
    def _normalize(raw: Any) -> dict[str, Any]:
        if isinstance(raw, str):
            return {"kind": "final", "content": raw, "usage": {}}
        if not isinstance(raw, dict) or not raw.get("choices"):
            raise TypeError("Agent backend must return text or an OpenAI-compatible response.")
        message = raw["choices"][0].get("message", {})
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            call = tool_calls[0]
            function = call.get("function", {})
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(function.get("name"), str) or not isinstance(arguments, dict):
                raise TypeError("Malformed Agent tool call.")
            return {"kind": "tool_call", "tool_name": function["name"], "arguments": arguments,
                    "tool_call_id": call.get("id"), "usage": raw.get("usage", {})}
        content = message.get("content")
        if not isinstance(content, str):
            raise TypeError("Agent final response content must be text.")
        return {"kind": "final", "content": content, "usage": raw.get("usage", {})}


__all__ = ["AgentRequest", "AgentResponse", "SingleAgentRuntime"]
