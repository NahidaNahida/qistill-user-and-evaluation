"""OpenRouter reasoning backend adapted from qistill-author."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import request


class OpenRouterClient:
    """Issue one OpenAI-compatible code-generation request."""

    def __init__(self, model: str, seed: int, api_key: str | None = None) -> None:
        self.model = model
        self.seed = seed
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required.")
        body = {
            "model": self.model,
            "seed": self.seed,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a Skill-using code-generation Agent. Follow the supplied Skill and "
                        "execute the task Prompt. A Skill may be absent for a baseline run. "
                        "Do not browse, use network resources, or assume access to local project files. "
                        "Use only tools explicitly supplied by the QistillUser runtime and return the generated code artifact."
                    ),
                },
                *messages,
            ],
        }
        if tools:
            body["tools"] = tools
        http_request = request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(http_request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("OpenRouter response must be a JSON object.")
        return payload


__all__ = ["OpenRouterClient"]
