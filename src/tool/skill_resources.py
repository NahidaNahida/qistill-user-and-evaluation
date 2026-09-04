"""Dispatch the Agent's mediated Skill-resource operations."""

from __future__ import annotations

from typing import Any

from sandbox import SkillResourceAccess


class SkillToolExecutor:
    """Expose only resource operations for one selected Skill bundle."""

    def __init__(self, skill_path: str | None, conda_environment: str = "qistill") -> None:
        self.resource_access = (
            SkillResourceAccess(skill_path, conda_environment=conda_environment)
            if skill_path is not None else None
        )

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a validated tool name through the Skill resource boundary."""
        if self.resource_access is None:
            raise PermissionError(
                "Skill resource tools are unavailable without a selected Skill."
            )
        if name == "list_skill_resources":
            return self.resource_access.list_resources()
        if name == "read_skill_resource":
            return self.resource_access.read_resource(arguments.get("path"))
        if name == "run_skill_resource":
            return self.resource_access.run_resource(
                arguments.get("path"),
                arguments.get("arguments"),
                arguments.get("stdin", ""),
            )
        raise PermissionError(f"Unsupported Skill resource tool: {name}")
