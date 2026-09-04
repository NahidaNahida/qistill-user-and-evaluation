"""In-memory information and execution boundary for code generation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


@dataclass(frozen=True)
class SandboxPolicy:
    """Validate and describe the fixed benchmark sandbox contract.

    This object is runtime configuration, not a generated artifact. The Agent
    receives only mediated Skill-resource tools; its only optional external
    input is the selected Skill bundle.
    """

    environment_file: str
    has_skill: bool
    conda_environment: str = "qistill"

    def validate(self) -> None:
        if not os.path.isfile(self.environment_file):
            raise FileNotFoundError(
                f"Sandbox environment definition does not exist: {self.environment_file}"
            )

    def as_dict(self) -> dict[str, Any]:
        """Expose the contract to an executor without writing policy files."""
        self.validate()
        return {
            "filesystem": {
                "readable_inputs": ["skill_bundle"] if self.has_skill else [],
                "project_access": "denied",
                "quanbench_access": "denied",
            },
            "network": "denied",
            "external_input": "skill_bundle_only" if self.has_skill else "none",
            "skill_condition": "skill" if self.has_skill else "no_skill",
            "skill_references": "relative_paths_inside_skill_bundle_only",
            "environment_definition": os.path.realpath(self.environment_file),
            # Skill scripts are executed by the mediated resource tool in this
            # Conda environment; the Agent cannot select an arbitrary env.
            "conda_environment": self.conda_environment,
            "environment_visible_to_agent": self.has_skill,
            "agent_tools": (
                ["list_skill_resources", "read_skill_resource", "run_skill_resource", "submit_code"]
                if self.has_skill else ["submit_code"]
            ),
        }


__all__ = ["SandboxPolicy"]
