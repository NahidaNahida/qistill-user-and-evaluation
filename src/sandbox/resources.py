"""Restricted access to resources owned by one selected Skill bundle."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SkillResourceAccess:
    """List, read, and invoke resources without leaving the Skill root."""

    skill_root: str
    execution_timeout_seconds: int = 30
    conda_environment: str = "qistill"

    def _resolve(self, relative_path: str) -> str:
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError("Skill resource path must be a non-empty relative path.")
        if os.path.isabs(relative_path):
            raise PermissionError("Skill resource path must be relative.")
        root = os.path.realpath(self.skill_root)
        candidate = os.path.realpath(os.path.join(root, relative_path))
        try:
            inside_root = os.path.commonpath([root, candidate]) == root
        except ValueError:
            inside_root = False
        if not inside_root:
            raise PermissionError("Skill resource path escapes the selected Skill bundle.")
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"Skill resource does not exist: {relative_path}")
        return candidate

    def list_resources(self) -> dict[str, Any]:
        resources: list[str] = []
        for current_root, directory_names, file_names in os.walk(self.skill_root):
            directory_names.sort()
            for file_name in sorted(file_names):
                file_path = os.path.join(current_root, file_name)
                resolved_path = os.path.realpath(file_path)
                if os.path.commonpath([os.path.realpath(self.skill_root), resolved_path]) != os.path.realpath(self.skill_root):
                    continue
                relative_path = os.path.relpath(resolved_path, self.skill_root).replace(os.sep, "/")
                if relative_path != "SKILL.md":
                    resources.append(relative_path)
        return {"resources": resources}

    def read_resource(self, relative_path: str) -> dict[str, Any]:
        resource_path = self._resolve(relative_path)
        with open(resource_path, "r", encoding="utf-8") as stream:
            return {"path": relative_path, "content": stream.read()}

    def run_resource(self, relative_path: str, arguments: list[str] | None = None,
                     stdin: str = "") -> dict[str, Any]:
        resource_path = self._resolve(relative_path)
        if os.path.splitext(resource_path)[1].lower() != ".py":
            raise PermissionError("Only Python Skill resources may be executed.")
        safe_arguments = arguments or []
        if not isinstance(safe_arguments, list) or not all(isinstance(item, str) for item in safe_arguments):
            raise TypeError("Skill resource arguments must be a string list.")

        # No shell is involved, and the child receives no project-specific
        # credentials. This executes only a selected, bundle-contained Skill
        # script; generated model code is never executed by QistillUser.
        child_environment = {
            "PATH": os.environ.get("PATH", ""),
            # Python on Windows needs SYSTEMROOT to initialize correctly.
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            # Conda's Windows launcher resolves the current user before
            # starting the selected environment's interpreter.
            "USERNAME": os.environ.get("USERNAME", ""),
            "USERPROFILE": os.environ.get("USERPROFILE", ""),
            "HOMEDRIVE": os.environ.get("HOMEDRIVE", ""),
            "HOMEPATH": os.environ.get("HOMEPATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
        # The Skill User may run only bundle-contained Python resources, and
        # those resources run in the fixed qistill Conda environment.  Keeping
        # the environment fixed prevents the model from selecting arbitrary
        # interpreters while allowing Skills to use the project dependencies.
        completed = subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", self.conda_environment,
             "python", "-I", resource_path, *safe_arguments],
            input=stdin,
            text=True,
            capture_output=True,
            cwd=self.skill_root,
            env=child_environment,
            timeout=self.execution_timeout_seconds,
            check=False,
        )
        return {
            "path": relative_path,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }


__all__ = ["SkillResourceAccess"]
