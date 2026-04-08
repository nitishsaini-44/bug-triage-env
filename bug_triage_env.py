"""
Bug Triage Environment — OpenEnv SDK Client Module.

Provides:
    BugTriageObservation   – typed observation model
    BugTriageAction        – typed action model
    BugTriageEnv           – async client (from_docker_image / from_url)

Usage:
    from bug_triage_env import BugTriageAction, BugTriageEnv

    env = await BugTriageEnv.from_docker_image(IMAGE_NAME)
    result = await env.reset(task_id="task_1_classify")
    result = await env.step(BugTriageAction(action_type="classify", bug_type="ui"))
    await env.close()
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import time
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field


# ═════════════════════════════════════════════════════════════════════════════
#  TYPED MODELS
# ═════════════════════════════════════════════════════════════════════════════


class BugTriageObservation(BaseModel):
    """Observation returned by the environment."""

    bug_report: str = ""
    stack_trace: str = ""
    code_snippet: str = ""
    available_files: List[str] = Field(default_factory=list)
    history: List[str] = Field(default_factory=list)
    current_step: int = 0
    task_id: str = ""
    max_steps: int = 1
    feedback: str = ""


class BugTriageAction(BaseModel):
    """Structured action to send to the environment."""

    action_type: str
    bug_type: Optional[str] = None
    file: Optional[str] = None
    function: Optional[str] = None
    patch: Optional[str] = None
    run_test: Optional[bool] = None


class BugTriageResetResult(BaseModel):
    """Result returned by env.reset()."""

    observation: BugTriageObservation
    done: bool = False


class BugTriageStepResult(BaseModel):
    """Result returned by env.step()."""

    observation: BugTriageObservation
    reward: Optional[float] = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def _find_free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ═════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT CLIENT
# ═════════════════════════════════════════════════════════════════════════════


class BugTriageEnv:
    """
    Async client for the Bug Triage & Debugging environment.

    Use ``from_docker_image()`` to spin up a Docker container,
    or ``from_url()`` to connect to an already-running server.
    """

    def __init__(self, base_url: str, container_id: Optional[str] = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._container_id = container_id
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=60.0)

    # ── Factory: Docker ──────────────────────────────────────────────────

    @classmethod
    async def from_docker_image(
        cls,
        image_name: str,
        port: Optional[int] = None,
        timeout: int = 120,
    ) -> "BugTriageEnv":
        """
        Start a Docker container running the environment server and return
        a connected BugTriageEnv instance.

        Parameters
        ----------
        image_name : str
            Docker image name (e.g. ``bug-triage-env:latest``).
        port : int, optional
            Host port to bind; auto-selected if *None*.
        timeout : int
            Seconds to wait for the container to become healthy.
        """
        if port is None:
            port = _find_free_port()

        container_id = (
            subprocess.check_output(
                [
                    "docker", "run", "-d",
                    "-p", f"{port}:7860",
                    image_name,
                ],
                stderr=subprocess.STDOUT,
            )
            .decode()
            .strip()
        )

        base_url = f"http://localhost:{port}"

        # Wait for server health
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                async with httpx.AsyncClient() as probe:
                    r = await probe.get(f"{base_url}/health", timeout=5)
                    if r.status_code == 200:
                        break
            except Exception:
                pass
            await asyncio.sleep(1.0)
        else:
            # Cleanup on failure
            subprocess.run(["docker", "stop", container_id], capture_output=True)
            subprocess.run(["docker", "rm", container_id], capture_output=True)
            raise TimeoutError(
                f"Environment container did not become healthy within {timeout}s"
            )

        return cls(base_url=base_url, container_id=container_id)

    # ── Factory: URL ─────────────────────────────────────────────────────

    @classmethod
    async def from_url(cls, base_url: str) -> "BugTriageEnv":
        """Connect to an already-running environment server."""
        return cls(base_url=base_url)

    # ── Core API ─────────────────────────────────────────────────────────

    async def reset(
        self,
        task_id: str = "task_1_classify",
        scenario_index: int = 0,
    ) -> BugTriageResetResult:
        """Reset the environment and begin a new episode."""
        r = await self._client.post(
            "/reset",
            json={"task_id": task_id, "scenario_index": scenario_index},
        )
        r.raise_for_status()
        data = r.json()
        return BugTriageResetResult(
            observation=BugTriageObservation(**data["observation"]),
            done=data.get("done", False),
        )

    async def step(self, action: BugTriageAction) -> BugTriageStepResult:
        """Execute one action and return the result."""
        r = await self._client.post(
            "/step",
            json=action.model_dump(exclude_none=True),
        )
        r.raise_for_status()
        data = r.json()
        return BugTriageStepResult(
            observation=BugTriageObservation(**data["observation"]),
            reward=data.get("reward", 0.0),
            done=data.get("done", False),
            info=data.get("info", {}),
        )

    async def close(self) -> None:
        """Shut down the HTTP client and, if applicable, stop the container."""
        await self._client.aclose()
        if self._container_id:
            subprocess.run(
                ["docker", "stop", self._container_id],
                capture_output=True,
            )
            subprocess.run(
                ["docker", "rm", self._container_id],
                capture_output=True,
            )
