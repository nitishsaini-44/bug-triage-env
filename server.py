"""
FastAPI server — exposes the BugTriageEnv as HTTP endpoints.

Endpoints:
    POST /reset   → reset the environment
    POST /step    → execute one action
    GET  /state   → return current state
    GET  /health  → health check
    GET  /tasks   → list available tasks
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.environment import BugTriageEnv
from env.models import Action, Observation
from env.tasks import TASK_CATALOGUE

app = FastAPI(
    title="AI Bug Triage & Debugging Environment",
    description="OpenEnv-compatible RL environment for AI bug triage.",
    version="1.0.0",
)

# Single global environment instance
env = BugTriageEnv()


# ── Request / Response schemas ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_1_classify"
    scenario_index: int = 0


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "environment": "BugTriageEnv"}


@app.get("/tasks")
def list_tasks():
    return {
        tid: {
            "name": t.name,
            "description": t.description,
            "difficulty": t.difficulty,
            "max_steps": t.max_steps,
            "num_scenarios": len(t.scenarios),
        }
        for tid, t in TASK_CATALOGUE.items()
    }


@app.post("/reset")
def reset(req: ResetRequest):
    try:
        obs = env.reset(task_id=req.task_id, scenario_index=req.scenario_index)
        return {"observation": obs.model_dump(), "done": False}
    except (KeyError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step")
def step(action: Action):
    try:
        obs, reward, done, info = env.step(action)
        return StepResponse(
            observation=obs.model_dump(),
            reward=reward.value,
            done=done,
            info=info,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state")
def state():
    return env.state()
