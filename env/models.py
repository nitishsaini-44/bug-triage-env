"""Pydantic models for Observation, Action, and Reward."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ── Observation ───────────────────────────────────────────────────────────────

class Observation(BaseModel):
    """What the agent sees at each step."""

    bug_report: str = Field(..., description="Natural-language description of the bug.")
    stack_trace: str = Field(default="", description="Stack trace associated with the bug (may be empty).")
    code_snippet: str = Field(default="", description="Relevant source code snippet (may be empty).")
    available_files: List[str] = Field(default_factory=list, description="List of file paths in the project.")
    history: List[str] = Field(default_factory=list, description="Log of actions taken so far in this episode.")
    current_step: int = Field(default=0, ge=0, description="Zero-indexed step counter.")
    task_id: str = Field(default="", description="Identifier of the current task.")
    max_steps: int = Field(default=1, ge=1, description="Maximum steps allowed for this task.")
    feedback: str = Field(default="", description="Feedback from the previous action.")


# ── Action ────────────────────────────────────────────────────────────────────

class Action(BaseModel):
    """Structured action the agent must submit."""

    action_type: str = Field(
        ...,
        description="One of: classify, locate, fix, test.",
        pattern=r"^(classify|locate|fix|test)$",
    )
    bug_type: Optional[str] = Field(
        default=None,
        description="Bug classification label (ui, backend, performance, security).",
    )
    file: Optional[str] = Field(default=None, description="Target file path.")
    function: Optional[str] = Field(default=None, description="Target function name.")
    patch: Optional[str] = Field(default=None, description="Proposed code patch.")
    run_test: Optional[bool] = Field(default=None, description="Whether to execute the test suite.")


# ── Reward ────────────────────────────────────────────────────────────────────

class Reward(BaseModel):
    """Reward signal returned after each step."""

    value: float = Field(..., ge=-1.0, le=1.0, description="Scalar reward value.")
    reason: str = Field(default="", description="Human-readable explanation of the reward.")
