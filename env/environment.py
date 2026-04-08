"""
BugTriageEnv — OpenEnv-compatible RL environment.

API:
    reset(task_id, scenario_index=0)  → Observation
    step(action)                      → (Observation, Reward, done, info)
    state()                           → dict
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

from env.graders import (
    compute_final_score_task3,
    grade_classification,
    grade_debug_step,
    grade_root_cause,
)
from env.models import Action, Observation, Reward
from env.tasks import BugScenario, TaskDefinition, get_task


class BugTriageEnv:
    """OpenEnv-compatible environment for AI bug triage and debugging."""

    # ── construction ─────────────────────────────────────────────────────
    def __init__(self) -> None:
        self._task: Optional[TaskDefinition] = None
        self._scenario: Optional[BugScenario] = None
        self._current_step: int = 0
        self._done: bool = True
        self._total_reward: float = 0.0
        self._rewards: list[float] = []
        self._history: list[str] = []
        self._episode_state: Dict[str, Any] = {}
        self._scenario_index: int = 0

    # ── reset ────────────────────────────────────────────────────────────
    def reset(
        self,
        task_id: str = "task_1_classify",
        scenario_index: int = 0,
    ) -> Observation:
        """
        Start (or restart) an episode.

        Parameters
        ----------
        task_id : str
            One of task_1_classify, task_2_locate, task_3_debug.
        scenario_index : int
            Which scenario within the task to use.

        Returns
        -------
        Observation
        """
        self._task = get_task(task_id)
        if scenario_index < 0 or scenario_index >= len(self._task.scenarios):
            raise IndexError(
                f"scenario_index {scenario_index} out of range "
                f"(0..{len(self._task.scenarios) - 1})"
            )
        self._scenario_index = scenario_index
        self._scenario = self._task.scenarios[scenario_index]
        self._current_step = 0
        self._done = False
        self._total_reward = 0.0
        self._rewards = []
        self._history = []
        self._episode_state = {}

        return self._make_observation(feedback="Episode started. Analyse the bug report.")

    # ── step ─────────────────────────────────────────────────────────────
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one agent action.

        Returns
        -------
        (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before step().")
        if self._task is None or self._scenario is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        # ── grade ────────────────────────────────────────────────────────
        reward = self._grade(action)

        self._total_reward += reward.value
        self._rewards.append(reward.value)
        self._history.append(
            f"step={self._current_step} action={action.action_type} reward={reward.value:.2f}"
        )
        self._current_step += 1

        # ── check termination ────────────────────────────────────────────
        if self._current_step >= self._task.max_steps:
            self._done = True

        # ── build info dict ──────────────────────────────────────────────
        info: Dict[str, Any] = {
            "task_id": self._task.task_id,
            "scenario_id": self._scenario.bug_id,
            "episode_state": copy.deepcopy(self._episode_state),
        }
        if self._done:
            info["final_score"] = self._compute_final_score()
            info["total_reward"] = round(self._total_reward, 4)
            info["rewards"] = list(self._rewards)

        obs = self._make_observation(feedback=reward.reason)
        return obs, reward, self._done, info

    # ── state ────────────────────────────────────────────────────────────
    def state(self) -> Dict[str, Any]:
        """Return the full internal state (for serialisation / debugging)."""
        return {
            "task_id": self._task.task_id if self._task else None,
            "scenario_id": self._scenario.bug_id if self._scenario else None,
            "scenario_index": self._scenario_index,
            "current_step": self._current_step,
            "max_steps": self._task.max_steps if self._task else None,
            "done": self._done,
            "total_reward": round(self._total_reward, 4),
            "rewards": list(self._rewards),
            "history": list(self._history),
            "episode_state": copy.deepcopy(self._episode_state),
        }

    # ── internal helpers ─────────────────────────────────────────────────
    def _make_observation(self, feedback: str = "") -> Observation:
        assert self._task is not None
        assert self._scenario is not None
        return Observation(
            bug_report=self._scenario.bug_report,
            stack_trace=self._scenario.stack_trace,
            code_snippet=self._scenario.code_snippet,
            available_files=list(self._scenario.available_files),
            history=list(self._history),
            current_step=self._current_step,
            task_id=self._task.task_id,
            max_steps=self._task.max_steps,
            feedback=feedback,
        )

    def _grade(self, action: Action) -> Reward:
        assert self._task is not None
        assert self._scenario is not None

        tid = self._task.task_id
        if tid == "task_1_classify":
            return grade_classification(action, self._scenario)
        elif tid == "task_2_locate":
            return grade_root_cause(action, self._scenario)
        elif tid == "task_3_debug":
            return grade_debug_step(
                action, self._scenario, self._current_step, self._episode_state
            )
        else:
            return Reward(value=0.0, reason=f"No grader for task '{tid}'.")

    def _compute_final_score(self) -> float:
        assert self._task is not None
        tid = self._task.task_id
        if tid == "task_1_classify":
            return self._rewards[0] if self._rewards else 0.0
        elif tid == "task_2_locate":
            return self._rewards[0] if self._rewards else 0.0
        elif tid == "task_3_debug":
            return compute_final_score_task3(self._episode_state)
        return 0.0
