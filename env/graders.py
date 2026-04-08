"""Deterministic graders for each task tier."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List

from env.models import Action, Reward
from env.tasks import BugScenario


def _normalise(s: str | None) -> str:
    """Lower-case, strip, and collapse whitespace for fuzzy comparison."""
    if s is None:
        return ""
    return " ".join(s.lower().split())


def _patch_similarity(proposed: str | None, reference: str) -> float:
    """Return 0.0-1.0 similarity between proposed patch and reference patch."""
    if not proposed:
        return 0.0
    return SequenceMatcher(None, _normalise(proposed), _normalise(reference)).ratio()


# ═════════════════════════════════════════════════════════════════════════════
#  TASK 1 — Bug Classification
# ═════════════════════════════════════════════════════════════════════════════

def grade_classification(action: Action, scenario: BugScenario) -> Reward:
    """
    Task 1 grader.
    • Correct classification → 1.0
    • Wrong classification  → 0.0
    """
    correct = _normalise(scenario.bug_type) == _normalise(action.bug_type)
    return Reward(
        value=1.0 if correct else 0.0,
        reason=(
            f"Classification {'correct' if correct else 'incorrect'}: "
            f"expected '{scenario.bug_type}', got '{action.bug_type}'."
        ),
    )


# ═════════════════════════════════════════════════════════════════════════════
#  TASK 2 — Root Cause Identification
# ═════════════════════════════════════════════════════════════════════════════

def grade_root_cause(action: Action, scenario: BugScenario) -> Reward:
    """
    Task 2 grader.
    • Correct file     → +0.5
    • Correct function → +0.5
    """
    file_correct = _normalise(action.file) == _normalise(scenario.faulty_file)
    func_correct = _normalise(action.function) == _normalise(scenario.faulty_function)

    score = 0.0
    reasons: List[str] = []

    if file_correct:
        score += 0.5
        reasons.append("file match ✓")
    else:
        reasons.append(f"file mismatch: expected '{scenario.faulty_file}', got '{action.file}'")

    if func_correct:
        score += 0.5
        reasons.append("function match ✓")
    else:
        reasons.append(
            f"function mismatch: expected '{scenario.faulty_function}', got '{action.function}'"
        )

    return Reward(value=score, reason="; ".join(reasons))


# ═════════════════════════════════════════════════════════════════════════════
#  TASK 3 — Multi-Step Debugging  (per-step grading)
# ═════════════════════════════════════════════════════════════════════════════

def grade_debug_step(
    action: Action,
    scenario: BugScenario,
    current_step: int,
    episode_state: Dict[str, Any],
) -> Reward:
    """
    Task 3 grader — gives feedback at EACH step.

    Step 0: classify  → +0.2 / -0.2
    Step 1: locate    → +0.3 / -0.2
    Step 2: fix       → +0.3 / -0.2
    Step 3: test      → +0.2 / -0.2
    Wrong action_type → -0.1 (random/irrelevant penalty)
    """
    EXPECTED_TYPES = ["classify", "locate", "fix", "test"]

    # ── Wrong action type for this step ──────────────────────────────────
    expected_type = EXPECTED_TYPES[min(current_step, len(EXPECTED_TYPES) - 1)]
    if action.action_type != expected_type:
        return Reward(
            value=-0.1,
            reason=f"Expected action '{expected_type}' at step {current_step}, got '{action.action_type}'.",
        )

    # ── Step 0: classify ─────────────────────────────────────────────────
    if current_step == 0:
        correct = _normalise(action.bug_type) == _normalise(scenario.bug_type)
        episode_state["classify_correct"] = correct
        return Reward(
            value=0.2 if correct else -0.2,
            reason=(
                f"Classification {'correct' if correct else 'wrong'}: "
                f"expected '{scenario.bug_type}', got '{action.bug_type}'."
            ),
        )

    # ── Step 1: locate ───────────────────────────────────────────────────
    if current_step == 1:
        file_ok = _normalise(action.file) == _normalise(scenario.faulty_file)
        func_ok = _normalise(action.function) == _normalise(scenario.faulty_function)
        both = file_ok and func_ok
        episode_state["locate_correct"] = both

        score = 0.0
        parts: List[str] = []
        if file_ok:
            score += 0.15
            parts.append("file ✓")
        else:
            parts.append(f"file ✗ (expected '{scenario.faulty_file}')")
        if func_ok:
            score += 0.15
            parts.append("function ✓")
        else:
            parts.append(f"function ✗ (expected '{scenario.faulty_function}')")

        if not both:
            score = -0.2

        return Reward(
            value=round(score, 2),
            reason=f"Root-cause location: {'; '.join(parts)}.",
        )

    # ── Step 2: fix ──────────────────────────────────────────────────────
    if current_step == 2:
        sim = _patch_similarity(action.patch, scenario.correct_patch)
        episode_state["patch_similarity"] = sim
        if sim >= 0.6:
            return Reward(value=0.3, reason=f"Patch accepted (similarity={sim:.2f}).")
        elif sim >= 0.3:
            return Reward(value=0.1, reason=f"Partial patch match (similarity={sim:.2f}).")
        else:
            return Reward(value=-0.2, reason=f"Patch rejected (similarity={sim:.2f}).")

    # ── Step 3: test ─────────────────────────────────────────────────────
    if current_step == 3:
        # Simulate test result based on accumulated correctness
        patch_sim = episode_state.get("patch_similarity", 0.0)
        test_passes = (
            action.run_test is True
            and episode_state.get("classify_correct", False)
            and episode_state.get("locate_correct", False)
            and patch_sim >= 0.6
        )
        episode_state["test_passed"] = test_passes
        if test_passes:
            return Reward(value=0.2, reason="All tests passed ✓.")
        elif action.run_test is True:
            return Reward(value=-0.1, reason="Tests ran but FAILED (earlier steps were incorrect).")
        else:
            return Reward(value=-0.2, reason="Tests were not executed (run_test was not True).")

    # Should not reach here
    return Reward(value=-0.1, reason="Step index out of expected range.")


# ═════════════════════════════════════════════════════════════════════════════
#  Final-episode scoring (used by the top-level grader endpoint)
# ═════════════════════════════════════════════════════════════════════════════

def compute_final_score_task3(episode_state: Dict[str, Any]) -> float:
    """
    Deterministic final score for Task 3.
    • Correct classification → 0.2
    • Correct file+function  → 0.3
    • Good patch             → 0.3
    • Passing test           → 0.2
    """
    score = 0.0
    if episode_state.get("classify_correct"):
        score += 0.2
    if episode_state.get("locate_correct"):
        score += 0.3
    sim = episode_state.get("patch_similarity", 0.0)
    if sim >= 0.6:
        score += 0.3
    elif sim >= 0.3:
        score += 0.15
    if episode_state.get("test_passed"):
        score += 0.2
    return round(score, 2)
