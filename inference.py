#!/usr/bin/env python3
"""
Inference Script — AI Bug Triage & Debugging Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from bug_triage_env import BugTriageAction, BugTriageEnv, BugTriageObservation

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME") or "bug-triage-env"
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o"

BENCHMARK = "bug-triage-debug-env"

TASKS = [
    {"task_id": "task_1_classify", "max_steps": 1, "scenario_index": 0},
    {"task_id": "task_2_locate",   "max_steps": 1, "scenario_index": 0},
    {"task_id": "task_3_debug",    "max_steps": 4, "scenario_index": 0},
]

TEMPERATURE = 0.0
MAX_TOKENS = 1024


# ═════════════════════════════════════════════════════════════════════════════
#  LOGGING HELPERS  (exact format required by OpenEnv)
# ═════════════════════════════════════════════════════════════════════════════

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] task={task} success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert software engineer performing bug triage and debugging.
    You must respond with ONLY a valid JSON object.
    No markdown fences, no explanation, no extra text — just raw JSON.
""").strip()


# ═════════════════════════════════════════════════════════════════════════════
#  PROMPT BUILDERS
# ═════════════════════════════════════════════════════════════════════════════

def build_prompt(task_id: str, step: int, obs: BugTriageObservation) -> str:
    """Build the user prompt based on task and step."""

    # ── classify ─────────────────────────────────────────────────────────
    if task_id == "task_1_classify" or (task_id == "task_3_debug" and step == 0):
        parts = [f"Bug report:\n{obs.bug_report}"]
        if obs.stack_trace:
            parts.append(f"Stack trace:\n{obs.stack_trace}")
        if obs.code_snippet:
            parts.append(f"Code snippet:\n{obs.code_snippet}")
        parts.append(f"Available files: {obs.available_files}")
        parts.append(
            "Classify this bug as exactly one of: ui, backend, performance, security.\n"
            'Respond with JSON: {"action_type": "classify", "bug_type": "<label>"}'
        )
        return "\n\n".join(parts)

    # ── locate ───────────────────────────────────────────────────────────
    if task_id == "task_2_locate" or (task_id == "task_3_debug" and step == 1):
        parts = [
            f"Bug report:\n{obs.bug_report}",
            f"Stack trace:\n{obs.stack_trace}",
            f"Code snippet:\n{obs.code_snippet}",
            f"Available files: {obs.available_files}",
        ]
        if obs.feedback:
            parts.append(f"Feedback from previous step: {obs.feedback}")
        parts.append(
            "Identify the faulty file and function where the root cause lies.\n"
            'Respond with JSON: {"action_type": "locate", "file": "<path>", "function": "<name>"}'
        )
        return "\n\n".join(parts)

    # ── fix ──────────────────────────────────────────────────────────────
    if task_id == "task_3_debug" and step == 2:
        parts = [
            f"Bug report:\n{obs.bug_report}",
            f"Code snippet:\n{obs.code_snippet}",
            f"Available files: {obs.available_files}",
            f"Action history: {obs.history}",
            f"Feedback: {obs.feedback}",
            "Propose a code patch that fixes the bug. "
            "Write the corrected version of the faulty code.\n"
            'Respond with JSON: {"action_type": "fix", "patch": "<fixed code>"}',
        ]
        return "\n\n".join(parts)

    # ── test ─────────────────────────────────────────────────────────────
    parts = [
        f"Action history: {obs.history}",
        f"Feedback: {obs.feedback}",
        "Run the test suite to validate your fix.\n"
        'Respond with JSON: {"action_type": "test", "run_test": true}',
    ]
    return "\n\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
#  ACTION PARSING
# ═════════════════════════════════════════════════════════════════════════════

def parse_llm_response(raw: str, task_id: str, step: int) -> BugTriageAction:
    """Parse LLM JSON response into an action, fall back on failure."""
    text = raw.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    for candidate in [text]:
        try:
            data = json.loads(candidate)
            return BugTriageAction(**data)
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end])
            return BugTriageAction(**data)
        except Exception:
            pass

    return _fallback_action(task_id, step)


def _fallback_action(task_id: str, step: int) -> BugTriageAction:
    if task_id == "task_1_classify" or step == 0:
        return BugTriageAction(action_type="classify", bug_type="backend")
    if task_id == "task_2_locate" or step == 1:
        return BugTriageAction(action_type="locate", file="unknown.py", function="unknown")
    if step == 2:
        return BugTriageAction(action_type="fix", patch="# no patch available")
    return BugTriageAction(action_type="test", run_test=True)


def format_action(action: BugTriageAction) -> str:
    if action.action_type == "classify":
        return f"classify(bug_type={action.bug_type})"
    if action.action_type == "locate":
        return f"locate(file={action.file},function={action.function})"
    if action.action_type == "fix":
        preview = (action.patch or "")[:60].replace("\n", "\\n")
        return f"fix(patch={preview})"
    if action.action_type == "test":
        return f"test(run_test={action.run_test})"
    return f"{action.action_type}(...)"


# ═════════════════════════════════════════════════════════════════════════════
#  LLM CALL
# ═════════════════════════════════════════════════════════════════════════════

def get_action_from_llm(client: OpenAI, task_id: str, step: int, obs: BugTriageObservation) -> BugTriageAction:
    prompt = build_prompt(task_id, step, obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        return parse_llm_response(raw, task_id, step)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return _fallback_action(task_id, step)


# ═════════════════════════════════════════════════════════════════════════════
#  TASK RUNNER
# ═════════════════════════════════════════════════════════════════════════════

async def run_task(client: OpenAI, env: BugTriageEnv, task_config: Dict[str, Any]) -> Dict[str, Any]:
    task_id = task_config["task_id"]
    max_steps = task_config["max_steps"]
    scenario_index = task_config.get("scenario_index", 0)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_info: Dict[str, Any] = {}

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id, scenario_index=scenario_index)
        obs = result.observation

        for step_num in range(1, max_steps + 1):
            if result.done:
                break

            action = get_action_from_llm(client, task_id, step_num - 1, obs)
            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step_num
            last_info = result.info

            action_str = format_action(action)
            log_step(step=step_num, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        if task_id == "task_3_debug":
            score = last_info.get("final_score", max(sum(rewards), 0.0))
        else:
            score = sum(rewards)

        score = min(max(score, 0.0), 1.0)
        success = score > 0

    except Exception as exc:
        print(f"[DEBUG] Error in task {task_id}: {exc}", flush=True)

    finally:
        log_end(task=task_id, success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "success": success,
        "steps": steps_taken,
        "score": score,
        "rewards": rewards,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env_url = os.getenv("ENV_URL") or os.getenv("SPACE_URL")
    env = None

    try:
        if env_url:
            print(f"[DEBUG] Using remote OpenEnv server: {env_url}", flush=True)
            env = await BugTriageEnv.from_url(env_url)
        else:
            print(f"[DEBUG] Starting local docker container: {IMAGE_NAME}", flush=True)
            env = await BugTriageEnv.from_docker_image(IMAGE_NAME)

        for task_config in TASKS:
            await run_task(client, env, task_config)

    except Exception as e:
        print(f"[DEBUG] Environment setup failed: {e}", flush=True)
        # FALLBACK: If env crashes before tasks run, we still MUST print [START] and [END] tags
        # so the hackathon parser doesn't crash with the "No structured output" error.
        for task_config in TASKS:
            task_id = task_config["task_id"]
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            log_end(task=task_id, success=False, steps=0, score=0.0, rewards=[])
        
        # Re-raise the exception so it hits the sys.exit(1) below.
        raise e

    finally:
        if env:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)


if __name__ == "__main__":
    import traceback
    try:
        asyncio.run(main())
    except Exception as e:
        traceback.print_exc()
        print(f"[DEBUG] Fatal error in main: {e}", flush=True)
        sys.exit(1)  # CRITICAL FIX: Ensures platform knows the script actually errored out.
