---
title: Bug Triage Env
emoji: 🐛
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server.py
pinned: false
---

# 🐛 AI Bug Triage & Debugging Environment

> **OpenEnv-compatible RL environment** for training and evaluating AI agents on real-world software engineering tasks — from bug classification through root-cause analysis, patch proposal, and test validation.

---

## 🎯 Motivation

Software teams spend **30–50 % of engineering time** on bug triage and debugging. This environment provides a structured, reproducible simulation of that workflow so AI agents can learn the multi-step reasoning required to:

1. **Classify** incoming bugs by category  
2. **Locate** the faulty file and function  
3. **Propose** a working code fix  
4. **Validate** the fix against automated tests  

Every interaction is deterministically graded, making the environment suitable for reinforcement learning, supervised fine-tuning, and agent benchmarking.

---

## 📂 Project Structure

```
project/
├── env/
│   ├── __init__.py          # Package entry point
│   ├── models.py            # Pydantic models: Observation, Action, Reward
│   ├── tasks.py             # Task definitions & bug scenario datasets
│   ├── graders.py           # Deterministic graders for all 3 tasks
│   └── environment.py       # Core RL environment (reset / step / state)
│
├── server/
│   └── app.py               # Application endpoints
│
├── bug_triage_env.py        # OpenEnv SDK Client Module
├── Dockerfile               # Container build file
├── inference.py             # Baseline LLM inference script handling outputs and logic
├── openenv.yaml             # OpenEnv specification file
├── pyproject.toml           # Python build configuration
├── requirements.txt         # Python dependencies
├── server.py                # FastAPI server exposing the environment to HTTP
├── test_env_full.py         # Local script to test deep RL environment flows
├── test_inference.py        # Simple local test rig for the inference pipeline
├── uv.lock                  # UV package lockfile
├── validate-submission.sh   # Evaluation validation script
└── README.md                # This file
```

---

## 🕹️ Action Space

All actions are **structured JSON objects** — no free-form text allowed.

| Field         | Type     | Values / Description                              |
|---------------|----------|---------------------------------------------------|
| `action_type` | `string` | `classify` \| `locate` \| `fix` \| `test`         |
| `bug_type`    | `string` | `ui` \| `backend` \| `performance` \| `security`  |
| `file`        | `string` | Path to the suspected faulty file                  |
| `function`    | `string` | Name of the suspected faulty function              |
| `patch`       | `string` | Proposed code fix                                  |
| `run_test`    | `bool`   | Whether to execute the test suite                  |

**Example:**

```json
{
  "action_type": "classify",
  "bug_type": "security"
}
```

---

## 👁️ Observation Space

| Field             | Type           | Description                           |
|-------------------|----------------|---------------------------------------|
| `bug_report`      | `string`       | Natural-language bug description      |
| `stack_trace`     | `string`       | Stack trace (may be empty)            |
| `code_snippet`    | `string`       | Relevant source code                  |
| `available_files` | `list[string]` | Files in the simulated project        |
| `history`         | `list[string]` | Log of previous actions in episode    |
| `current_step`    | `int`          | Zero-indexed step counter             |
| `task_id`         | `string`       | Current task identifier               |
| `max_steps`       | `int`          | Maximum steps for this episode        |
| `feedback`        | `string`       | Feedback from the previous action     |

---

## 📋 Tasks

### 🟢 Task 1 — Bug Classification (Easy)

- **Input:** Bug report text  
- **Output:** One of `ui`, `backend`, `performance`, `security`  
- **Steps:** 1  
- **Scenarios:** 8 diverse, real-world bugs  
- **Grading:** correct → `0.99`, wrong → `0.01` (strictly between 0 and 1)

### 🟡 Task 2 — Root Cause Identification (Medium)

- **Input:** Bug report + stack trace + code snippet  
- **Output:** Faulty file path and function name  
- **Steps:** 1  
- **Scenarios:** 4 scenarios with realistic traces  
- **Grading:** correct file → `+0.49`, correct function → `+0.5`

### 🔴 Task 3 — Multi-Step Debugging (Hard)

- **Input:** Full bug context  
- **Steps:** 4 (classify → locate → fix → test)  
- **Scenarios:** 3 complex, multi-step scenarios  
- **Grading:**

| Step | Reward  |
|------|---------|
| Correct classification | +0.2 |
| Correct file + function | +0.3 |
| Correct fix (≥60% similarity) | +0.3 |
| Passing test | +0.2 |
| Wrong action type | −0.1 |
| Incorrect action | −0.2 |

---

## 🏆 Reward Function

Rewards are provided **at each step**, not just at episode end:

| Event | Reward |
|-------|--------|
| Correct classification | `+0.2` |
| Correct file identification | `+0.15` |
| Correct function identification | `+0.15` |
| Correct fix (≥60% match) | `+0.3` |
| Partial fix (≥30% match) | `+0.1` |
| All tests pass | `+0.2` |
| Wrong action | `−0.2` |
| Random / irrelevant action | `−0.1` |

All graders are **deterministic and reproducible**.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the environment programmatically

```python
from env.environment import BugTriageEnv
from env.models import Action

env = BugTriageEnv()

# Task 1 — classify
obs = env.reset(task_id="task_1_classify", scenario_index=0)
action = Action(action_type="classify", bug_type="ui")
obs, reward, done, info = env.step(action)
print(f"Reward: {reward.value}, Done: {done}")

# Task 3 — multi-step debugging
obs = env.reset(task_id="task_3_debug", scenario_index=0)
for step_action in [
    Action(action_type="classify", bug_type="security"),
    Action(action_type="locate", file="src/api/notes.py", function="get_note"),
    Action(action_type="fix", patch="...fixed code..."),
    Action(action_type="test", run_test=True),
]:
    obs, reward, done, info = env.step(step_action)
    print(f"Step {obs.current_step}: reward={reward.value:.2f}, done={done}")
```

### 3. Run the baseline inference script

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4"
export OPENAI_API_KEY="sk-..."
python inference.py
```

### 4. Run via HTTP server

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

Then interact with the API:

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_1_classify", "scenario_index": 0}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "classify", "bug_type": "ui"}'

# State
curl http://localhost:7860/state
```

### 5. Docker

```bash
docker build -t bug-triage-env .
docker run -p 7860:7860 bug-triage-env
```

---

## 🤗 Hugging Face Deployment

This environment is designed to deploy as a **Hugging Face Space** (Docker SDK):

1. Create a new Space with **Docker** SDK  
2. Push this repository  
3. The server starts on port `7860` and responds to `/reset`, `/step`, and `/state`

---

## 📊 Baseline Results

Results with GPT-4 (temperature=0):

| Task | Score | Steps |
|------|-------|-------|
| Bug Classification | 1.0 | 1 |
| Root Cause Identification | 1.0 | 1 |
| Multi-Step Debugging | 0.85 | 4 |

*Scores are deterministic and reproducible given the same model outputs.*

---

## ⚙️ Resource Constraints

| Resource | Limit |
|----------|-------|
| vCPU | 2 |
| RAM | 8 GB |
| Max runtime | 20 minutes |

---

## 📜 License

MIT
