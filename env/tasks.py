"""Task definitions — three difficulty tiers with realistic bug data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Individual Bug Scenario ──────────────────────────────────────────────────

@dataclass
class BugScenario:
    """A single bug scenario used inside a task."""

    bug_id: str
    bug_report: str
    bug_type: str  # ui | backend | performance | security
    stack_trace: str = ""
    code_snippet: str = ""
    faulty_file: str = ""
    faulty_function: str = ""
    correct_patch: str = ""
    test_command: str = ""
    available_files: List[str] = field(default_factory=list)


# ── Task definition ──────────────────────────────────────────────────────────

@dataclass
class TaskDefinition:
    """Metadata for one evaluation task."""

    task_id: str
    name: str
    description: str
    difficulty: str  # easy | medium | hard
    max_steps: int
    scenarios: List[BugScenario] = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════════
#  BUG SCENARIO DATASETS
# ═════════════════════════════════════════════════════════════════════════════

# ── Easy scenarios (classification only) ─────────────────────────────────────

EASY_SCENARIOS: List[BugScenario] = [
    BugScenario(
        bug_id="easy-001",
        bug_report=(
            "After updating to v2.3, the login button on the dashboard page is "
            "invisible on mobile devices.  The button's CSS media query appears "
            "to be missing a max-width breakpoint."
        ),
        bug_type="ui",
        available_files=["src/components/LoginButton.tsx", "src/styles/dashboard.css"],
    ),
    BugScenario(
        bug_id="easy-002",
        bug_report=(
            "The /api/users endpoint returns a 500 Internal Server Error when the "
            "request body contains a Unicode surname.  The ORM serialiser does not "
            "handle multi-byte characters."
        ),
        bug_type="backend",
        available_files=["src/api/users.py", "src/models/user.py"],
    ),
    BugScenario(
        bug_id="easy-003",
        bug_report=(
            "Loading the analytics dashboard takes over 12 seconds.  Profiling "
            "shows the SQL query in `get_monthly_stats` performs a full table scan "
            "on the events table (50 M rows) because the date index is missing."
        ),
        bug_type="performance",
        available_files=["src/analytics/queries.py", "src/models/event.py"],
    ),
    BugScenario(
        bug_id="easy-004",
        bug_report=(
            "User-supplied SVG avatars are rendered directly into the page HTML "
            "without sanitisation, allowing embedded <script> tags to execute in "
            "other users' browsers (stored XSS)."
        ),
        bug_type="security",
        available_files=["src/components/Avatar.tsx", "src/api/upload.py"],
    ),
    BugScenario(
        bug_id="easy-005",
        bug_report=(
            "The date picker widget does not render correctly in Safari — the "
            "drop-down is clipped by the parent container's overflow:hidden rule."
        ),
        bug_type="ui",
        available_files=["src/components/DatePicker.tsx", "src/styles/forms.css"],
    ),
    BugScenario(
        bug_id="easy-006",
        bug_report=(
            "Passwords are stored as SHA-1 hashes without salt.  An attacker who "
            "gains read access to the database can reverse most passwords with a "
            "rainbow-table lookup."
        ),
        bug_type="security",
        available_files=["src/auth/hash.py", "src/models/user.py"],
    ),
    BugScenario(
        bug_id="easy-007",
        bug_report=(
            "The background job that sends weekly digest e-mails allocates a new "
            "SMTP connection for every recipient instead of reusing a pool, causing "
            "the worker to OOM when the mailing list exceeds 100 k subscribers."
        ),
        bug_type="performance",
        available_files=["src/jobs/digest.py", "src/email/client.py"],
    ),
    BugScenario(
        bug_id="easy-008",
        bug_report=(
            "Creating a new project via POST /api/projects fails with a 422 "
            "Unprocessable Entity.  The Pydantic model expects `owner_id` as an "
            "integer, but the front-end sends it as a UUID string."
        ),
        bug_type="backend",
        available_files=["src/api/projects.py", "src/models/project.py"],
    ),
]

# ── Medium scenarios (root-cause identification) ─────────────────────────────

MEDIUM_SCENARIOS: List[BugScenario] = [
    BugScenario(
        bug_id="medium-001",
        bug_report=(
            "Users report intermittent 500 errors when uploading profile images "
            "larger than 5 MB.  The error occurs roughly 30 % of the time."
        ),
        bug_type="backend",
        stack_trace=(
            "Traceback (most recent call last):\n"
            '  File "src/api/upload.py", line 47, in upload_image\n'
            "    data = request.files['image'].read()\n"
            '  File "src/storage/s3.py", line 112, in save_blob\n'
            "    bucket.put_object(Key=key, Body=data)\n"
            "botocore.exceptions.ClientError: An error occurred (EntityTooLarge)"
        ),
        code_snippet=(
            "# src/storage/s3.py  (lines 100-120)\n"
            "class S3Storage:\n"
            "    MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB\n\n"
            "    def save_blob(self, key: str, data: bytes) -> str:\n"
            "        if len(data) > self.MAX_UPLOAD_SIZE:\n"
            "            # BUG: should use multipart upload for large files\n"
            "            raise ValueError('File too large')\n"
            "        self.bucket.put_object(Key=key, Body=data)\n"
            "        return f's3://{self.bucket_name}/{key}'\n"
        ),
        faulty_file="src/storage/s3.py",
        faulty_function="save_blob",
        available_files=[
            "src/api/upload.py",
            "src/storage/s3.py",
            "src/storage/local.py",
            "src/models/image.py",
        ],
    ),
    BugScenario(
        bug_id="medium-002",
        bug_report=(
            "The search endpoint returns duplicate results when the query string "
            "contains accented characters like 'café'."
        ),
        bug_type="backend",
        stack_trace=(
            "No crash — the endpoint returns HTTP 200 but the JSON payload "
            "contains duplicate entries with ids [14, 14, 14]."
        ),
        code_snippet=(
            "# src/search/engine.py  (lines 60-80)\n"
            "def search(query: str, limit: int = 20) -> list[dict]:\n"
            "    normalised = query.lower()\n"
            "    # BUG: should also apply unicodedata.normalize('NFKD', ...)\n"
            "    results = db.execute(\n"
            "        'SELECT * FROM items WHERE name LIKE %s',\n"
            "        (f'%{normalised}%',)\n"
            "    )\n"
            "    return [dict(r) for r in results]\n"
        ),
        faulty_file="src/search/engine.py",
        faulty_function="search",
        available_files=[
            "src/search/engine.py",
            "src/search/indexer.py",
            "src/api/search.py",
            "src/models/item.py",
        ],
    ),
    BugScenario(
        bug_id="medium-003",
        bug_report=(
            "The nightly cron job that computes billing totals occasionally "
            "double-charges customers.  Finance noticed the discrepancy in the "
            "March invoice run."
        ),
        bug_type="backend",
        stack_trace=(
            "Traceback (most recent call last):\n"
            '  File "src/billing/invoicer.py", line 88, in generate_invoice\n'
            "    total = sum(line.amount for line in lines)\n"
            '  File "src/billing/ledger.py", line 34, in get_unbilled_lines\n'
            "    return session.query(LineItem).filter(\n"
            "        LineItem.billed == False\n"
            "    ).all()\n"
            "sqlalchemy.exc.OperationalError: database is locked"
        ),
        code_snippet=(
            "# src/billing/ledger.py  (lines 28-45)\n"
            "def get_unbilled_lines(self, customer_id: int) -> list[LineItem]:\n"
            "    lines = session.query(LineItem).filter(\n"
            "        LineItem.customer_id == customer_id,\n"
            "        LineItem.billed == False\n"
            "    ).all()\n"
            "    # BUG: marking billed outside the same transaction\n"
            "    for line in lines:\n"
            "        line.billed = True\n"
            "    session.commit()\n"
            "    return lines\n"
        ),
        faulty_file="src/billing/ledger.py",
        faulty_function="get_unbilled_lines",
        available_files=[
            "src/billing/invoicer.py",
            "src/billing/ledger.py",
            "src/models/line_item.py",
            "src/api/billing.py",
        ],
    ),
    BugScenario(
        bug_id="medium-004",
        bug_report=(
            "After deploying the new caching layer, the homepage shows stale "
            "product prices — sometimes hours out of date — even though the cache "
            "TTL is set to 60 seconds."
        ),
        bug_type="performance",
        stack_trace="",
        code_snippet=(
            "# src/cache/store.py  (lines 15-35)\n"
            "class CacheStore:\n"
            "    def get(self, key: str):\n"
            "        entry = self._store.get(key)\n"
            "        if entry is None:\n"
            "            return None\n"
            "        # BUG: comparing timestamp in seconds vs milliseconds\n"
            "        if time.time() - entry['ts'] > self.ttl:\n"
            "            del self._store[key]\n"
            "            return None\n"
            "        return entry['value']\n\n"
            "    def set(self, key: str, value):\n"
            "        self._store[key] = {\n"
            "            'value': value,\n"
            "            'ts': int(time.time() * 1000),  # <-- milliseconds!\n"
            "        }\n"
        ),
        faulty_file="src/cache/store.py",
        faulty_function="set",
        available_files=[
            "src/cache/store.py",
            "src/cache/invalidator.py",
            "src/api/products.py",
            "src/models/product.py",
        ],
    ),
]

# ── Hard scenarios (multi-step debugging) ────────────────────────────────────

HARD_SCENARIOS: List[BugScenario] = [
    BugScenario(
        bug_id="hard-001",
        bug_report=(
            "Authenticated users can access other users' private notes by "
            "changing the note ID in the URL (/api/notes/<id>).  The endpoint "
            "does not verify that the requesting user owns the note."
        ),
        bug_type="security",
        stack_trace=(
            "GET /api/notes/42  HTTP 200\n"
            "Response body contains note belonging to user_id=7, but the "
            "request was made by user_id=12."
        ),
        code_snippet=(
            "# src/api/notes.py  (lines 20-40)\n"
            "@app.route('/api/notes/<int:note_id>')\n"
            "@login_required\n"
            "def get_note(note_id: int):\n"
            "    note = Note.query.get_or_404(note_id)\n"
            "    # BUG: missing ownership check\n"
            "    return jsonify(note.to_dict())\n"
        ),
        faulty_file="src/api/notes.py",
        faulty_function="get_note",
        correct_patch=(
            "@app.route('/api/notes/<int:note_id>')\n"
            "@login_required\n"
            "def get_note(note_id: int):\n"
            "    note = Note.query.get_or_404(note_id)\n"
            "    if note.user_id != current_user.id:\n"
            "        abort(403)\n"
            "    return jsonify(note.to_dict())\n"
        ),
        test_command="pytest tests/test_notes.py -k test_access_control",
        available_files=[
            "src/api/notes.py",
            "src/models/note.py",
            "src/auth/decorators.py",
            "tests/test_notes.py",
        ],
    ),
    BugScenario(
        bug_id="hard-002",
        bug_report=(
            "The real-time notification system drops messages when more than "
            "100 users are connected simultaneously.  WebSocket connections "
            "are silently closed by the server after ~30 seconds of inactivity."
        ),
        bug_type="performance",
        stack_trace=(
            "WARNING:websockets:connection handler failed\n"
            "  File \"src/ws/handler.py\", line 55, in _dispatch\n"
            "    await asyncio.wait_for(ws.recv(), timeout=30)\n"
            "asyncio.TimeoutError\n"
        ),
        code_snippet=(
            "# src/ws/handler.py  (lines 45-70)\n"
            "class NotificationHandler:\n"
            "    async def _dispatch(self, ws):\n"
            "        while True:\n"
            "            try:\n"
            "                # BUG: timeout too aggressive; should send pings\n"
            "                msg = await asyncio.wait_for(ws.recv(), timeout=30)\n"
            "                await self._process(msg)\n"
            "            except asyncio.TimeoutError:\n"
            "                await ws.close()\n"
            "                break\n"
        ),
        faulty_file="src/ws/handler.py",
        faulty_function="_dispatch",
        correct_patch=(
            "class NotificationHandler:\n"
            "    async def _dispatch(self, ws):\n"
            "        while True:\n"
            "            try:\n"
            "                msg = await asyncio.wait_for(ws.recv(), timeout=120)\n"
            "                await self._process(msg)\n"
            "            except asyncio.TimeoutError:\n"
            "                # send a ping instead of closing\n"
            "                try:\n"
            "                    pong = await ws.ping()\n"
            "                    await asyncio.wait_for(pong, timeout=10)\n"
            "                except Exception:\n"
            "                    await ws.close()\n"
            "                    break\n"
        ),
        test_command="pytest tests/test_ws.py -k test_idle_connection",
        available_files=[
            "src/ws/handler.py",
            "src/ws/manager.py",
            "src/api/notifications.py",
            "tests/test_ws.py",
        ],
    ),
    BugScenario(
        bug_id="hard-003",
        bug_report=(
            "The CSV export feature produces garbled output when the data "
            "contains commas or newlines inside quoted fields.  Customers "
            "report that Excel cannot parse the exported file."
        ),
        bug_type="backend",
        stack_trace=(
            "No crash — but downstream parsing fails.\n"
            "csv.Error: new-line character seen in unquoted field\n"
        ),
        code_snippet=(
            "# src/export/csv_writer.py  (lines 10-30)\n"
            "def write_csv(rows: list[dict], output) -> None:\n"
            "    headers = rows[0].keys()\n"
            "    output.write(','.join(headers) + '\\n')\n"
            "    for row in rows:\n"
            "        # BUG: not quoting fields that contain commas/newlines\n"
            "        line = ','.join(str(row[h]) for h in headers)\n"
            "        output.write(line + '\\n')\n"
        ),
        faulty_file="src/export/csv_writer.py",
        faulty_function="write_csv",
        correct_patch=(
            "import csv\n\n"
            "def write_csv(rows: list[dict], output) -> None:\n"
            "    headers = list(rows[0].keys())\n"
            "    writer = csv.DictWriter(output, fieldnames=headers, quoting=csv.QUOTE_ALL)\n"
            "    writer.writeheader()\n"
            "    writer.writerows(rows)\n"
        ),
        test_command="pytest tests/test_export.py -k test_csv_special_chars",
        available_files=[
            "src/export/csv_writer.py",
            "src/export/xlsx_writer.py",
            "src/api/export.py",
            "tests/test_export.py",
        ],
    ),
]


# ═════════════════════════════════════════════════════════════════════════════
#  TASK CATALOGUE
# ═════════════════════════════════════════════════════════════════════════════

TASK_CATALOGUE: Dict[str, TaskDefinition] = {
    "task_1_classify": TaskDefinition(
        task_id="task_1_classify",
        name="Bug Classification",
        description=(
            "Given a bug report, classify the bug into one of four categories: "
            "ui, backend, performance, or security."
        ),
        difficulty="easy",
        max_steps=1,
        scenarios=EASY_SCENARIOS,
    ),
    "task_2_locate": TaskDefinition(
        task_id="task_2_locate",
        name="Root Cause Identification",
        description=(
            "Given a bug report, stack trace, and code snippet, identify the "
            "file and function where the root cause lies."
        ),
        difficulty="medium",
        max_steps=1,
        scenarios=MEDIUM_SCENARIOS,
    ),
    "task_3_debug": TaskDefinition(
        task_id="task_3_debug",
        name="Multi-Step Debugging",
        description=(
            "Perform end-to-end debugging: classify the bug, locate the faulty "
            "file, propose a fix, and validate with a test."
        ),
        difficulty="hard",
        max_steps=4,
        scenarios=HARD_SCENARIOS,
    ),
}


def get_task(task_id: str) -> TaskDefinition:
    """Return a task by ID, raising KeyError for unknown tasks."""
    if task_id not in TASK_CATALOGUE:
        raise KeyError(f"Unknown task '{task_id}'. Available: {list(TASK_CATALOGUE)}")
    return TASK_CATALOGUE[task_id]


def list_tasks() -> List[str]:
    """Return sorted list of task IDs."""
    return sorted(TASK_CATALOGUE)
