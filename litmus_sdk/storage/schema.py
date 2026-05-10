"""
SQLite schema definition and initialization.
"""

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS traces (
    trace_id            TEXT PRIMARY KEY,
    run_id              TEXT,
    version             TEXT NOT NULL,
    project_id          TEXT,
    timestamp           TEXT NOT NULL,
    system_prompt       TEXT,
    user_input          TEXT,
    context             TEXT,
    final_prompt        TEXT,
    model               TEXT,
    temperature         REAL,
    response            TEXT,
    tokens_input        INTEGER DEFAULT 0,
    tokens_output       INTEGER DEFAULT 0,
    latency_ms          REAL    DEFAULT 0,
    cost                REAL    DEFAULT 0,
    agent_name          TEXT,
    step_number         INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_traces_version    ON traces(version);
CREATE INDEX IF NOT EXISTS idx_traces_run_id     ON traces(run_id);
CREATE INDEX IF NOT EXISTS idx_traces_ts         ON traces(timestamp);

CREATE TABLE IF NOT EXISTS test_cases (
    test_id             TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    project_id          TEXT,
    inputs              TEXT NOT NULL,
    expected_pattern    TEXT,
    expected_behavior   TEXT,
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS test_runs (
    run_id      TEXT NOT NULL,
    version     TEXT NOT NULL,
    project_id  TEXT,
    test_id     TEXT NOT NULL,
    run_number  INTEGER NOT NULL,
    output      TEXT,
    passed      INTEGER NOT NULL DEFAULT 0,
    latency_ms  REAL    DEFAULT 0,
    error       TEXT,
    timestamp   TEXT NOT NULL,
    FOREIGN KEY (test_id) REFERENCES test_cases(test_id)
);

CREATE INDEX IF NOT EXISTS idx_test_runs_version    ON test_runs(version);
CREATE INDEX IF NOT EXISTS idx_test_runs_test_id    ON test_runs(test_id);

CREATE TABLE IF NOT EXISTS version_snapshots (
    version                 TEXT PRIMARY KEY,
    project_id              TEXT,
    created_at              TEXT NOT NULL,
    pass_rate               REAL,
    avg_latency_ms          REAL,
    avg_cost                REAL,
    stability_score         REAL,
    notes                   TEXT
);

CREATE TABLE IF NOT EXISTS projects (
    project_id  TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT DEFAULT '',
    agent_url   TEXT DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS baselines (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT NOT NULL,
    test_id     TEXT NOT NULL,
    version     TEXT NOT NULL,
    run_id      TEXT,
    output      TEXT,
    approved    INTEGER NOT NULL DEFAULT 1,
    set_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (test_id) REFERENCES test_cases(test_id)
);
CREATE INDEX IF NOT EXISTS idx_baselines_test_id    ON baselines(test_id);
CREATE INDEX IF NOT EXISTS idx_baselines_project_id ON baselines(project_id);

CREATE TABLE IF NOT EXISTS runners (
    runner_id       TEXT PRIMARY KEY,
    project_id      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'online',
    last_heartbeat  TEXT NOT NULL DEFAULT (datetime('now')),
    registered_at   TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (project_id) REFERENCES projects(project_id)
);
CREATE INDEX IF NOT EXISTS idx_runners_project_id ON runners(project_id);

CREATE TABLE IF NOT EXISTS run_queue (
    queue_id        TEXT PRIMARY KEY,
    project_id      TEXT NOT NULL,
    version         TEXT NOT NULL,
    num_runs        INTEGER NOT NULL DEFAULT 3,
    status          TEXT NOT NULL DEFAULT 'pending',
    runner_id       TEXT,
    queued_at       TEXT NOT NULL DEFAULT (datetime('now')),
    claimed_at      TEXT,
    completed_at    TEXT,
    result_summary  TEXT,
    FOREIGN KEY (project_id) REFERENCES projects(project_id)
);
CREATE INDEX IF NOT EXISTS idx_run_queue_project_id ON run_queue(project_id);
CREATE INDEX IF NOT EXISTS idx_run_queue_status     ON run_queue(status);

CREATE TABLE IF NOT EXISTS judge_verdicts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id          TEXT NOT NULL,
    version_a           TEXT NOT NULL,
    version_b           TEXT NOT NULL,
    test_name           TEXT NOT NULL,
    verdict             TEXT NOT NULL,
    confidence          REAL NOT NULL DEFAULT 0.0,
    score               REAL NOT NULL DEFAULT 0.5,
    reasoning           TEXT,
    output_a_sample     TEXT,
    output_b_sample     TEXT,
    expected_behavior   TEXT,
    judge_model         TEXT,
    judge_tokens_input  INTEGER DEFAULT 0,
    judge_tokens_output INTEGER DEFAULT 0,
    judge_cost          REAL DEFAULT 0.0,
    judge_latency_ms    REAL DEFAULT 0.0,
    timestamp           TEXT NOT NULL DEFAULT (datetime('now')),
    error               TEXT
);
CREATE INDEX IF NOT EXISTS idx_judge_verdicts_versions
    ON judge_verdicts(project_id, version_a, version_b);
"""


def init_schema(conn) -> None:
    """Initialize SQLite schema."""
    # executescript() commits any pending transaction itself;
    # calling conn.commit() afterwards raises in Python 3.12 WAL mode.
    conn.executescript(SCHEMA_SQL)

    # --- Migration: add project_id columns to existing databases ---
    _migrate_add_project_id(conn)

    # --- Migration: add agent_url to projects ---
    _migrate_add_agent_url(conn)

    # Create project-specific indexes AFTER migration so old DBs work.
    _ensure_project_indexes(conn)


def _migrate_add_project_id(conn) -> None:
    """Add project_id column to existing tables that don't have it yet."""
    tables = ["traces", "test_cases", "test_runs", "version_snapshots"]
    for table in tables:
        try:
            cols = [
                row[1]
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            ]
            if "project_id" not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN project_id TEXT")
        except Exception:
            pass  # table might not exist yet (handled by CREATE IF NOT EXISTS)


def _migrate_add_agent_url(conn) -> None:
    """Add agent_url column to projects table if missing."""
    try:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(projects)").fetchall()]
        if "agent_url" not in cols:
            conn.execute("ALTER TABLE projects ADD COLUMN agent_url TEXT DEFAULT ''")
            conn.commit()
    except Exception:
        pass


def _ensure_project_indexes(conn) -> None:
    """Create project_id indexes after project_id columns are guaranteed."""
    conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_project_id ON traces(project_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_test_runs_project_id ON test_runs(project_id)")
