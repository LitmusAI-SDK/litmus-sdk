"""
SQL Agent example — self-contained in examples/sql-agent/.

Run from the repo root:
    python examples/sql-agent/run.py

Prerequisites:
    - OPENAI_API_KEY set in .env or environment
    - pip install -r requirements.txt
    - python examples/sql-agent/create_db.py (creates test_database.db)
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(EXAMPLE_DIR, "..", ".."))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))
CONFIG_PATH = os.path.join(EXAMPLE_DIR, "litmus.toml")

# Add this example's directory to path so sql_agent module is importable
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from litmus_sdk.core import LitmusSDK
from litmus_sdk.testing import GoldenTestRunner

# One shared SDK + DB for both versions (driven by litmus.toml)
sdk = LitmusSDK(config_path=CONFIG_PATH)

# ---------------------------------------------------------------------------
# --- Version 1: baseline run
# Auto-resolves version from LITMUS_VERSION / GITHUB_REF_NAME / etc.
# Falls back to a short UUID if no env var is set (rename later in UI).
# Set LITMUS_VERSION=sql-agent-v1 in .env for a stable label.
# ---------------------------------------------------------------------------

sdk.init()                          # ← zero-arg; env / UUID fallback
v1_label = sdk.version              # capture resolved label for comparison
runner_v1 = GoldenTestRunner(sdk)

# Load golden tests from config + entrypoint
runner_v1.load_from_config(config_path=CONFIG_PATH)

print("=" * 60)
print(f"Running golden tests — {v1_label} (baseline)")
print("=" * 60)
report_v1 = runner_v1.run_tests(num_runs=3)
report_v1.display()

# ---------------------------------------------------------------------------
# --- Version 2: simulated schema prompt change
# Explicit label for this release; in CI bump LITMUS_VERSION in the workflow.
# ---------------------------------------------------------------------------
import sql_agent  # noqa: E402

# Tweak the static prompt (this is what Litmus is designed to detect)
sql_agent.DATABASE_SCHEMA += "\n  8. Always alias aggregations with meaningful names."

sdk.init("sql-agent-v2")           # explicit label for the second version
runner_v2 = GoldenTestRunner(sdk)
runner_v2.load_from_config(config_path=CONFIG_PATH)

print("\n" + "=" * 60)
print("Running golden tests — v2 (modified system prompt)")
print("=" * 60)
report_v2 = runner_v2.run_tests(num_runs=3)
report_v2.display()

# ---------------------------------------------------------------------------
# --- Compare versions
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"Drift comparison: {v1_label} → sql-agent-v2")
print("=" * 60)
drift_report = sdk.compare(v1_label, "sql-agent-v2")
drift_report.display()

sdk.close()
