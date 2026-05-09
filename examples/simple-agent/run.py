"""
Simple Agent example — self-contained in examples/simple-agent/.

Config lives in litmus.toml alongside this file.

Usage
-----
  # Run trace/drift demo:
  python examples/simple-agent/run.py
"""
from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(EXAMPLE_DIR, "..", ".."))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))
CONFIG_PATH = os.path.join(EXAMPLE_DIR, "litmus.toml")

import litellm

from litmus_sdk.core import LitmusSDK

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------
MODEL      = os.getenv("LITMUS_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Your real agent function — replace the body with your actual pipeline.
# The SDK callback is already registered (sdk.init() below), so every
# litellm.completion() call made here is automatically traced.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = "You are a concise, helpful assistant."

def my_agent(question: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """
    Thin wrapper around a single LiteLLM call.
    Replace this with your real pipeline: RAG retrieval, tool calls,
    multi-step reasoning, memory lookups, etc.
    """
    response = litellm.completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


# ===========================================================================
# MODE — Trace / drift demo  (default)
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Initialise — no version argument needed.
#    Set LITMUS_VERSION=v1.0.0 in .env for a stable label, or let it
#    auto-generate a short UUID ("run-<8hex>") that can be renamed in the UI.
# ---------------------------------------------------------------------------
sdk = LitmusSDK(config_path=CONFIG_PATH)
sdk.init()                          # ← auto-resolves; no hard-coded label needed
v1_label = sdk.version              # capture whatever was resolved

print("=" * 60)
print("LitmusAI — simple integration example")
print("=" * 60)

# ---------------------------------------------------------------------------
# 2. Make a few LLM calls via my_agent() (traces captured automatically)
# ---------------------------------------------------------------------------
QUESTIONS = [
    "What is the capital of France?",
    "Explain recursion in one sentence.",
    "What is 12 * 7?",
]

for q in QUESTIONS:
    answer = my_agent(q)
    print(f"Q: {q}")
    print(f"A: {answer}\n")

# ---------------------------------------------------------------------------
# 3. Verify captures
# ---------------------------------------------------------------------------
traces = sdk.get_traces()
print(f"Captured {len(traces)} trace(s) for {v1_label}")

# ---------------------------------------------------------------------------
# 4. Simulate a version bump — explicit label for the new release.
#    In CI you would bump LITMUS_VERSION (or re-tag) instead of hard-coding.
# ---------------------------------------------------------------------------
sdk.init("v2.0.0")

for q in QUESTIONS:
    my_agent(q)
    print(f"✓ Captured v2.0.0 trace for: {q[:40]}...")

time.sleep(0.1)

print("v2.0.0 traces captured:", len(sdk.get_traces("v2.0.0")))

# ---------------------------------------------------------------------------
# 5. Compare versions — v1_label is whatever was auto-resolved above
# ---------------------------------------------------------------------------
print(f"\nRunning drift comparison {v1_label} → v2.0.0 …")
report = sdk.compare(v1_label, "v2.0.0")
report.display()

sdk.close()
print("\nTraces saved to litmus.db — open the dashboard to explore.")
