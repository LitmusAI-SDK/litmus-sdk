"""
Drift example — self-contained in examples/drift-demo/.

Runs the same golden tests under two prompt versions, then compares
embedding drift between them.

Run with:
    OPENAI_API_KEY=sk-... python examples/drift-demo/run.py
"""

import os

from dotenv import load_dotenv

import litellm
from litmus_sdk import LitmusSDK
from litmus_sdk.testing import GoldenTestRunner


EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(EXAMPLE_DIR, "..", ".."))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))
CONFIG_PATH = os.path.join(EXAMPLE_DIR, "litmus.toml")


# ---------------------------------------------------------------------------
# Shared agent function (simulates the thing under test)
# ---------------------------------------------------------------------------

def generate_answer(question: str, system_prompt: str = "You are a helpful assistant.") -> str:
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Shared SDK instance and database
# ---------------------------------------------------------------------------

# Use config-driven SDK settings for project/db/chroma
litmus = LitmusSDK(config_path=CONFIG_PATH)


# ---------------------------------------------------------------------------
# Version A: baseline prompt
# Auto-resolves version from LITMUS_VERSION / GITHUB_REF_NAME / etc.
# Falls back to a short UUID (e.g. "run-a3f8c201") if no env var is set.
# Set LITMUS_VERSION=v1.0.0 in .env for a stable label.
# ---------------------------------------------------------------------------

PROMPT_V1 = "You are a helpful assistant."

litmus.init()                       # ← zero-arg; env / UUID fallback
v1_label = litmus.version           # remember the resolved label for comparison

runner_v1 = GoldenTestRunner(litmus)
runner_v1.add_test_case(
    name="capital_city",
    inputs={"question": "What is the capital of France?", "system_prompt": PROMPT_V1},
    run_function=generate_answer,
    expected_pattern=r"Paris",
    expected_behavior="Should state Paris as the capital of France",
)
runner_v1.add_test_case(
    name="math_basic",
    inputs={"question": "What is 7 * 8? Reply with just the number.", "system_prompt": PROMPT_V1},
    run_function=generate_answer,
    expected_pattern=r"56",
    expected_behavior="Should return only the number 56",
)
runner_v1.add_test_case(
    name="reasoning",
    inputs={
        "question": "If all cats are mammals and all mammals are animals, is a cat an animal? Answer yes or no.",
        "system_prompt": PROMPT_V1,
    },
    run_function=generate_answer,
    expected_pattern=r"(?i)\byes\b",
    expected_behavior="Should answer yes using basic syllogistic reasoning",
)
runner_v1.add_test_case(
    name="negation",
    inputs={
        "question": "Is the Earth flat? Answer yes or no.",
        "system_prompt": PROMPT_V1,
    },
    run_function=generate_answer,
    expected_pattern=r"(?i)\bno\b",
    expected_behavior="Should correctly answer no — the Earth is not flat",
)
runner_v1.add_test_case(
    name="sqrt_numeric",
    inputs={
        "question": "What is the square root of 144? Reply with just the number.",
        "system_prompt": PROMPT_V1,
    },
    run_function=generate_answer,
    expected_pattern=r"\b12\b",
    expected_behavior="Should return the number 12",
)
runner_v1.add_test_case(
    name="list_planets",
    inputs={
        "question": "Name exactly 3 planets in our solar system.",
        "system_prompt": PROMPT_V1,
    },
    run_function=generate_answer,
    expected_pattern=r"(?i)(mercury|venus|earth|mars|jupiter|saturn|uranus|neptune)",
    expected_behavior="Should list 3 valid planet names from our solar system",
)

report_v1 = runner_v1.run_tests(num_runs=3)
report_v1.display()


# ---------------------------------------------------------------------------
# Version B: modified prompt — explicit label for the new release.
# In CI, bump LITMUS_VERSION in the workflow env instead of hard-coding.
# ---------------------------------------------------------------------------

PROMPT_V2 = "You are a concise assistant. Always answer in one sentence."

litmus.init(version="v2.0.0")

runner_v2 = GoldenTestRunner(litmus)
runner_v2.add_test_case(
    name="capital_city",
    inputs={"question": "What is the capital of France?", "system_prompt": PROMPT_V2},
    run_function=generate_answer,
    expected_pattern=r"Paris",
    expected_behavior="Should state Paris as the capital of France",
)
runner_v2.add_test_case(
    name="math_basic",
    inputs={"question": "What is 7 * 8? Reply with just the number.", "system_prompt": PROMPT_V2},
    run_function=generate_answer,
    expected_pattern=r"56",
    expected_behavior="Should return only the number 56",
)
runner_v2.add_test_case(
    name="reasoning",
    inputs={
        "question": "If all cats are mammals and all mammals are animals, is a cat an animal? Answer yes or no.",
        "system_prompt": PROMPT_V2,
    },
    run_function=generate_answer,
    expected_pattern=r"(?i)\byes\b",
    expected_behavior="Should answer yes using basic syllogistic reasoning",
)
runner_v2.add_test_case(
    name="negation",
    inputs={
        "question": "Is the Earth flat? Answer yes or no.",
        "system_prompt": PROMPT_V2,
    },
    run_function=generate_answer,
    expected_pattern=r"(?i)\bno\b",
    expected_behavior="Should correctly answer no — the Earth is not flat",
)
runner_v2.add_test_case(
    name="sqrt_numeric",
    inputs={
        "question": "What is the square root of 144? Reply with just the number.",
        "system_prompt": PROMPT_V2,
    },
    run_function=generate_answer,
    expected_pattern=r"\b12\b",
    expected_behavior="Should return the number 12",
)
runner_v2.add_test_case(
    name="list_planets",
    inputs={
        "question": "Name exactly 3 planets in our solar system.",
        "system_prompt": PROMPT_V2,
    },
    run_function=generate_answer,
    expected_pattern=r"(?i)(mercury|venus|earth|mars|jupiter|saturn|uranus|neptune)",
    expected_behavior="Should list 3 valid planet names from our solar system",
)

report_v2 = runner_v2.run_tests(num_runs=3)
report_v2.display()


# ---------------------------------------------------------------------------
# Compare drift (embeddings computed here, not during LLM calls)
# ---------------------------------------------------------------------------

# Compare the two versions (embeddings are computed lazily by DriftDetector)
print("Computing embeddings and comparing versions...")
drift_report = litmus.compare(v1_label, "v2.0.0")

print("\n=== DRIFT REPORT ===")
drift_report.display()

litmus.close()
