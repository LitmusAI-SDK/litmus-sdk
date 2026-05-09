"""
litmus_sdk package — LLM regression testing via embedding drift.

Public API surface:
    from litmus_sdk import LitmusSDK
    from litmus_sdk.testing import GoldenTestRunner
    from litmus_sdk.drift import DriftDetector, DriftConfig
    from litmus_sdk.callback import LitmusCallback
    from litmus_sdk.storage import LitmusStorage
    from litmus_sdk.models import LLMTrace, PromptComponents, TestCase, TestResult, TestReport, DriftReport
"""

# Ensure Unicode output (✓ ✗ ✅ 🚨 etc.) renders correctly on Windows terminals.
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from litmus_sdk.core import LitmusSDK
from litmus_sdk.testing import GoldenTestRunner
from litmus_sdk.drift import DriftDetector, DriftConfig
from litmus_sdk.callback import LitmusCallback
from litmus_sdk.config_file import LitmusProjectConfig, discover_config, load_config, save_config
from litmus_sdk.storage import LitmusStorage
from litmus_sdk.testing.models import LLMTrace, PromptComponents, TestCase, TestResult, TestReport, DriftReport

__all__ = [
    "LitmusSDK",
    "GoldenTestRunner",
    "DriftDetector",
    "DriftConfig",
    "LitmusCallback",
    "LitmusProjectConfig",
    "discover_config",
    "load_config",
    "save_config",
    "LitmusStorage",
    "LLMTrace",
    "PromptComponents",
    "TestCase",
    "TestResult",
    "TestReport",
    "DriftReport",
]
