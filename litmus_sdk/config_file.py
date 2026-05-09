from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field
import tomlkit


CONFIG_FILE_NAME = "litmus.toml"


class ProjectConfig(BaseModel):
    id: Optional[str] = None
    name: str = ""
    description: str = ""


class AgentConfig(BaseModel):
    entrypoint: Optional[str] = None


class StorageConfig(BaseModel):
    db_path: Optional[str] = None   # None → use global default
    chroma_path: Optional[str] = None


class GoldenTestConfig(BaseModel):
    test_id: Optional[str] = None
    name: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    expected_pattern: str = ""
    expected_behavior: str = ""


class RunRecordConfig(BaseModel):
    version: str
    run_id: Optional[str] = None
    timestamp: Optional[str] = None
    passed: Optional[int] = None
    failed: Optional[int] = None
    total_runs: Optional[int] = None


class LitmusProjectConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    settings: Dict[str, Any] = Field(default_factory=dict)
    golden_tests: list[GoldenTestConfig] = Field(default_factory=list)
    runs: list[RunRecordConfig] = Field(default_factory=list)


def discover_config(start_dir: str | Path | None = None) -> Optional[Path]:
    current = Path(start_dir or os.getcwd()).resolve()
    for directory in [current, *current.parents]:
        candidate = directory / CONFIG_FILE_NAME
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_relative_path(path_value: Optional[str], base_dir: Path) -> Optional[str]:
    if not path_value:
        return path_value
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


_GLOBAL_DEFAULT_DB = str(Path.home() / ".litmus" / "litmus.db")
_GLOBAL_DEFAULT_CHROMA = str(Path.home() / ".litmus" / "chroma")


def load_config(config_path: str | Path) -> LitmusProjectConfig:
    path = Path(config_path).resolve()
    text = path.read_text(encoding="utf-8")
    parsed = tomlkit.parse(text)
    config = LitmusProjectConfig.model_validate(parsed.unwrap())

    # If db_path is set in the TOML, resolve it relative to the config file.
    # If absent (None), leave it as None — core.py will use the global default.
    config.storage.db_path = _resolve_relative_path(config.storage.db_path, path.parent)
    config.storage.chroma_path = _resolve_relative_path(config.storage.chroma_path, path.parent)
    return config


def save_config(config: LitmusProjectConfig, config_path: str | Path) -> None:
    path = Path(config_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = tomlkit.document()

    project = tomlkit.table()
    if config.project.id:
        project.add("id", config.project.id)
    project.add("name", config.project.name)
    project.add("description", config.project.description)
    doc.add("project", project)

    agent = tomlkit.table()
    if config.agent.entrypoint:
        agent.add("entrypoint", config.agent.entrypoint)
    doc.add("agent", agent)

    storage = tomlkit.table()
    if config.storage.db_path:
        storage.add("db_path", config.storage.db_path)
    if config.storage.chroma_path:
        storage.add("chroma_path", config.storage.chroma_path)
    doc.add("storage", storage)

    settings = tomlkit.table()
    for key, value in config.settings.items():
        settings.add(key, value)
    doc.add("settings", settings)

    tests = tomlkit.aot()
    for test in config.golden_tests:
        item = tomlkit.table()
        if test.test_id:
            item.add("test_id", test.test_id)
        item.add("name", test.name)
        item.add("expected_pattern", test.expected_pattern)
        item.add("expected_behavior", test.expected_behavior)

        inputs_table = tomlkit.table()
        for key, value in test.inputs.items():
            inputs_table.add(key, value)
        item.add("inputs", inputs_table)
        tests.append(item)
    doc.add("golden_tests", tests)

    runs = tomlkit.aot()
    for run in config.runs:
        item = tomlkit.table()
        item.add("version", run.version)
        if run.run_id:
            item.add("run_id", run.run_id)
        if run.timestamp:
            item.add("timestamp", run.timestamp)
        if run.passed is not None:
            item.add("passed", run.passed)
        if run.failed is not None:
            item.add("failed", run.failed)
        if run.total_runs is not None:
            item.add("total_runs", run.total_runs)
        runs.append(item)
    doc.add("runs", runs)

    path.write_text(tomlkit.dumps(doc), encoding="utf-8")


def merge_server_config(local: LitmusProjectConfig, remote: LitmusProjectConfig) -> LitmusProjectConfig:
    merged = local.model_copy(deep=True)
    merged.project = remote.project
    if remote.golden_tests:
        merged.golden_tests = remote.golden_tests
    if remote.runs:
        merged.runs = remote.runs
    return merged


def resolve_entrypoint(entrypoint: str, *, base_dir: str | Path | None = None) -> Callable[..., Any]:
    if ":" not in entrypoint:
        raise ValueError("Agent entrypoint must be in 'module:function' format.")

    module_name, function_name = entrypoint.split(":", 1)
    module_name = module_name.strip()
    function_name = function_name.strip()
    if not module_name or not function_name:
        raise ValueError("Agent entrypoint must include both module and function names.")

    base_path = Path(base_dir).resolve() if base_dir else None
    should_pop = False
    if base_path and str(base_path) not in sys.path:
        sys.path.insert(0, str(base_path))
        should_pop = True

    try:
        module = importlib.import_module(module_name)
        run_function = getattr(module, function_name)
    finally:
        if should_pop and sys.path and sys.path[0] == str(base_path):
            sys.path.pop(0)

    if not callable(run_function):
        raise TypeError(f"Resolved entrypoint '{entrypoint}' is not callable.")
    return run_function
