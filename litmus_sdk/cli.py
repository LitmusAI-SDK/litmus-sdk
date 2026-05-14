"""
Litmus CLI — config-driven golden test runner.

Commands
--------
  litmus init     Create a new project and generate litmus.toml
  litmus push     Sync local tests/config → API server
  litmus pull     Sync API server tests/config → local litmus.toml
  litmus run      Run golden tests locally, report to API
  litmus watch    Start a runner daemon for UI-triggered test runs
  litmus status   Show project info, test count, and runner status

Environment
-----------
  LITMUS_API_URL    Base URL of the Litmus API server (default: http://localhost:8000)
  OPENAI_API_KEY    Required for LLM calls
  LITMUS_VERSION    Default version label for runs
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import requests

from litmus_sdk.config_file import (
    AgentConfig,
    GoldenTestConfig,
    LitmusProjectConfig,
    ProjectConfig,
    RunRecordConfig,
    StorageConfig,
    discover_config,
    load_config,
    save_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_API = "http://localhost:8000"
API_ENV = "LITMUS_API_URL"


def _api(ctx_obj: dict) -> str:
    return ctx_obj.get("api_url", DEFAULT_API).rstrip("/")


def _require_config(search_from: str = ".") -> tuple[Path, LitmusProjectConfig]:
    path = discover_config(search_from)
    if not path:
        raise click.ClickException(
            "No litmus.toml found in this directory or any parent.\n"
            "Run `litmus init` to create one."
        )
    return path, load_config(path)


def _get(api_base: str, path: str, **kwargs) -> requests.Response:
    return requests.get(f"{api_base}{path}", timeout=10, **kwargs)


def _post(api_base: str, path: str, **kwargs) -> requests.Response:
    return requests.post(f"{api_base}{path}", timeout=10, **kwargs)


def _put(api_base: str, path: str, **kwargs) -> requests.Response:
    return requests.put(f"{api_base}{path}", timeout=10, **kwargs)


def _delete(api_base: str, path: str, **kwargs) -> requests.Response:
    return requests.delete(f"{api_base}{path}", timeout=10, **kwargs)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.option(
    "--api",
    default=None,
    envvar=API_ENV,
    help=f"Litmus API server URL (env: {API_ENV}). Default: {DEFAULT_API}",
)
@click.pass_context
def cli(ctx: click.Context, api: Optional[str]) -> None:
    """Litmus — config-driven golden test runner for LLM agents."""
    ctx.ensure_object(dict)
    ctx.obj["api_url"] = api or DEFAULT_API


# ---------------------------------------------------------------------------
# litmus init
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--name", default=None, help="Project name")
@click.option("--description", default=None, help="Project description")
@click.option("--entrypoint", default=None, help="Agent entrypoint in module:function format")
@click.pass_context
def init(ctx: click.Context, name: Optional[str], description: Optional[str], entrypoint: Optional[str]) -> None:
    """Initialize a new Litmus project and create litmus.toml."""
    config_path = Path.cwd() / "litmus.toml"

    if config_path.exists():
        click.confirm(
            f"litmus.toml already exists in {Path.cwd()}. Overwrite?",
            abort=True,
        )

    if name is None:
        name = click.prompt("Project name", default=Path.cwd().name)
    if description is None:
        description = click.prompt("Description (optional)", default="")
    if entrypoint is None:
        entrypoint = click.prompt("Agent entrypoint (module:function, optional)", default="")

    api_base = _api(ctx.obj)
    project_id: Optional[str] = None

    try:
        resp = _post(
            api_base,
            "/api/projects",
            json={"name": name, "description": description},
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            project_id = data.get("project_id")
            click.echo(f"  Created project on server: {project_id}")
        else:
            click.echo(f"  [warn] Could not reach API server ({resp.status_code}). Config will use local ID.")
    except requests.exceptions.ConnectionError:
        click.echo(f"  [warn] API server not reachable at {api_base}. Config will use local ID.")

    project_id = project_id or str(uuid.uuid4())

    # Ask the backend which host-side DB / Chroma paths it's reading from,
    # so this project writes to the same storage. Falls back to SDK defaults
    # when the backend can't be reached or wasn't launched via setup.sh.
    storage_cfg = StorageConfig()
    try:
        paths_resp = _get(api_base, "/api/storage/paths")
        if paths_resp.status_code == 200:
            paths = paths_resp.json()
            db = paths.get("db_path")
            chroma = paths.get("chroma_path")
            if db:
                storage_cfg.db_path = db
            if chroma:
                storage_cfg.chroma_path = chroma
            if db or chroma:
                click.echo(f"  Storage paths discovered from server:")
                if db:
                    click.echo(f"    db     : {db}")
                if chroma:
                    click.echo(f"    chroma : {chroma}")
            else:
                click.echo("  [warn] Server did not advertise storage paths. SDK will use local defaults — runs may not appear on the dashboard.")
    except requests.exceptions.ConnectionError:
        pass

    config = LitmusProjectConfig(
        project=ProjectConfig(id=project_id, name=name, description=description),
        agent=AgentConfig(entrypoint=entrypoint.strip() or None),
        storage=storage_cfg,
        settings={},
        golden_tests=[],
        runs=[],
    )
    save_config(config, config_path)

    from pathlib import Path as _Path
    _default_db = str(_Path.home() / ".litmus" / "litmus.db")

    click.echo(f"\nCreated litmus.toml")
    click.echo(f"  project.id  : {project_id}")
    click.echo(f"  storage.db  : {_default_db}  (shared global default)")
    if entrypoint:
        click.echo(f"  agent       : {entrypoint}")
    click.echo("\nNext steps:")
    click.echo("  1. Add [[golden_tests]] entries to litmus.toml")
    click.echo("  2. litmus push          # sync tests to server")
    click.echo("  3. litmus run           # run tests locally")
    click.echo("  4. litmus watch         # start runner for UI-triggered runs")


# ---------------------------------------------------------------------------
# litmus push
# ---------------------------------------------------------------------------

@cli.command()
@click.pass_context
def push(ctx: click.Context) -> None:
    """Push local litmus.toml (project metadata + tests) to API server."""
    config_path, config = _require_config()
    api_base = _api(ctx.obj)
    project_id = config.project.id

    if not project_id:
        raise click.ClickException("litmus.toml is missing project.id. Run `litmus init`.")

    click.echo(f"Pushing config for project {project_id} …")

    try:
        resp = _put(
            api_base,
            f"/api/projects/{project_id}/config",
            json={
                "project": {"name": config.project.name, "description": config.project.description},
                "golden_tests": [
                    {
                        "test_id": t.test_id,
                        "name": t.name,
                        "inputs": t.inputs,
                        "expected_pattern": t.expected_pattern,
                        "expected_behavior": t.expected_behavior,
                    }
                    for t in config.golden_tests
                ],
            },
        )
    except requests.exceptions.ConnectionError:
        raise click.ClickException(f"Cannot reach API server at {api_base}. Is it running?")

    if resp.status_code in (200, 201):
        data = resp.json()
        click.echo(f"  Tests synced: {data.get('tests_pushed', '?')} test case(s) pushed")

        # Sync server-assigned test_ids back into litmus.toml so the runner
        # uses the same IDs and doesn't create duplicate test case rows.
        try:
            pull_resp = _get(api_base, f"/api/projects/{project_id}/config")
            if pull_resp.status_code == 200:
                remote_tests = {t["name"]: t["test_id"] for t in pull_resp.json().get("golden_tests", [])}
                updated = False
                for test in config.golden_tests:
                    server_id = remote_tests.get(test.name)
                    if server_id and test.test_id != server_id:
                        test.test_id = server_id
                        updated = True
                if updated:
                    save_config(config, config_path)
                    click.echo("  Test IDs synced back to litmus.toml")
        except Exception:
            pass  # non-fatal

        click.echo("  Done.")
    else:
        raise click.ClickException(f"Server returned {resp.status_code}: {resp.text}")


# ---------------------------------------------------------------------------
# litmus pull
# ---------------------------------------------------------------------------

@cli.command()
@click.pass_context
def pull(ctx: click.Context) -> None:
    """Pull project config and test cases from API server into litmus.toml."""
    config_path, local_config = _require_config()
    api_base = _api(ctx.obj)
    project_id = local_config.project.id

    if not project_id:
        raise click.ClickException("litmus.toml is missing project.id. Run `litmus init`.")

    click.echo(f"Pulling config for project {project_id} …")

    try:
        resp = _get(api_base, f"/api/projects/{project_id}/config")
    except requests.exceptions.ConnectionError:
        raise click.ClickException(f"Cannot reach API server at {api_base}. Is it running?")

    if resp.status_code == 404:
        raise click.ClickException(f"Project {project_id} not found on server. Run `litmus push` first.")
    if resp.status_code != 200:
        raise click.ClickException(f"Server returned {resp.status_code}: {resp.text}")

    remote = resp.json()

    # Update project metadata
    local_config.project.name = remote["project"]["name"]
    local_config.project.description = remote["project"].get("description", "")

    # Merge remote test cases (preserve local [agent] and [storage])
    remote_tests = remote.get("golden_tests", [])
    local_config.golden_tests = [
        GoldenTestConfig(
            test_id=t.get("test_id"),
            name=t["name"],
            inputs=t.get("inputs", {}),
            expected_pattern=t.get("expected_pattern", ""),
            expected_behavior=t.get("expected_behavior", ""),
        )
        for t in remote_tests
    ]

    # Append new run records
    existing_run_ids = {r.run_id for r in local_config.runs if r.run_id}
    for run in remote.get("recent_runs", []):
        if run.get("run_id") not in existing_run_ids:
            local_config.runs.append(
                RunRecordConfig(
                    version=run["version"],
                    run_id=run.get("run_id"),
                    timestamp=run.get("timestamp"),
                    passed=run.get("passed"),
                    failed=run.get("failed"),
                    total_runs=run.get("total_runs"),
                )
            )

    save_config(local_config, config_path)

    click.echo(f"  Test cases : {len(local_config.golden_tests)} synced")
    click.echo(f"  Saved to   : {config_path}")


# ---------------------------------------------------------------------------
# litmus run
# ---------------------------------------------------------------------------

@cli.command("run")
@click.option("--version", default=None, help="Version label (default: env or auto-UUID)")
@click.option("--num-runs", default=3, show_default=True, help="Number of runs per test case")
@click.pass_context
def run_tests(ctx: click.Context, version: Optional[str], num_runs: int) -> None:
    """Run golden tests locally and report results to API server."""
    config_path, config = _require_config()
    api_base = _api(ctx.obj)

    if not config.agent.entrypoint:
        raise click.ClickException(
            "No [agent].entrypoint defined in litmus.toml.\n"
            "Add: entrypoint = \"module:function\""
        )

    if not config.golden_tests:
        raise click.ClickException(
            "No [[golden_tests]] defined in litmus.toml.\n"
            "Add test cases or run `litmus pull` to fetch from server."
        )

    click.echo(f"Loading agent from: {config.agent.entrypoint}")
    click.echo(f"Tests: {len(config.golden_tests)} | Runs per test: {num_runs}")

    import litmus_sdk
    sdk_instance = litmus_sdk.LitmusSDK(config_path=str(config_path))
    sdk_instance.init(version=version)
    click.echo(f"Version: {sdk_instance.version}")
    click.echo("")

    runner = litmus_sdk.GoldenTestRunner(sdk_instance)
    try:
        loaded = runner.load_from_config(config_path=str(config_path))
        click.echo(f"Loaded {loaded} test case(s) from config")
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc))

    click.echo("Running …\n")
    report = runner.run_tests(num_runs=num_runs)
    report.display()

    # Record run in litmus.toml
    passed = sum(1 for r in report.results if r.passed)
    failed = len(report.results) - passed
    config.runs.append(
        RunRecordConfig(
            version=sdk_instance.version,
            run_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat() + "Z",
            passed=passed,
            failed=failed,
            total_runs=len(report.results),
        )
    )
    save_config(config, config_path)
    click.echo(f"\nRun recorded in litmus.toml (passed={passed}, failed={failed})")

    # Push results to API
    try:
        _post(
            api_base,
            f"/api/projects/{config.project.id}/runs/queue",
            json={
                "version": sdk_instance.version,
                "num_runs": num_runs,
                "source": "cli",
                "passed": passed,
                "failed": failed,
                "total_tests": report.total_tests,
                "total_runs": len(report.results),
            },
        )
    except requests.exceptions.ConnectionError:
        click.echo("[warn] Could not report to API server (not running?). Results saved locally.")

    sdk_instance.close()


# ---------------------------------------------------------------------------
# litmus watch
# ---------------------------------------------------------------------------

@cli.command()
@click.pass_context
def watch(ctx: click.Context) -> None:
    """
    Start a runner daemon.  Long-polls the API server for pending runs
    queued from the UI and executes them locally.

    Press Ctrl-C to stop.
    """
    config_path, config = _require_config()
    api_base = _api(ctx.obj)
    project_id = config.project.id

    if not project_id:
        raise click.ClickException("litmus.toml is missing project.id.")
    if not config.agent.entrypoint:
        raise click.ClickException(
            "No [agent].entrypoint in litmus.toml. Cannot execute tests."
        )

    runner_id = str(uuid.uuid4())
    click.echo(f"Starting Litmus runner daemon")
    click.echo(f"  project   : {project_id}")
    click.echo(f"  runner_id : {runner_id}")
    click.echo(f"  api       : {api_base}")
    click.echo(f"  entrypoint: {config.agent.entrypoint}")
    click.echo("\nPolling for runs… (Ctrl-C to stop)\n")

    # Register runner
    try:
        resp = _post(
            api_base,
            f"/api/projects/{project_id}/runners",
            json={"runner_id": runner_id},
        )
        if resp.status_code not in (200, 201):
            click.echo(f"[warn] Could not register runner: {resp.status_code}")
    except requests.exceptions.ConnectionError:
        raise click.ClickException(f"Cannot reach API server at {api_base}.")

    stop_event = {"stop": False}

    def _shutdown(sig=None, frame=None):
        stop_event["stop"] = True
        click.echo("\nShutting down runner …")
        try:
            _delete(api_base, f"/api/projects/{project_id}/runners/{runner_id}")
        except Exception:
            pass

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    last_heartbeat = 0.0
    HEARTBEAT_INTERVAL = 30
    POLL_INTERVAL = 2

    import litmus_sdk

    # Create a single SDK instance for the lifetime of the runner.
    # Calling sdk_instance.init(version=...) on each run re-initialises the version label and re-registers callbacks without closing the DB connection, which prevents "Cannot operate on a closed database" errors that occur when a new connection is opened immediately after closing the previous one. (ChromaDB WAL locks or Python 3.12 executescript behaviour can leave the new connection in a broken state on the second and subsequent runs).
    sdk_instance = litmus_sdk.LitmusSDK(config_path=str(config_path))

    while not stop_event["stop"]:
        now = time.time()

        # Heartbeat
        if now - last_heartbeat > HEARTBEAT_INTERVAL:
            try:
                _post(api_base, f"/api/projects/{project_id}/runners/{runner_id}/heartbeat")
                last_heartbeat = now
            except Exception:
                pass

        # Poll for pending run
        try:
            resp = _get(api_base, f"/api/projects/{project_id}/runs/pending")
        except requests.exceptions.ConnectionError:
            click.echo("[warn] Lost API connection. Retrying …")
            time.sleep(5)
            continue

        if resp.status_code != 200:
            time.sleep(POLL_INTERVAL)
            continue

        queued_run = resp.json()
        if not queued_run:
            time.sleep(POLL_INTERVAL)
            continue

        queue_id = queued_run.get("queue_id")
        version = queued_run.get("version")
        num_runs = queued_run.get("num_runs", 3)

        click.echo(f"[{datetime.utcnow().isoformat()}] Picked up run: queue_id={queue_id} version={version} num_runs={num_runs}")

        # Claim the run
        try:
            claim_resp = _post(
                api_base,
                f"/api/projects/{project_id}/runs/{queue_id}/claim",
                json={"runner_id": runner_id},
            )
            if claim_resp.status_code != 200:
                click.echo(f"  [skip] Already claimed by another runner.")
                time.sleep(POLL_INTERVAL)
                continue
        except Exception as exc:
            click.echo(f"  [warn] Failed to claim run: {exc}")
            time.sleep(POLL_INTERVAL)
            continue

        # Execute run
        try:
            sdk_instance.init(version=version)
            runner_obj = litmus_sdk.GoldenTestRunner(sdk_instance)

            # Load test cases from the server so UI-added tests are included.
            # Fall back to local config if the server is unreachable.
            server_config_loaded = False
            try:
                srv = _get(api_base, f"/api/projects/{project_id}/config")
                if srv.status_code == 200:
                    remote_tests = srv.json().get("golden_tests", [])
                    if remote_tests:
                        merged = config.model_copy(deep=True)
                        merged.golden_tests = [
                            GoldenTestConfig(
                                test_id=t.get("test_id"),
                                name=t["name"],
                                inputs=t.get("inputs", {}),
                                expected_pattern=t.get("expected_pattern", ""),
                                expected_behavior=t.get("expected_behavior", ""),
                            )
                            for t in remote_tests
                        ]
                        runner_obj.load_from_config(config=merged)
                        server_config_loaded = True
            except Exception:
                pass

            if not server_config_loaded:
                runner_obj.load_from_config(config_path=str(config_path))
            report = runner_obj.run_tests(num_runs=num_runs)

            passed = sum(1 for r in report.results if r.passed)
            failed = len(report.results) - passed
            result_summary = {
                "total_tests": report.total_tests,
                "total_runs": len(report.results),
                "passed": passed,
                "failed": failed,
                "version": version,
            }
            click.echo(f"  Completed: passed={passed} failed={failed}")
        except Exception as exc:
            result_summary = {"error": str(exc)}
            click.echo(f"  [error] Run failed: {exc}")

        # Report completion
        try:
            _post(
                api_base,
                f"/api/projects/{project_id}/runs/{queue_id}/complete",
                json={"runner_id": runner_id, "result_summary": result_summary},
            )
        except Exception as exc:
            click.echo(f"  [warn] Failed to report completion: {exc}")

        time.sleep(POLL_INTERVAL)

    sdk_instance.close()
    click.echo("Runner stopped.")


# ---------------------------------------------------------------------------
# litmus status
# ---------------------------------------------------------------------------

@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show project info, test count, and active runners."""
    api_base = _api(ctx.obj)
    path = discover_config()

    if path:
        config = load_config(path)
        click.echo(f"Config       : {path}")
        click.echo(f"Project      : {config.project.name} ({config.project.id})")
        click.echo(f"Tests        : {len(config.golden_tests)}")
        click.echo(f"Local runs   : {len(config.runs)}")
        if config.runs:
            last = config.runs[-1]
            click.echo(f"Last run     : {last.version} — {last.passed}✓ {last.failed}✗  @ {last.timestamp}")
        click.echo(f"Storage      : {config.storage.db_path}")
        if config.agent.entrypoint:
            click.echo(f"Entrypoint   : {config.agent.entrypoint}")
        else:
            click.echo("Entrypoint   : (not set)")
    else:
        click.echo("No litmus.toml found. Run `litmus init` to create one.")
        return

    # API status
    project_id = config.project.id
    if not project_id:
        return

    try:
        runners_resp = _get(api_base, f"/api/projects/{project_id}/runners")
        if runners_resp.status_code == 200:
            runners = runners_resp.json()
            online = [r for r in runners if r.get("status") == "online"]
            click.echo(f"\nRunners      : {len(online)} online / {len(runners)} total")
            for r in runners:
                icon = "●" if r["status"] == "online" else "○"
                click.echo(f"  {icon} {r['runner_id'][:16]}…  last heartbeat: {r.get('last_heartbeat', '?')}")
        else:
            click.echo(f"\n[API]        : {api_base} (project endpoint returned {runners_resp.status_code})")
    except requests.exceptions.ConnectionError:
        click.echo(f"\nAPI server   : not reachable at {api_base}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
