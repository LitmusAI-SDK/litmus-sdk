# SQL Agent Example

This directory is the Litmus config for the SQL agent system.

## Files
- `litmus.toml` project configuration and golden tests

## Notes
- The runnable agent currently lives in `sql-agent/sql_agent.py` (`sql_agent:generate_sql`).
- The example runner script is `examples/sql_agent_example.py`.

## Quick Start
1. `python sql-agent/create_db.py`
2. Ensure `OPENAI_API_KEY` is set
3. From repo root, run `python examples/sql_agent_example.py`
