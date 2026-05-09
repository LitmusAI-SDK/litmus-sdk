"""
SQL Agent — powered by CrewAI + LiteLLM.

This agent converts natural-language questions into SQL queries for a
two-table SQLite schema (users / orders).  It serves as the test harness
for the Litmus SDK — a bounded domain with deterministic expected outputs.

Public API:
    generate_sql(query: str, system_prompt: str | None) -> dict
        Returns a dict with "output" (SQL query) or "error" key.

The DATABASE_SCHEMA constant is the system prompt "static" part — changing
it between v1 and v2 is the canonical demo of Litmus detecting prompt drift.
"""
from __future__ import annotations

import os
import sys

import litellm
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# Ensure repo root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Initialize FastAPI app
app = FastAPI(title="SQL Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunTestRequest(BaseModel):
    """Request body for /run endpoint."""
    inputs: dict  # {"question": "...", "system_prompt": "..." (optional)}
    version: str  # version label for tracking

# ---------------------------------------------------------------------------
# Static prompt (version-controlled constant)
# v1.0.0 — baseline schema description
# ---------------------------------------------------------------------------

DATABASE_SCHEMA = """
You are an expert SQL developer working with a SQLite database.

Schema:

Table: users
  - id         INTEGER PRIMARY KEY
  - name       TEXT    NOT NULL
  - email      TEXT    UNIQUE NOT NULL
  - created_at DATE    NOT NULL

Table: orders
  - id         INTEGER PRIMARY KEY
  - user_id    INTEGER NOT NULL  (foreign key → users.id)
  - product    TEXT    NOT NULL
  - amount     DECIMAL(10,2) NOT NULL
  - order_date DATE    NOT NULL

Rules:
  1. Return ONLY the SQL query — no explanation, no markdown fences, no semicolons.
  2. Use standard SQLite syntax.
  3. Column and table names are case-sensitive — match the schema exactly.
  4. For JOINs, always use the explicit JOIN ... ON syntax.
  5. For aggregations, always include a GROUP BY clause.
  6. Never generate DROP, DELETE, UPDATE, or INSERT statements.
  7. If the request is ambiguous, return the most general valid SELECT.
"""


def generate_sql(query: str, system_prompt: str | None = None) -> dict:
    """
    Convert a natural language *query* into a SQL string using litellm directly.

    Args:
        query: Plain-English question, e.g. "Show all users".
        system_prompt: Optional custom system prompt. Defaults to DATABASE_SCHEMA.

    Returns:
        Dict with "output" key containing SQL query string, or "error" key on failure.
    """
    try:
        system_content = system_prompt if system_prompt else DATABASE_SCHEMA
        
        response = litellm.completion(
            model=os.getenv("LITMUS_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("LITMUS_TEMPERATURE", "0.0")),
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user",   "content": f"Generate a SQL query for: {query}"},
            ],
        )

        sql = (response.choices[0].message.content or "").strip()

        # Sanitise: remove markdown fences if the model added them
        if sql.startswith("```"):
            lines = sql.split("\n")
            sql = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        # Strip trailing semicolons
        sql = sql.rstrip(";").strip()

        return {"output": sql}
    
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "sql-agent"}


@app.post("/run")
async def run_test(body: RunTestRequest):
    """
    Run the SQL agent with the given inputs.
    
    Args:
        body: Request containing inputs (question, optional system_prompt) and version
        
    Returns:
        {"output": "...", "error": "..."} depending on result
    """
    inputs = body.inputs
    question = inputs.get("question")
    system_prompt = inputs.get("system_prompt")
    version = body.version
    
    if not question:
        return {"output": None, "error": "Missing 'question' in inputs"}
    
    # Run the agent
    result = generate_sql(question, system_prompt)
    return result


if __name__ == "__main__":
    import sys
    import uvicorn

    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        # Start server on port 8080
        port = int(os.getenv("PORT", 8080))
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
    else:
        # CLI mode
        q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Show all users"
        print(f"Query  : {q}")
        result = generate_sql(q)
        if "error" in result:
            print(f"Error  : {result['error']}")
        else:
            print(f"SQL    : {result['output']}")
