# litmus-sdk

LLM regression testing via embedding drift.

## Install

```bash
pip install litmus-sdk
```

## Quick start

```python
from litmus_sdk import LitmusSDK

sdk = LitmusSDK(db_path="./litmus.db", project_id="my-project")
sdk.init("v1.0.0")

# Your litellm calls are now instrumented automatically
import litellm
response = litellm.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hello"}])

# Compare two versions
report = sdk.compare("v1.0.0", "v2.0.0")
report.display()
```

## CLI

```bash
litmus --help
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```
