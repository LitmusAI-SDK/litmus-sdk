"""
HTTP client for the Litmus backend.

The SDK does not persist data itself — every method on LitmusClient is a
thin wrapper around a backend HTTP endpoint. Storage of the SQLite/Chroma
data lives entirely in the litmus-backend service.
"""

from .litmus_client import LitmusClient

__all__ = ["LitmusClient"]
