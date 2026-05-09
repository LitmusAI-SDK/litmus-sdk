"""
Callback utilities — helper functions for trace extraction and processing.
"""

from __future__ import annotations

from typing import Any


def extract_system_user(messages: list) -> tuple[str, str]:
    """Pull the last system and last user message content from a messages list."""
    system_prompt = ""
    user_input = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role == "system":
            system_prompt = content
        elif role == "user":
            user_input = content
    return system_prompt, user_input


def extract_response_text(response_obj: Any) -> str:
    """Safely extract the assistant's text from a litellm ModelResponse."""
    try:
        if hasattr(response_obj, "choices") and response_obj.choices:
            msg = response_obj.choices[0].message
            return (getattr(msg, "content", None) or "").strip()
    except Exception:
        pass
    return ""
