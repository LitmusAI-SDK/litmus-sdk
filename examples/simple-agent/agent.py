from __future__ import annotations

import os

import litellm


MODEL = os.getenv("LITMUS_MODEL", "gpt-4o-mini")


def my_agent(question: str, system_prompt: str = "You are a concise, helpful assistant.") -> str:
    response = litellm.completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=float(os.getenv("LITMUS_TEMPERATURE", "0.0")),
    )
    return (response.choices[0].message.content or "").strip()
