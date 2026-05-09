from __future__ import annotations

import os

import litellm


def generate_answer(
    question: str,
    system_prompt: str = "You are a helpful assistant. Be concise, factual, and answer in one sentence when possible.",
) -> str:
    response = litellm.completion(
        model=os.getenv("LITMUS_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content or ""
