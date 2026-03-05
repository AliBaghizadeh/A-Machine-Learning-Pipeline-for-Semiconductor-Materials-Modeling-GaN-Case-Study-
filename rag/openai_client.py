from __future__ import annotations

from typing import Optional


def generate(
    prompt: str,
    *,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_s: float = 60.0,
) -> str:
    """
    Minimal OpenAI client wrapper for demo UI.

    Notes:
    - This is an optional cloud mode. Only used if the user selects it in-app.
    - Requires `pip install openai` and a valid API key.
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: install with `pip install openai`") from e

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
    resp = client.responses.create(model=model, input=prompt)
    # openai-python exposes a convenience string for text output
    text = getattr(resp, "output_text", None)
    if not isinstance(text, str):
        raise RuntimeError("OpenAI response missing output_text")
    return text.strip()

