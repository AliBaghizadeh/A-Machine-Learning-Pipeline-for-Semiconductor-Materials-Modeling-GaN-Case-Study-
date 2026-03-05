from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests


def _host(host: Optional[str] = None) -> str:
    if host:
        return host.rstrip("/")
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def is_running(timeout_s: float = 0.5, host: Optional[str] = None) -> bool:
    try:
        r = requests.get(_host(host) + "/api/tags", timeout=timeout_s)
        return r.status_code == 200
    except Exception:
        return False


def embed(text: str, model: str = "nomic-embed-text", timeout_s: float = 30.0, host: Optional[str] = None) -> List[float]:
    payload = {"model": model, "prompt": text}
    r = requests.post(_host(host) + "/api/embeddings", json=payload, timeout=timeout_s)
    r.raise_for_status()
    data: Dict[str, Any] = r.json()
    emb = data.get("embedding")
    if not isinstance(emb, list):
        raise RuntimeError("Ollama embeddings response missing 'embedding' list")
    return [float(x) for x in emb]


def generate(prompt: str, model: str = "llama3.1:8b", timeout_s: float = 60.0, host: Optional[str] = None) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(_host(host) + "/api/generate", json=payload, timeout=timeout_s)
    r.raise_for_status()
    data: Dict[str, Any] = r.json()
    text = data.get("response")
    if not isinstance(text, str):
        raise RuntimeError("Ollama generate response missing 'response' string")
    return text.strip()
