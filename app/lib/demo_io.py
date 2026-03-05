from __future__ import annotations

import json
import os
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_demo_dir() -> Path:
    raw = os.environ.get("DEMO_DATA_DIR", "")
    if raw.strip():
        return Path(raw).expanduser()
    return PROJECT_ROOT / "app" / "demo_data"


def load_manifest(demo_dir: Path) -> dict:
    manifest = demo_dir / "run_manifest.json"
    if not manifest.exists():
        return {"status": "missing", "message": f"Missing demo manifest: {manifest}"}
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as e:
        return {"status": "error", "message": f"Failed to read manifest: {manifest} -> {e!r}"}


def path_exists(p: str | None) -> bool:
    if not p:
        return False
    try:
        return Path(p).exists()
    except Exception:
        return False


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def pretty_path(p: str | Path | None, *, root: Path | None = None) -> str:
    """
    UI-friendly path: prefer repo-relative; never show full absolute paths.
    """
    if not p:
        return ""
    root = (root or PROJECT_ROOT).resolve()
    try:
        pp = Path(p).expanduser()
        pr = pp.resolve()
        if pr == root or root in pr.parents:
            return str(pr.relative_to(root))
        # Outside repo: hide absolute location.
        return f"external/{pp.name}"
    except Exception:
        # If parsing fails, avoid leaking absolute paths.
        s = str(p)
        return f"external/{Path(s).name}"


def sanitize_text(s: str, *, root: Path | None = None) -> str:
    """
    Best-effort sanitizer for stdout/log text shown in the UI.

    - Rewrites repo-absolute paths to repo-relative paths.
    - Avoids printing raw `/mnt/...` absolute paths.
    """
    if not s:
        return ""
    root = (root or PROJECT_ROOT).resolve()
    prefixes = {str(root), str(root.resolve())}

    out_lines: list[str] = []
    for line in s.splitlines():
        # Mask common non-repo absolute paths (privacy + portability).
        def _mask_abs(m: re.Match[str]) -> str:
            try:
                return f"external/{Path(m.group(0)).name}"
            except Exception:
                return "external/<path>"

        line = re.sub(r"/home/[^\s'\"<>]+", _mask_abs, line)
        line = re.sub(r"/usr/[^\s'\"<>]+", _mask_abs, line)
        line = re.sub(r"/opt/[^\s'\"<>]+", _mask_abs, line)

        # Strip repo prefix wherever it appears.
        for pre in prefixes:
            if pre:
                line = line.replace(pre, "")
        # Preserve leading whitespace (helps when rendering markdown/code).
        line = line.rstrip()
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        # If we removed the repo prefix, a path may now start with "/...".
        if stripped.startswith("/"):
            stripped = stripped[1:]
        out_lines.append(indent + stripped)
    return "\n".join(out_lines).strip() + ("\n" if s.endswith("\n") else "")
