"""
Offline paper ingestion entrypoint.

This is intentionally lightweight for demo purposes:
- Accepts a directory of PDF/txt files
- Extracts text (best-effort) and writes a JSONL of chunks

If the sources directory is empty, we fall back to the bundled demo snippet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None  # type: ignore


def _iter_source_files(sources: Path):
    if not sources.exists():
        return []
    files = []
    for ext in (".txt", ".pdf"):
        files.extend(sorted(sources.rglob(f"*{ext}")))
    return files


def _chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 200):
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_chars)
        chunks.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap)
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    sources = Path(args.sources)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = _iter_source_files(sources)
    if not files:
        # Demo fallback: use bundled snippet so the pipeline always runs offline.
        demo = Path("app/demo_data/rag/demo_paper_snippet.txt")
        text = demo.read_text(encoding="utf-8") if demo.exists() else "Demo snippet missing."
        payload = {"source": str(demo), "text": text}
        (out / "papers_parsed.json").write_text(json.dumps([payload], indent=2), encoding="utf-8")
        with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
            for idx, c in enumerate(_chunk_text(text)):
                f.write(json.dumps({"source": str(demo), "chunk_id": idx, "chunk": c}) + "\n")
        print("RAG_INGEST_DONE fallback=demo_snippet out=", out)
        return 0

    parsed = []
    chunks_written = 0
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for p in files:
            if p.suffix.lower() == ".txt":
                text = p.read_text(encoding="utf-8", errors="replace")
                note = ""
            else:
                if PdfReader is None:
                    text = ""
                    note = "pdf_text_empty (install pypdf for PDF extraction: pip install pypdf)"
                else:
                    try:
                        reader = PdfReader(str(p))
                        pages = []
                        for pg in reader.pages:
                            pages.append(pg.extract_text() or "")
                        text = "\n".join(pages)
                        note = ""
                    except Exception as e:
                        text = ""
                        note = f"pdf_text_empty (read failed: {e})"
            parsed.append({"source": str(p), "text": text, "note": note})
            for idx, c in enumerate(_chunk_text(text)):
                f.write(json.dumps({"source": str(p), "chunk_id": idx, "chunk": c}) + "\n")
                chunks_written += 1

    (out / "papers_parsed.json").write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    print("RAG_INGEST_DONE files=", len(files), "chunks=", chunks_written, "out=", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
