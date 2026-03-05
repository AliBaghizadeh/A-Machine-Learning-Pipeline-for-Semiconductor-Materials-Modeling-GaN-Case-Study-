import csv
import subprocess
import sys
from pathlib import Path


def test_rag_index_builds_and_writes_table(tmp_path: Path):
    out = tmp_path / "rag_index"
    out.mkdir(parents=True, exist_ok=True)

    # Ingest (falls back to demo snippet if rag/sources is empty).
    subprocess.run(
        [sys.executable, "-m", "rag.ingest.ingest_papers", "--sources", "rag/sources", "--out", str(out)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Build index + param table (regex baseline; no Ollama required).
    subprocess.run(
        [sys.executable, "-m", "rag.index.build_index", "--in", str(out), "--out", str(out)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    table = out / "gan_dft_params_table.csv"
    assert table.exists()

    with open(table, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Schema: must have param/value/citation columns; should extract at least 1 row from the demo snippet.
    assert rows, "Expected at least one extracted param row"
    for r in rows:
        assert set(r.keys()) >= {"param", "value", "citation"}
        assert r["param"]
        assert r["value"]
        assert r["citation"]

