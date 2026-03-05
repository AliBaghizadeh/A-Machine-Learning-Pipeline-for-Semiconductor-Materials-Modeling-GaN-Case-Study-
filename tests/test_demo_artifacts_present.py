import json
from pathlib import Path


def test_demo_manifest_present():
    p = Path("app/demo_data/run_manifest.json")
    assert p.exists(), "Missing app/demo_data/run_manifest.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    assert "gates" in d
    assert "energy_gate" in (d["gates"] or {})
    assert "force_gate" in (d["gates"] or {})


def test_demo_rag_snippet_present():
    p = Path("app/demo_data/rag/demo_paper_snippet.txt")
    assert p.exists(), "Missing demo RAG snippet"
    assert len(p.read_text(encoding="utf-8").strip()) > 10

