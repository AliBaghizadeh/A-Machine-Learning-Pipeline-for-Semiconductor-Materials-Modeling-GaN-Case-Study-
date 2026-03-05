from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import streamlit as st

from app.lib.demo_io import get_demo_dir, pretty_path, read_text_if_exists, sanitize_text
from rag.ollama_client import generate, is_running
from rag.openai_client import generate as openai_generate


# UX intent:
# - Demo-first: useful offline even without any LLM backend.
# - Plain language for non-experts; keep jargon and provider settings in advanced controls.
# - Preserve existing ingest/index/report workflows and optional LLM code paths.

st.set_page_config(page_title="📚 RAG Assistant", layout="wide")
st.title("RAG Demo: GaN DFT Setup from Literature (Offline)")
st.caption("Showcase feature: summarize literature setup choices and compare them to this local demo run.")

root = Path(__file__).resolve().parents[2]
demo_dir = get_demo_dir()


def _load_paper_excerpts() -> list[dict]:
    p = root / "rag" / "index" / "chunks.jsonl"
    excerpts: list[dict] = []
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    excerpts.append(
                        {
                            "source": str(row.get("source") or ""),
                            "excerpt": str(row.get("chunk") or ""),
                        }
                    )
                except Exception:
                    continue
    if excerpts:
        return excerpts

    # Offline-safe fallback for demo mode.
    snippet_path = demo_dir / "rag" / "demo_paper_snippet.txt"
    snippet = read_text_if_exists(snippet_path)
    if snippet.strip():
        return [{"source": str(snippet_path), "excerpt": snippet.strip()}]
    return []


def _top_k_excerpts(query: str, excerpts: list[dict], k: int = 4) -> list[dict]:
    q = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
    if not q:
        return [{**x, "score": 0} for x in excerpts[:k]]
    scored = []
    for c in excerpts:
        toks = set(re.findall(r"[a-zA-Z0-9]+", str(c.get("excerpt", "")).lower()))
        overlap = len(q & toks)
        scored.append((overlap, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, row in scored[:k]:
        out.append({**row, "score": score})
    return out


def _run_cmd(cmd: list[str], env: dict | None = None) -> tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return p.returncode, p.stdout


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _clean_param_rows(rows: list[dict]) -> list[dict]:
    out = []
    seen = set()
    for r in rows:
        param = str(r.get("param") or "").strip()
        value = str(r.get("value") or "").strip()
        paper = str(r.get("paper_title") or "").strip()
        if not paper:
            cit = str(r.get("citation") or "")
            paper = cit.split("#")[0].strip() if cit else ""
        if not param or not value or not paper:
            continue
        k = (param, value, paper)
        if k in seen:
            continue
        seen.add(k)
        out.append({"param": param, "value": value, "paper_title": paper})
    out.sort(key=lambda x: (x["param"], x["paper_title"], x["value"]))
    return out


def _canonical_param_name(param: str) -> str:
    p = param.strip().lower()
    if not p:
        return "unknown"
    if "xc" in p or "functional" in p:
        return "exchange-correlation functional"
    if "ecut" in p or "cutoff" in p:
        return "plane-wave cutoff (eV)"
    if "kpoint" in p or "k-point" in p or "kpts" in p:
        return "k-point mesh"
    if "smear" in p:
        return "smearing"
    if "convergence" in p or "conv" in p:
        return "convergence criteria"
    return param


def extract_param_stats(rows: list[dict]) -> dict:
    """
    Build simple deterministic stats for template summaries.
    """
    grouped: dict[str, list[str]] = {}
    for r in rows:
        p = _canonical_param_name(str(r.get("param") or ""))
        v = str(r.get("value") or "").strip()
        if not p or not v:
            continue
        grouped.setdefault(p, []).append(v)

    out: dict[str, dict] = {}
    for p, vals in grouped.items():
        counts = Counter(vals)
        most_common_value, most_common_n = counts.most_common(1)[0]

        numeric_vals = []
        for v in vals:
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", v)
            if m:
                try:
                    numeric_vals.append(float(m.group(0)))
                except Exception:
                    pass

        stat = {
            "n_rows": len(vals),
            "n_unique": len(set(vals)),
            "common_value": most_common_value,
            "common_count": most_common_n,
        }
        if numeric_vals:
            stat["min"] = min(numeric_vals)
            stat["max"] = max(numeric_vals)
        out[p] = stat
    return out


def _load_latest_report_markdown() -> Path | None:
    art = root / "analysis" / "artifacts"
    reports: list[Path] = []
    if art.exists():
        reports.extend(sorted(art.glob("golden_run_*/rag/rag_report.md")))
        reports.extend(sorted(art.glob("rag_report_*/rag_report.md")))
    demo_report = demo_dir / "rag" / "rag_report.md"
    if demo_report.exists():
        reports.append(demo_report)
    if not reports:
        return None
    # Prefer most recently updated report, not lexical path order.
    try:
        return max(reports, key=lambda p: p.stat().st_mtime)
    except Exception:
        return reports[-1]


def _load_extracted_rows() -> tuple[list[dict], Path | None]:
    latest = _load_latest_report_markdown()
    search_paths = []
    if latest is not None:
        search_paths.append(latest.parent / "extracted_params_by_paper.csv")
        search_paths.append(latest.parent / "gan_dft_params_table.csv")
    search_paths.append(root / "rag" / "index" / "gan_dft_params_table.csv")
    search_paths.append(demo_dir / "rag" / "gan_dft_params_table.csv")

    rows: list[dict] = []
    for p in search_paths:
        rows = _read_csv(p)
        if rows:
            break
    if rows and ("paper_title" not in rows[0] or "param" not in rows[0]):
        rows = _clean_param_rows(rows)
    return rows, latest


def load_latest_run_manifest() -> dict:
    p = demo_dir / "run_manifest.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_local_dft_config_snapshot() -> dict:
    """
    Best-effort read of local DFT config snapshot (no external dependencies).
    """
    p = root / "configs" / "dft_tierb.yaml"
    txt = read_text_if_exists(p)
    if not txt.strip():
        return {}
    out = {}
    m_ecut = re.search(r"gpu_ecut_eV:\s*([0-9.eE+-]+)", txt)
    if m_ecut:
        try:
            out["plane_wave_cutoff_eV"] = float(m_ecut.group(1))
        except Exception:
            out["plane_wave_cutoff_eV"] = m_ecut.group(1)
    for key in ("energy", "density", "eigenstates"):
        m = re.search(rf"{key}:\s*([0-9.eE+-]+)", txt)
        if m:
            out[f"convergence_{key}"] = m.group(1)
    return out


def render_metric_cards(rows: list[dict], manifest: dict) -> None:
    gates = manifest.get("gates") or {}
    eg = gates.get("energy_gate") or {}
    fg = gates.get("force_gate") or {}
    n_papers = len({str(r.get("paper_title") or r.get("citation") or "") for r in rows if r})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Parameter rows", len(rows))
    with c2:
        st.metric("Paper sources", n_papers)
    with c3:
        st.metric("Energy check", "Passed" if eg.get("pass") else "Failed" if eg.get("pass") is False else "Unknown")
    with c4:
        st.metric("Force check", "Passed" if fg.get("pass") else "Failed" if fg.get("pass") is False else "Unknown")


def _build_template_summary(
    *,
    question: str,
    stats: dict,
    manifest: dict,
    local_cfg: dict,
    evidence: list[dict],
    llm_note: str | None = None,
) -> str:
    gates = manifest.get("gates") or {}
    eg = gates.get("energy_gate") or {}
    fg = gates.get("force_gate") or {}
    dset = (manifest.get("dataset") or {}).get("dataset_stats") or {}

    lines = []
    lines.append("### Summary (template)")
    lines.append(f"Question: {question}")
    if llm_note:
        lines.append(f"_Note: {llm_note}_")
    lines.append("")

    lines.append("**Common literature setup trends (from local papers):**")
    if not stats:
        lines.append("- No extracted parameter rows found yet. Use 'Refresh data (optional)' to update.")
    else:
        order = [
            "exchange-correlation functional",
            "plane-wave cutoff (eV)",
            "k-point mesh",
            "smearing",
            "convergence criteria",
        ]
        keys = [k for k in order if k in stats] + [k for k in sorted(stats.keys()) if k not in order]
        for k in keys[:8]:
            s = stats[k]
            if "min" in s and "max" in s:
                lines.append(f"- {k}: range {s['min']:.4g} to {s['max']:.4g} (n={s['n_rows']})")
            else:
                lines.append(f"- {k}: most common value '{s['common_value']}' (n={s['common_count']}/{s['n_rows']})")

    lines.append("")
    lines.append("**Compare to local demo configuration and outputs:**")
    if local_cfg:
        cutoff = local_cfg.get("plane_wave_cutoff_eV")
        if cutoff is not None:
            lines.append(f"- Local plane-wave cutoff (config snapshot): {cutoff} eV")
            lit_cut = stats.get("plane-wave cutoff (eV)") or {}
            if "min" in lit_cut and "max" in lit_cut:
                in_range = lit_cut["min"] <= float(cutoff) <= lit_cut["max"]
                lines.append(
                    f"- Cutoff vs literature range: {'within range' if in_range else 'outside range'} "
                    f"({lit_cut['min']:.4g} to {lit_cut['max']:.4g} eV)"
                )
    else:
        lines.append("- Local DFT config snapshot not found (`configs/dft_tierb.yaml`).")

    if dset:
        lines.append(
            f"- Dataset size: {dset.get('total_structures', 'unknown')} filtered structures "
            f"(train={dset.get('train', 'unknown')}, val={dset.get('val', 'unknown')}, test={dset.get('test', 'unknown')})."
        )
    lines.append(
        f"- Quality checks: energy={'pass' if eg.get('pass') else 'fail' if eg.get('pass') is False else 'unknown'}, "
        f"force={'pass' if fg.get('pass') else 'fail' if fg.get('pass') is False else 'unknown'}."
    )

    if evidence:
        lines.append("")
        lines.append("**Evidence used:**")
        for ev in evidence[:4]:
            src = Path(str(ev.get("source") or "")).name or "unknown"
            lines.append(f"- {src}")
    return "\n".join(lines)


def _wsl_nameserver_host() -> str | None:
    try:
        p = Path("/etc/resolv.conf")
        if not p.exists():
            return None
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("nameserver "):
                return line.split()[1]
    except Exception:
        return None
    return None


st.subheader("What this does")
st.markdown(
    "1. Collects local papers (txt preferred for demo).\n"
    "2. Extracts common DFT setup parameters (exchange-correlation functional, plane-wave cutoff, "
    "k-point mesh, smearing, convergence).\n"
    "3. Compares literature patterns with your local run configuration and validation outputs."
)
st.info("Offline demo: no external APIs required. No DFT executed here.")

with st.expander("Refresh data (optional)", expanded=False):
    sources = st.text_input(
        "Local paper folder (relative path)",
        value="literature",
        help="Folder with local papers to parse (txt/pdf).",
    )

    col_b1, col_b2 = st.columns(2)
    if col_b1.button("Collect & prepare papers"):
        env = dict(os.environ)
        rc, out = _run_cmd(
            [sys.executable, "-m", "rag.ingest.ingest_papers", "--sources", sources, "--out", "rag/index"],
            env=env,
        )
        rc2, out2 = _run_cmd(
            [sys.executable, "-m", "rag.index.build_index", "--in", "rag/index", "--out", "rag/index"],
            env=env,
        )
        if rc == 0 and rc2 == 0:
            st.success("Search library updated.")
        else:
            st.error("Collect & prepare failed. See logs.")
        with st.expander("Logs (sanitized)"):
            st.code(sanitize_text(out + "\n" + out2, root=root), language="text")

    if col_b2.button("Generate literature summary report"):
        rc, out = _run_cmd(
            [sys.executable, "-m", "rag.run_agent", "--index", "rag/index", "--demo-data", "app/demo_data", "--out", "analysis/artifacts"]
        )
        if rc == 0:
            st.success("Report created (CSV + markdown).")
        else:
            st.error("Report generation failed.")
        with st.expander("Logs (sanitized)"):
            st.code(sanitize_text(out, root=root), language="text")

    st.caption(
        "Collect & prepare builds a searchable library of paper excerpts. "
        "Generate report writes a CSV parameter table and short markdown summary."
    )

# Query section (demo-first)
st.subheader("Ask a question (demo-first)")
if "rag_question" not in st.session_state:
    st.session_state["rag_question"] = "Compare literature ranges to my current config"

qp1, qp2, qp3 = st.columns(3)
if qp1.button("What DFT parameters are most common for wurtzite GaN bulk?"):
    st.session_state["rag_question"] = "What DFT parameters are most common for wurtzite GaN bulk?"
if qp2.button("Compare literature ranges to my current config"):
    st.session_state["rag_question"] = "Compare literature ranges to my current config"
if qp3.button("Summarize my latest run outputs (model, dataset size, gates)"):
    st.session_state["rag_question"] = "Summarize my latest run outputs (model, dataset size, gates)"

question = st.text_input("Question", key="rag_question", help="Use a preset or type your own question.")
answer_mode = st.selectbox(
    "Answer mode",
    ["Offline demo (template)", "Local Ollama (optional)", "OpenAI (cloud, optional)"],
    index=0,
    help="Offline demo mode is deterministic and requires no model server.",
)
enable_real_llm = st.checkbox("Enable real LLM calls", value=False, help="OFF by default for demo safety.")

ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
openai_api_key = os.environ.get("OPENAI_API_KEY", "")
openai_base_url = os.environ.get("OPENAI_BASE_URL", "")
model = "qwen3:4b"

with st.expander("Advanced settings (experimental / not needed for demo)", expanded=False):
    st.caption("Local Ollama = local only. OpenAI = sends text to cloud.")
    if answer_mode == "Local Ollama (optional)":
        st.warning("Local only: requires a reachable Ollama server.")
        ns_host = _wsl_nameserver_host()
        host_options = ["http://localhost:11434", "http://127.0.0.1:11434"]
        if ns_host:
            host_options.append(f"http://{ns_host}:11434")
        preset_host = st.selectbox("Host preset", host_options, index=0)
        ollama_host = st.text_input("Ollama host", value=preset_host)
        model = st.text_input("Ollama model", value="qwen3:4b")
        if st.button("Test local Ollama connection"):
            if is_running(host=ollama_host):
                st.success(f"Connected: {ollama_host}")
            else:
                st.error(f"Not reachable: {ollama_host}")

    if answer_mode == "OpenAI (cloud, optional)":
        st.warning("Cloud mode: question + evidence text may be sent to OpenAI.")
        openai_api_key = st.text_input("OPENAI_API_KEY", value=openai_api_key, type="password")
        openai_base_url = st.text_input("OpenAI base_url (optional)", value=openai_base_url)
        model = st.text_input("OpenAI model", value="gpt-4o-mini")

excerpts = _load_paper_excerpts()
retrieved = _top_k_excerpts(question, excerpts, k=4)
rows, latest_report = _load_extracted_rows()
stats = extract_param_stats(rows)
manifest = load_latest_run_manifest()
local_cfg = _load_local_dft_config_snapshot()

render_metric_cards(rows, manifest)

if "rag_answer_text" not in st.session_state:
    st.session_state["rag_answer_text"] = ""
if "rag_answer_mode" not in st.session_state:
    st.session_state["rag_answer_mode"] = "template"

if st.button("Ask", type="primary"):
    llm_note = None
    answer_text = _build_template_summary(
        question=question,
        stats=stats,
        manifest=manifest,
        local_cfg=local_cfg,
        evidence=retrieved,
        llm_note=None,
    )
    used_mode = "template"

    if answer_mode != "Offline demo (template)":
        if not enable_real_llm:
            llm_note = "LLM calls disabled in demo mode."
            answer_text = _build_template_summary(
                question=question,
                stats=stats,
                manifest=manifest,
                local_cfg=local_cfg,
                evidence=retrieved,
                llm_note=llm_note,
            )
            used_mode = "template"
        else:
            context = []
            for i, ev in enumerate(retrieved, start=1):
                src = Path(str(ev.get("source") or "")).name
                context.append(f"[Excerpt {i} | {src}]\n{ev.get('excerpt','')}\n")
            prompt = (
                "You are a focused assistant for GaN DFT setup. "
                "Use only the provided evidence. "
                "Return practical bullets for parameters and compare to local run.\n\n"
                f"Question:\n{question}\n\n"
                "Evidence:\n" + "\n".join(context)
            )
            try:
                if answer_mode == "Local Ollama (optional)":
                    if not is_running(host=ollama_host):
                        raise RuntimeError(f"Ollama not reachable: {ollama_host}")
                    answer_text = generate(prompt=prompt, model=model, timeout_s=120.0, host=ollama_host)
                    used_mode = "ollama"
                elif answer_mode == "OpenAI (cloud, optional)":
                    if not openai_api_key:
                        raise RuntimeError("Missing OPENAI_API_KEY")
                    answer_text = openai_generate(
                        prompt=prompt,
                        model=model,
                        timeout_s=120.0,
                        api_key=openai_api_key,
                        base_url=openai_base_url or None,
                    )
                    used_mode = "openai"
            except Exception as e:
                st.error(f"LLM call failed: {e}. Falling back to template summary.")
                answer_text = _build_template_summary(
                    question=question,
                    stats=stats,
                    manifest=manifest,
                    local_cfg=local_cfg,
                    evidence=retrieved,
                    llm_note="LLM failed; using deterministic template.",
                )
                used_mode = "template"

    st.session_state["rag_answer_text"] = answer_text
    st.session_state["rag_answer_mode"] = used_mode

answer_text = st.session_state.get("rag_answer_text") or _build_template_summary(
    question=question,
    stats=stats,
    manifest=manifest,
    local_cfg=local_cfg,
    evidence=retrieved,
    llm_note="Template answer shown by default.",
)
used_mode = st.session_state.get("rag_answer_mode", "template")

st.markdown("### Summary")
if used_mode == "template":
    st.info("Summary mode: deterministic template (offline demo).")
elif used_mode == "ollama":
    st.success("Summary mode: Local Ollama.")
else:
    st.warning("Summary mode: OpenAI cloud.")
st.markdown(answer_text)

st.divider()
st.subheader("Extracted parameter table")
if rows:
    display_rows = []
    for r in rows:
        display_rows.append(
            {
                "parameter": _canonical_param_name(str(r.get("param") or "")),
                "value": r.get("value"),
                "paper title": r.get("paper_title") or Path(str(r.get("citation") or "")).name,
            }
        )
    st.dataframe(display_rows, width="stretch")
else:
    st.info("No extracted parameter table found yet. Use 'Refresh data (optional)' to build it.")

st.subheader("Latest report summary")
if latest_report is not None:
    st.write("Latest report:", pretty_path(latest_report, root=root))
    st.markdown(sanitize_text(read_text_if_exists(latest_report), root=root))
else:
    st.info("No report markdown found yet.")

st.subheader("Evidence excerpts")
if retrieved:
    for i, ev in enumerate(retrieved, start=1):
        src = Path(str(ev.get("source") or "")).name or "unknown"
        score = ev.get("score", 0)
        with st.expander(f"Excerpt {i}: {src} (match score: {score})", expanded=False):
            st.write(ev.get("excerpt") or "(empty excerpt)")
else:
    st.info("No paper excerpts found. Using the demo snippet fallback is recommended.")

with st.expander("Fallback demo snippet (local text)"):
    snippet = read_text_if_exists(demo_dir / "rag" / "demo_paper_snippet.txt")
    if snippet:
        st.code(sanitize_text(snippet, root=root), language="text")
    else:
        st.write("No fallback snippet found.")
