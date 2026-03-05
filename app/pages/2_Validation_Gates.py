from __future__ import annotations

import streamlit as st

from app.lib.demo_io import get_demo_dir, load_manifest

# UX intent:
# - Keep the same gate data/logic from the manifest, but present it in plain language.
# - Show a fast pass/fail summary first for non-experts.
# - Keep details and raw technical JSON available in collapsed sections.
# - Translate internal jargon into beginner-friendly labels in the main UI.


st.set_page_config(page_title="⚡ Validation Gates", layout="wide")
st.title("⚡ Quality Checks")
st.caption(
    "Sanity checks for the trained ML potential (a learned force field) against reference calculations (DFT)."
)
st.info("These are quick demo sanity checks, not full scientific validation.")

demo_dir = get_demo_dir()
m = load_manifest(demo_dir)

gates = m.get("gates") or {}
eg = gates.get("energy_gate") or {}
fg = gates.get("force_gate") or {}
energy_rows = eg.get("results") or []


def _status_label(v: bool | None) -> str:
    if v is True:
        return "✅ Passed"
    if v is False:
        return "❌ Failed"
    return "⚪ Unknown"


def _status_icon(v: bool | None) -> str:
    if v is True:
        return "✅"
    if v is False:
        return "❌"
    return "⚪"


def _friendly_structure_name(raw: object) -> str:
    s = str(raw or "unknown").strip()
    if not s:
        return "unknown"
    return " ".join(s.replace("_", " ").split())


def _selection_label(raw: object) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return "unknown"
    if s == "coord":
        return "atoms with unusual local coordination (near defect)"
    return s


def _source_label(raw: object) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return "unknown"
    if s in {"sr", "tier_b_sr"}:
        return "geometry from short relaxation"
    if s in {"sp", "tier_b_sp"}:
        return "single-step geometry (fixed positions)"
    return s


st.subheader("What this page tells you")
st.markdown(
    "- **Energy check:** does the model give similar energies to the reference calculation (DFT)?\n"
    "- **Force check:** does the model push atoms in similar directions near the defect region?\n"
    "- **Outcome:** a quick pass/fail signal for demo readiness."
)

st.divider()

energy_pass = eg.get("pass")
force_pass = fg.get("pass")
energy_threshold = eg.get("threshold_eV_per_atom")
force_mae = fg.get("selected_mae_eV_per_A")
force_max = fg.get("selected_max_eV_per_A")

st.subheader("Plain language summary")
st.markdown(
    "Energy mismatch per atom (`|ΔE|/atom`) asks: *how far is model energy from reference energy per atom?*  \n"
    "Force MAE/MAX asks: *what is the average and worst-case force mismatch in the region of interest?*"
)

card1, card2 = st.columns(2)
with card1:
    st.markdown("**Energy check**")
    st.markdown(f"Status: {_status_label(energy_pass)}")
    st.metric("Threshold (eV/atom)", f"{energy_threshold}" if energy_threshold is not None else "unknown")
    st.caption("Compares model energy vs reference calculation (DFT).")

with card2:
    st.markdown("**Force check**")
    st.markdown(f"Status: {_status_label(force_pass)}")
    st.metric("Average force mismatch (MAE, eV/A)", f"{force_mae}" if force_mae is not None else "unknown")
    st.metric("Worst-case force mismatch (MAX, eV/A)", f"{force_max}" if force_max is not None else "unknown")
    st.caption("Compares model forces vs reference forces in a region near the defect.")

if energy_pass is True and force_pass is True:
    st.success(
        "Both checks passed. The trained potential is suitable for this demo's relaxation and comparison steps."
    )
elif energy_pass is False or force_pass is False:
    st.warning("At least one check failed. Review details below before relying on this model in the demo.")
else:
    st.info("Check status is incomplete (missing fields in manifest).")

st.divider()

with st.expander("Energy check details", expanded=False):
    st.markdown("**Energy mismatch per atom (`|ΔE|/atom`)**")
    st.caption("Lower values mean the model and reference energies are closer.")
    st.markdown(f"Threshold: **{energy_threshold if energy_threshold is not None else 'unknown'} eV/atom**")

    if energy_rows:
        energy_table = []
        raw_ids = []
        for r in energy_rows:
            raw_id = r.get("structure_id")
            de = r.get("abs_de_per_atom_eV")
            energy_table.append(
                {
                    "Structure": _friendly_structure_name(raw_id),
                    "Energy mismatch per atom (eV)": de,
                    "Status": f"{_status_icon(r.get('pass'))} {_status_label(r.get('pass')).replace('✅ ', '').replace('❌ ', '').replace('⚪ ', '')}",
                }
            )
            raw_ids.append({"Display name": _friendly_structure_name(raw_id), "Raw ID": raw_id})

        st.dataframe(energy_table, width="stretch", hide_index=True)

        with st.expander("Raw IDs", expanded=False):
            st.dataframe(raw_ids, width="stretch", hide_index=True)

        try:
            import pandas as pd

            chart_df = pd.DataFrame(
                [
                    {
                        "Structure": _friendly_structure_name(r.get("structure_id")),
                        "energy_mismatch_eV_per_atom": r.get("abs_de_per_atom_eV"),
                    }
                    for r in energy_rows
                ]
            )
            if not chart_df.empty and "energy_mismatch_eV_per_atom" in chart_df.columns:
                st.markdown("**Energy mismatch by structure**")
                st.caption(f"Threshold: {energy_threshold if energy_threshold is not None else 'unknown'} eV/atom")
                st.bar_chart(chart_df.set_index("Structure"), width="stretch")
        except Exception:
            pass
    else:
        st.info("No energy-check rows found in manifest.")

st.divider()

with st.expander("Force check details", expanded=False):
    selected_atoms = fg.get("selected_atoms")
    total_atoms = fg.get("total_atoms")
    selected_pct = None
    try:
        if selected_atoms is not None and total_atoms not in (None, 0):
            selected_pct = 100.0 * float(selected_atoms) / float(total_atoms)
    except Exception:
        selected_pct = None

    selection_text = _selection_label(fg.get("selection"))
    source_text = _source_label(fg.get("source") or fg.get("dft_source"))

    region_rows = [
        {"Metric": "Selection", "Value": selection_text},
        {"Metric": "Geometry source", "Value": source_text},
        {
            "Metric": "Atoms evaluated",
            "Value": (
                f"{selected_atoms} / {total_atoms}"
                + (f" ({selected_pct:.1f}%)" if selected_pct is not None else "")
            ),
        },
    ]
    st.markdown("**Region near the defect / region of interest**")
    st.dataframe(region_rows, width="stretch", hide_index=True)

    force_rows = [
        {"Metric": "Average force mismatch (MAE, eV/A)", "Value": force_mae if force_mae is not None else "unknown"},
        {"Metric": "Worst-case force mismatch (MAX, eV/A)", "Value": force_max if force_max is not None else "unknown"},
        {"Metric": "Result", "Value": _status_label(force_pass)},
    ]
    st.dataframe(force_rows, width="stretch", hide_index=True)

    try:
        import pandas as pd

        force_chart_df = pd.DataFrame(
            [
                {"Metric": "MAE", "Error (eV/A)": force_mae},
                {"Metric": "MAX", "Error (eV/A)": force_max},
            ]
        )
        if not force_chart_df.empty:
            st.markdown("**Force mismatch summary (average vs worst-case)**")
            st.bar_chart(force_chart_df.set_index("Metric"), width="stretch")
    except Exception:
        pass

st.divider()

with st.expander("How these checks are computed (1 minute)", expanded=False):
    st.markdown(
        "1. **Energy check:** compare model and reference energies for selected structures, then compute mismatch per atom (`|ΔE|/atom`).\n"
        "2. **Force check:** compare model and reference forces in a region near the defect, then summarize with MAE (average) and MAX (worst-case).\n"
        "3. **Pass/fail:** compare those values against fixed thresholds in the manifest."
    )
    st.caption("These checks are designed as demo quality checks, not full physical validation.")

with st.expander("Advanced / raw JSON", expanded=False):
    st.markdown("**Energy gate JSON**")
    st.json(eg)
    st.markdown("**Force gate JSON**")
    st.json(fg)
