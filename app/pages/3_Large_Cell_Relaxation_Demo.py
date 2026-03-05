from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.lib.demo_io import get_demo_dir, load_manifest, path_exists, pretty_path
from app.lib.relax_utils import format_float, read_text_preview, summarize_atoms_state


st.set_page_config(page_title="🧪 MLIP Relaxation (Demo)", layout="wide")
st.title("MLIP Relaxation (Demo)")
st.info(
    "Goal: show how a trained interatomic potential can quickly relax large atomic structures.\n\n"
    "Relaxation = moving atoms slightly to reduce forces (a fast geometry optimization).\n\n"
    "This demo never runs DFT. It uses a frozen ML model and either loads saved examples or runs a small local optimization."
)
st.markdown(
    "**Why it matters**\n"
    "- DFT is accurate but slow for large cells.\n"
    "- An ML interatomic potential (MLIP) is a fast surrogate.\n"
    "- This page demonstrates scalability on larger atom counts."
)

root = Path(__file__).resolve().parents[2]
precomp = root / "analysis" / "results" / "large_scale_mlip"
demo_dir = get_demo_dir()
m = load_manifest(demo_dir)
model_path = (m.get("frozen") or {}).get("model_path")
model_ok = path_exists(model_path)

structures_dir = demo_dir / "structures"
cifs = sorted(structures_dir.glob("*.cif"))

if "relax_demo_last_result" not in st.session_state:
    st.session_state["relax_demo_last_result"] = None

mode = st.radio(
    "Mode",
    ["Demo mode (saved examples)", "Local compute mode (quick run)"],
    horizontal=True,
    help="Demo mode only displays saved artifacts. Local compute mode can run a short relaxation on your machine.",
)
is_demo_mode = mode.startswith("Demo mode")

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Controls")
    st.caption("Step 1: choose structure and settings. Step 2: run local relaxation (optional).")
    st.write("Frozen model:", pretty_path(model_path))
    st.write("Model status:", "✅ Available" if model_ok else "❌ Missing")

    if not cifs:
        st.warning(f"No CIFs found in demo folder: {pretty_path(structures_dir)}")
        cif_sel = None
    else:
        cif_sel = st.selectbox(
            "Structure to relax",
            cifs,
            format_func=lambda p: pretty_path(p),
            help="Pick a structure file from the demo bundle.",
        )

    max_steps = int(
        st.number_input(
            "Max optimization steps",
            min_value=1,
            max_value=50,
            value=20,
            step=1,
            help="Upper bound on optimization iterations (capped at 50 for demo safety).",
        )
    )
    force_tol = float(
        st.number_input(
            "Force tolerance (quality vs speed, eV/A)",
            min_value=0.05,
            max_value=1.0,
            value=0.20,
            step=0.05,
            format="%.2f",
            help="Lower means tighter convergence but longer runtime. Higher means faster but less precise.",
        )
    )
    device_choice = st.selectbox(
        "Compute device",
        ["auto", "cpu", "cuda"],
        index=0,
        help="Auto picks CUDA if available, otherwise CPU.",
    )

    cuda_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    if device_choice == "cuda" and not cuda_available:
        st.warning("CUDA was selected but is not available in this environment. The run will fall back to CPU.")

    st.caption(
        "What happens when you click Run: the app loads the selected structure, applies the frozen ML model, "
        "runs a short optimization, and saves a relaxed CIF plus logs."
    )

    run_disabled = is_demo_mode or (cif_sel is None) or (not model_ok)
    run_help = "Switch to Local compute mode to enable this button."
    if not is_demo_mode and not model_ok:
        run_help = "Model missing. Run `make demo-artifacts` first."
    if not is_demo_mode and cif_sel is None:
        run_help = "No input CIF found in demo_data/structures."

    run_clicked = st.button(
        "Run quick local relaxation",
        disabled=run_disabled,
        help=run_help,
    )

    if is_demo_mode:
        st.info("Demo mode is active: compute is disabled and only saved examples are shown.")

    # Optional tiny structure preview (no heavy viewer dependency).
    if cif_sel is not None:
        try:
            from ase.io import read

            preview_atoms = read(str(cif_sel))
            p1, p2 = st.columns(2)
            with p1:
                st.metric("Atoms", len(preview_atoms))
            with p2:
                st.metric("Formula", preview_atoms.get_chemical_formula())
            try:
                a, b, c, _, _, _ = preview_atoms.cell.cellpar()
                st.caption(f"Cell lengths (A): a={a:.2f}, b={b:.2f}, c={c:.2f}")
            except Exception:
                pass
        except Exception:
            pass

with right:
    st.subheader("Saved examples (no compute)")
    st.caption("Previously generated relaxation outputs. These are loaded directly and are safe for offline demo mode.")
    if not precomp.exists():
        st.info(f"Missing directory: {pretty_path(precomp)}")
    else:
        files = sorted(precomp.glob("*"))
        if not files:
            st.info("No saved relaxation examples found.")
        else:
            top_n = 10
            st.write(f"Found {len(files)} files.")
            st.code("\n".join(str(p.relative_to(root)) for p in files[:top_n]), language="text")
            if len(files) > top_n:
                with st.expander("Show remaining files"):
                    st.code("\n".join(str(p.relative_to(root)) for p in files[top_n:]), language="text")

if run_clicked and cif_sel is not None:
    if not model_path or not Path(model_path).exists():
        st.error("Model path missing or not found. Run `make demo-artifacts` or update the manifest.")
    else:
        missing = []
        try:
            from ase.io import read, write
            from ase.optimize import LBFGS
        except Exception:
            missing.append("ASE (`pip install ase`)")
        try:
            from mace.calculators import MACECalculator
        except Exception:
            missing.append("MACE (`pip install mace-torch`)")

        if missing:
            st.error("Missing required local dependencies for relaxation.")
            for dep in missing:
                st.write(f"- {dep}")
        else:
            device = device_choice
            if device_choice == "auto":
                device = "cuda" if cuda_available else "cpu"
            elif device_choice == "cuda" and not cuda_available:
                device = "cpu"

            atoms = read(str(cif_sel))
            atoms.calc = MACECalculator(model_paths=str(model_path), device=device)
            before = summarize_atoms_state(atoms)

            out_dir = root / "analysis" / "results" / "large_scale_mlip_demo"
            out_dir.mkdir(parents=True, exist_ok=True)
            tag = Path(str(cif_sel)).stem
            traj = out_dir / f"{tag}.traj"
            log = out_dir / f"{tag}.opt.log"

            dyn = LBFGS(atoms, trajectory=str(traj), logfile=str(log))
            dyn.run(fmax=force_tol, steps=max_steps)
            after = summarize_atoms_state(atoms)

            out_cif = out_dir / f"{tag}__relaxed.cif"
            write(str(out_cif), atoms)
            steps_taken = getattr(dyn, "nsteps", None)

            st.session_state["relax_demo_last_result"] = {
                "structure": pretty_path(cif_sel),
                "atoms": len(atoms),
                "device": device,
                "steps_requested": max_steps,
                "steps_taken": steps_taken,
                "before": before,
                "after": after,
                "out_cif": pretty_path(out_cif),
                "traj": pretty_path(traj),
                "log": pretty_path(log),
                "log_preview": read_text_preview(log, max_lines=40),
            }
            st.success("Local relaxation completed.")

st.divider()
st.subheader("Run results")
result = st.session_state.get("relax_demo_last_result")
if not result:
    st.info("No local run result yet. Use Local compute mode and click 'Run quick local relaxation'.")
else:
    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Structure", result.get("structure", "unknown"))
        st.metric("Atom count", result.get("atoms", "unknown"))
    with r2:
        st.metric("Device used", str(result.get("device", "unknown")).upper())
        st.metric(
            "Steps (taken/requested)",
            f"{result.get('steps_taken', 'unknown')}/{result.get('steps_requested', 'unknown')}",
        )
    with r3:
        b_e = result.get("before", {}).get("energy_eV")
        a_e = result.get("after", {}).get("energy_eV")
        delta_e = None
        try:
            if b_e is not None and a_e is not None:
                delta_e = float(a_e) - float(b_e)
        except Exception:
            delta_e = None
        st.metric("Energy before (eV)", format_float(b_e, digits=6))
        st.metric(
            "Energy after (eV)",
            format_float(a_e, digits=6),
            delta=(f"{delta_e:+.6f}" if delta_e is not None else None),
        )

    f1, f2 = st.columns(2)
    with f1:
        st.metric("Max force before (eV/A)", format_float(result.get("before", {}).get("max_force_eV_per_A"), digits=6))
    with f2:
        b_f = result.get("before", {}).get("max_force_eV_per_A")
        a_f = result.get("after", {}).get("max_force_eV_per_A")
        delta_f = None
        try:
            if b_f is not None and a_f is not None:
                delta_f = float(a_f) - float(b_f)
        except Exception:
            delta_f = None
        st.metric(
            "Max force after (eV/A)",
            format_float(a_f, digits=6),
            delta=(f"{delta_f:+.6f}" if delta_f is not None else None),
        )

    st.markdown("**Output files**")
    st.code(
        f"Relaxed CIF: {result.get('out_cif')}\n"
        f"Trajectory: {result.get('traj')}\n"
        f"Optimizer log: {result.get('log')}",
        language="text",
    )

    with st.expander("Optimizer log preview (first 40 lines)"):
        st.code(result.get("log_preview", "(no log preview)"), language="text")

    st.markdown("**What to look for**")
    st.markdown(
        "- Energy decreases\n"
        "- Max force decreases\n"
        "- A relaxed CIF is saved"
    )
