"""
Tier B DFT Calculations: Single-Point and Short Relaxations
===========================================================

GaN (wurtzite) pivot:
- Selection is tag-based (bulk/defect/rattle/...) from dft/structures/structure_info.json
- Default magnetic config is non-magnetic ('none')
- GPU execution requires the real CuPy backend (not gpaw.gpu.cpupy)

This Tier-B layer is intended to be lightweight and iterative (pipeline demo),
not publication-grade DFT.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.gpaw_params import (
    GPAW_PARAMS, HUBBARD_U, GPU_CONFIG, MAGNETIC_MOMENTS,
    RELAXATION_PARAMS, setup_gpu_environment, get_kpts_for_supercell,
    set_initial_magnetic_moments, LATTICE_PARAMS
)
from config.dft_budget import DFTBudgetTracker

# ASE imports
from ase import Atoms
from ase.io import read, write
from ase.optimize import LBFGS
from ase.parallel import parprint

# GPAW imports
try:
    from gpaw import GPAW, PW, LCAO
    from gpaw.mixer import Mixer
    from gpaw.occupations import FermiDirac
    GPAW_AVAILABLE = True
except ImportError:
    GPAW_AVAILABLE = False
    print("WARNING: GPAW not available. Running in test mode.")


def gpaw_gpu_supported():
    """Return True only if GPAW GPU backend dependencies are available."""
    try:
        from gpaw.gpu import cupy as gpaw_cupy
        # gpaw.gpu.cpupy is the fake backend (CPU emulation), not real GPU.
        if gpaw_cupy.__name__.startswith('gpaw.gpu.cpupy'):
            return False
        return True
    except Exception:
        return False


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
STRUCTURES_DIR = PROJECT_ROOT / "dft" / "structures"
RESULTS_DIR = PROJECT_ROOT / "dft" / "results"
TRAJECTORY_DIR = RESULTS_DIR / "trajectories"
LOGS_DIR = RESULTS_DIR / "logs"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
DATASET_DIR = PROJECT_ROOT / "mlip" / "data"

# Tier B calculation settings
TIER_B_CONFIG = {
    'single_point': {
        'n_calculations': 80,
        'maxiter': 30,
        'description': 'Single-point calculations for MLIP training',
    },
    'short_relax': {
        'n_calculations': 40,
        'fmax': 0.10,
        'max_steps': 30,
        'description': 'Short ionic relaxations for off-equilibrium data',
    },
}

# Legacy: older oxide workflow used composition buckets. GaN uses tag-based selection.
COMPOSITION_DISTRIBUTION = {}

# =============================================================================
# GPAW Calculator for Tier B
# =============================================================================

def create_gpaw_calculator_tierb(calc_type='single_point', supercell_size=(1,1,1),
                                 gpu=False, scf_overrides=None, mag_config='afm',
                                 txt_path=None, gpaw_overrides=None):
    """
    Create a GPAW calculator for Tier B calculations.
    
    Args:
        calc_type: 'single_point' or 'short_relax'
        supercell_size: tuple for k-point scaling
        gpu: whether to use GPU (often faster on CPU for small cells)
    
    Returns:
        GPAW calculator object
    """
    if not GPAW_AVAILABLE:
        return None
    
    # Setup GPU environment if requested
    if gpu:
        setup_gpu_environment()
    
    # Get base parameters
    params = GPAW_PARAMS.copy()
    # Explicit LCAO basis selection ('dzp') is incompatible with this GPAW
    # setup stack for Fe in our environment; use GPAW's default basis handling.
    params.pop('basis', None)

    # Translate legacy GPAW params to GPAW >=25 API
    sigma = params.pop('sigma', None)
    occ = params.get('occupations', None)
    if isinstance(occ, str) and occ.lower() == 'smearing':
        width = 0.1 if sigma is None else float(sigma)
        params['occupations'] = FermiDirac(width=width)
    elif sigma is not None and occ is None:
        params['occupations'] = FermiDirac(width=float(sigma))

    # Remove unsupported/legacy keys
    params.pop('verbose', None)

    # Convert legacy mixer dict to GPAW mixer object
    mixer_cfg = params.get('mixer')
    if isinstance(mixer_cfg, dict):
        method = mixer_cfg.get('method', 'Mixer')
        if method == 'Mixer':
            params['mixer'] = Mixer(
                beta=mixer_cfg.get('beta', 0.1),
                nmaxold=mixer_cfg.get('nmaxold', 5)
            )
        else:
            params['mixer'] = Mixer(beta=0.1, nmaxold=5)
    
    # Scale k-points for supercell (use config base kpts when present)
    base_kpts = tuple(GPAW_PARAMS.get('kpts', (6, 6, 4)))
    params['kpts'] = get_kpts_for_supercell(supercell_size, base_kpts=base_kpts)
    
    # Setup DFT+U in GPAW-25 setup-string format ("type:l,U").
    setups = {}
    for element, u_value in HUBBARD_U.items():
        if u_value > 0:
            setups[element] = f"paw:d,{float(u_value)}"
    params['setups'] = setups

    # Magnetism policy: default non-magnetic for GaN.
    mag_mode = (mag_config or 'none').lower()
    if mag_mode in ('none', 'nonmag', 'nm'):
        params['spinpol'] = False
    else:
        params['spinpol'] = True
    
    # Runtime configuration:
    # LCAO+GPU has been unstable in this environment. Use PW on GPU.
    if gpu:
        ecut = float(os.environ.get('MLIP_GPAW_GPU_ECUT', 350.0))
        params['mode'] = PW(ecut)
        params.pop('h', None)
        params['parallel'] = {'gpu': True}
    else:
        params['parallel'] = {'gpu': False}
    
    # Calculation-specific settings
    if calc_type == 'single_point':
        params['maxiter'] = TIER_B_CONFIG['single_point']['maxiter']
        params['txt'] = str(txt_path) if txt_path else str(LOGS_DIR / 'gpaw_tierb_sp.out')
        conv = params.get('convergence', {})
        if not isinstance(conv, dict):
            conv = {}
        conv['energy'] = 1e-4
        conv['density'] = 1e-3
        params['convergence'] = conv
    else:
        params['txt'] = str(txt_path) if txt_path else str(LOGS_DIR / 'gpaw_tierb_sr.out')

    # User-provided SCF overrides (CLI)
    if scf_overrides:
        if scf_overrides.get('maxiter') is not None:
            params['maxiter'] = int(scf_overrides['maxiter'])
        conv = params.get('convergence', {})
        if not isinstance(conv, dict):
            conv = {}
        for key in ('energy', 'density', 'forces', 'eigenstates'):
            if scf_overrides.get(key) is not None:
                conv[key] = float(scf_overrides[key])
        if conv:
            params['convergence'] = conv
    
    # Optional direct GPAW constructor overrides (used sparingly for special cases).
    if gpaw_overrides:
        params.update(gpaw_overrides)

    # Create calculator
    calc = GPAW(**params)
    
    return calc


def _json_default(obj):
    """JSON serializer for numpy/path objects in results."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _restart_kwargs_from_overrides(scf_overrides=None):
    """Build GPAW restart keyword overrides from CLI SCF options."""
    kwargs = {}
    if not scf_overrides:
        return kwargs
    # NOTE: Passing 'maxiter' when constructing from a .gpw restart has caused
    # ASECalculator.set() keyword errors in some GPAW builds. Keep restarts
    # minimal and rely on convergence overrides only.
    conv = {}
    for key in ('energy', 'density', 'forces', 'eigenstates'):
        if scf_overrides.get(key) is not None:
            conv[key] = float(scf_overrides[key])
    if conv:
        kwargs['convergence'] = conv
    return kwargs


def _already_counted_calculation(budget_tracker, calc_type, structure_id):
    """Return True if this exact calc already exists as completed in budget state."""
    records = budget_tracker.state.get('calculations', [])
    for rec in records:
        if (
            rec.get('type') == calc_type and
            rec.get('structure') == structure_id and
            rec.get('status') == 'completed'
        ):
            return True
    return False


def _atomic_json_dump(path: Path, payload):
    """Write JSON atomically to avoid half-written files during long runs."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2, default=_json_default)
    tmp.replace(path)


def _run_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _versioned_path(base_path: Path, tag: str) -> Path:
    return base_path.with_name(f"{base_path.stem}__{tag}{base_path.suffix}")

def _infer_supercell_size(atoms):
    """Infer (nx, ny, nz) from cell lengths vs GaN lattice parameters."""
    try:
        lengths = atoms.get_cell().lengths()
    except Exception:
        return (1, 1, 1)
    base = LATTICE_PARAMS.get('GaN', {}) if isinstance(LATTICE_PARAMS, dict) else {}
    a0 = float(base.get('a', 3.2))
    c0 = float(base.get('c', 5.2))
    nx = max(1, int(round(float(lengths[0]) / a0)))
    ny = max(1, int(round(float(lengths[1]) / a0)))
    nz = max(1, int(round(float(lengths[2]) / c0)))
    return (nx, ny, nz)


# =============================================================================
# Single-Point Calculation
# =============================================================================

def run_single_point(atoms, name, gpu=False, scf_overrides=None, mag_config='none',
                     use_restart=False, run_tag=None, sp_log_file=None,
                     supercell_size=None, structure_id=None, tags=None):
    """
    Run single-point energy and force calculation.
    
    Args:
        atoms: ASE Atoms object
        name: Identifier for this calculation
        gpu: Whether to use GPU
    
    Returns:
        dict with results
    """
    print(f"\n  Single-point: {name}")
    
    if supercell_size is None:
        supercell_size = _infer_supercell_size(atoms)
    
    # Set initial magnetic moments according to selected magnetic model.
    atoms = set_initial_magnetic_moments(atoms, mode=mag_config)
    print(f"    structure_id: {structure_id or name}")
    if tags:
        print(f"    tags: {tags}")
    print(f"    supercell_size: {tuple(supercell_size)}")
    print(f"    magnetic: {mag_config}")
    try:
        print(f"    initial net magmom: {float(np.sum(atoms.get_initial_magnetic_moments())):.3f}")
    except Exception:
        pass
    
    if not GPAW_AVAILABLE:
        print("    WARNING: GPAW not available, returning dummy results")
        return {
            'name': name,
            'status': 'skipped',
            'reason': 'GPAW not available',
            'energy': None,
            'forces': None,
            'stress': None,
            'used_gpu': False,
            'structure_id': structure_id,
            'tags': tags,
        }
    
    start_time = time.time()
    checkpoint_base = CHECKPOINT_DIR / f"{name}.gpw"
    checkpoint_file = checkpoint_base
    restarted_from_checkpoint = False

    used_gpu = bool(gpu)
    if used_gpu and not gpaw_gpu_supported():
        raise RuntimeError(
            "GPU requested for Tier B, but GPAW GPU backend is unavailable "
            "(real CuPy backend not detected)."
        )
    reason = None
    try:
        calc = None
        restart_source = None
        if use_restart:
            matches = sorted(CHECKPOINT_DIR.glob(f"{name}*.gpw"), key=lambda p: p.stat().st_mtime, reverse=True)
            if matches:
                restart_source = matches[0]
        if restart_source is not None:
            print(f"    Restarting from checkpoint: {restart_source}")
            try:
                restart_kwargs = _restart_kwargs_from_overrides(scf_overrides)
                calc = GPAW(
                    str(restart_source),
                    txt=str(sp_log_file) if sp_log_file else str(LOGS_DIR / 'gpaw_tierb_sp.out'),
                    **restart_kwargs
                )
                restarted_from_checkpoint = True
            except Exception as e:
                print(f"    WARNING: Restart load failed, using fresh start: {e!r}")
                calc = None

        if calc is None:
            calc = create_gpaw_calculator_tierb(
                'single_point', supercell_size, gpu=used_gpu,
                scf_overrides=scf_overrides, mag_config=mag_config,
                txt_path=sp_log_file
            )
        atoms.calc = calc

        # Calculate energy, forces, stress
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress()

        status = 'completed'
        max_force = abs(forces).max()

        print(f"    Energy: {energy:.6f} eV")
        print(f"    Max force: {max_force:.6f} eV/Ang")
        if restarted_from_checkpoint:
            print("    Reused existing checkpoint (no rewrite).")
        else:
            checkpoint_file = checkpoint_base if not checkpoint_base.exists() else _versioned_path(checkpoint_base, run_tag or _run_tag())
            try:
                atoms.calc.write(str(checkpoint_file), mode='all')
                print(f"    Saved checkpoint: {checkpoint_file}")
            except TypeError:
                atoms.calc.write(str(checkpoint_file))
                print(f"    Saved checkpoint: {checkpoint_file}")
            except Exception as e:
                print(f"    WARNING: Failed to save checkpoint: {e!r}")

    except Exception as e:
        print(f"    ERROR: {e!r}")
        energy = None
        forces = None
        stress = None
        max_force = None
        status = 'failed'
        reason = repr(e)
    
    elapsed = time.time() - start_time
    
    return {
        'name': name,
        'status': status,
        'reason': reason,
        'energy': float(energy) if energy is not None else None,
        'forces': forces.tolist() if forces is not None else None,
        'stress': stress.tolist() if stress is not None else None,
        'max_force': float(max_force) if max_force is not None else None,
        'elapsed_time': float(elapsed),
        'n_atoms': len(atoms),
        'used_gpu': used_gpu,
        'restart_file': str(checkpoint_file),
        'restarted_from_checkpoint': restarted_from_checkpoint,
        'structure_id': structure_id,
        'tags': tags,
    }

# =============================================================================
# Short Relaxation
# =============================================================================

def run_short_relaxation(atoms, name, fmax=0.10, max_steps=30, gpu=False,
                         scf_overrides=None, mag_config='none', use_restart=False,
                         run_tag=None, sr_log_file=None,
                         supercell_size=None, structure_id=None, tags=None):
    """
    Run short ionic relaxation (fixed cell).
    
    Args:
        atoms: ASE Atoms object
        name: Identifier for this calculation
        fmax: Maximum force threshold (eV/Ang)
        max_steps: Maximum optimization steps
        gpu: Whether to use GPU
    
    Returns:
        dict with results
    """
    print(f"\n  Short relaxation: {name}")
    
    if supercell_size is None:
        supercell_size = _infer_supercell_size(atoms)
    
    # Set initial magnetic moments according to selected magnetic model.
    atoms = set_initial_magnetic_moments(atoms, mode=mag_config)
    print(f"    structure_id: {structure_id or name}")
    if tags:
        print(f"    tags: {tags}")
    print(f"    supercell_size: {tuple(supercell_size)}")
    print(f"    magnetic: {mag_config}")
    try:
        print(f"    initial net magmom: {float(np.sum(atoms.get_initial_magnetic_moments())):.3f}")
    except Exception:
        pass
    
    if not GPAW_AVAILABLE:
        print("    WARNING: GPAW not available, returning dummy results")
        return {
            'name': name,
            'status': 'skipped',
            'reason': 'GPAW not available',
            'used_gpu': False,
            'structure_id': structure_id,
            'tags': tags,
        }
    
    start_time = time.time()
    checkpoint_base = CHECKPOINT_DIR / f"{name}.gpw"
    checkpoint_file = checkpoint_base
    # Allow short-relax A1b to reuse A1 single-point geometry when present.
    fallback_sp_traj = None
    if name.startswith('tierb_short_relax_'):
        sp_name = name.replace('tierb_short_relax_', 'tierb_single_point_', 1)
        sp_matches = sorted(TRAJECTORY_DIR.glob(f"{sp_name}*.traj"), key=lambda p: p.stat().st_mtime, reverse=True)
        if sp_matches:
            fallback_sp_traj = sp_matches[0]
    restarted_from_checkpoint = False

    used_gpu = bool(gpu)
    if used_gpu and not gpaw_gpu_supported():
        raise RuntimeError(
            "GPU requested for Tier B, but GPAW GPU backend is unavailable "
            "(real CuPy backend not detected)."
        )
    # Initial state
    try:
        if use_restart and fallback_sp_traj is not None and fallback_sp_traj.exists():
            try:
                atoms_from_traj = read(str(fallback_sp_traj))
                atoms.set_positions(atoms_from_traj.get_positions())
                print(f"    Reused positions from: {fallback_sp_traj}")
            except Exception as e_traj:
                print(f"    WARNING: Failed to read fallback trajectory: {e_traj!r}")

        calc = None
        restart_source = None
        if use_restart:
            sr_matches = sorted(CHECKPOINT_DIR.glob(f"{name}*.gpw"), key=lambda p: p.stat().st_mtime, reverse=True)
            if sr_matches:
                restart_source = sr_matches[0]

        if restart_source is not None:
            print(f"    Restarting from checkpoint: {restart_source}")
            try:
                calc = GPAW(
                    str(restart_source),
                    txt=str(sr_log_file) if sr_log_file else str(LOGS_DIR / 'gpaw_tierb_sr.out')
                )
                restarted_from_checkpoint = True
            except Exception as e:
                print(f"    WARNING: Restart load failed, using fresh start: {e!r}")
                print("             Short relaxation will continue with a fresh calculator.")
                calc = None

        if calc is None:
            calc = create_gpaw_calculator_tierb(
                'short_relax', supercell_size, gpu=used_gpu,
                scf_overrides=scf_overrides, mag_config=mag_config,
                txt_path=sr_log_file
            )
        atoms.calc = calc
        e_initial = atoms.get_potential_energy()
        f_initial = atoms.get_forces()
        print(f"    Initial energy: {e_initial:.6f} eV")
        print(f"    Initial max force: {abs(f_initial).max():.6f} eV/Ang")
    except Exception as e:
        print(f"    ERROR getting initial state: {e!r}")
        return {
            'name': name,
            'status': 'failed',
            'reason': repr(e),
            'used_gpu': used_gpu,
            'structure_id': structure_id,
            'tags': tags,
        }
    
    # Run relaxation
    print(f"    Running relaxation (fmax={fmax}, max_steps={max_steps})...")
    
    try:
        optimizer_log = (LOGS_DIR / f'{name}_sr.log') if run_tag is None else (LOGS_DIR / f'{name}_sr__{run_tag}.log')
        optimizer = LBFGS(atoms, logfile=str(optimizer_log))
        optimizer.run(fmax=fmax, steps=max_steps)
        status = 'completed'
    except Exception as e:
        print(f"    ERROR during relaxation: {e!r}")
        status = 'failed'
    
    # Final state
    try:
        e_final = atoms.get_potential_energy()
        f_final = atoms.get_forces()
        stress = atoms.get_stress()
        
        print(f"    Final energy: {e_final:.6f} eV")
        print(f"    Final max force: {abs(f_final).max():.6f} eV/Ang")
        print(f"    Energy change: {e_final - e_initial:.6f} eV")
        if restarted_from_checkpoint:
            print("    Reused existing checkpoint (no rewrite).")
        else:
            checkpoint_file = checkpoint_base if not checkpoint_base.exists() else _versioned_path(checkpoint_base, run_tag or _run_tag())
            try:
                atoms.calc.write(str(checkpoint_file), mode='all')
                print(f"    Saved checkpoint: {checkpoint_file}")
            except TypeError:
                atoms.calc.write(str(checkpoint_file))
                print(f"    Saved checkpoint: {checkpoint_file}")
            except Exception as e:
                print(f"    WARNING: Failed to save checkpoint: {e!r}")
        
    except Exception as e:
        print(f"    ERROR getting final state: {e!r}")
        e_final = None
        f_final = None
        stress = None
    
    elapsed = time.time() - start_time
    
    return {
        'name': name,
        'status': status,
        'initial_energy': float(e_initial) if e_initial is not None else None,
        'final_energy': float(e_final) if e_final is not None else None,
        'energy_change': float(e_final - e_initial) if e_final is not None else None,
        'forces': f_final.tolist() if f_final is not None else None,
        'stress': stress.tolist() if stress is not None else None,
        'max_force': float(abs(f_final).max()) if f_final is not None else None,
        'elapsed_time': float(elapsed),
        'n_atoms': len(atoms),
        'used_gpu': used_gpu,
        'restart_file': str(checkpoint_file),
        'restarted_from_checkpoint': restarted_from_checkpoint,
        'structure_id': structure_id,
        'tags': tags,
    }


def reset_tier_b_state():
    """Remove Tier-B bookkeeping and outputs so runs can restart from zero."""
    removed = []
    targets = [
        RESULTS_DIR / 'tier_b_results.json',
        RESULTS_DIR / 'dft_budget.json',
    ]
    for path in targets:
        if path.exists():
            path.unlink()
            removed.append(str(path))

    for path in TRAJECTORY_DIR.glob('tierb_*'):
        if path.is_file():
            path.unlink()
            removed.append(str(path))
    for path in RESULTS_DIR.glob('tier_b_results__*.json'):
        if path.is_file():
            path.unlink()
            removed.append(str(path))
    for path in LOGS_DIR.glob('gpaw_tierb_*'):
        if path.is_file():
            path.unlink()
            removed.append(str(path))
    if CHECKPOINT_DIR.exists():
        for path in CHECKPOINT_DIR.glob('tierb_*'):
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    print("\nReset Tier-B state complete.")
    print(f"  Removed {len(removed)} files.")
    return removed

# =============================================================================
# Select Structures for Calculation
# =============================================================================

def select_structures_for_tierb(calc_type='all', tags=None, structure_ids=None,
                                max_structures=None, compositions=None):
    """
    Select structures for Tier B calculations.

    GaN mode:
      - reads dft/structures/structure_info.json
      - selects by tags and/or structure_ids

    Legacy oxide mode:
      - if structure_info.json contains 'compositions', select by composition buckets

    Returns:
        List of (structure_file, meta, calc_type) tuples
    """
    info_file = STRUCTURES_DIR / 'structure_info.json'
    if not info_file.exists():
        print(f"ERROR: Structure info not found: {info_file}")
        print("Please run dft/scripts/structure_generation.py first.")
        return []

    with open(info_file, 'r', encoding='utf-8') as f:
        structure_info = json.load(f)

    # Legacy path (kept so old runs don't explode).
    if 'compositions' in structure_info:
        selections = []
        allowed_compositions = None
        if compositions:
            allowed_compositions = {round(float(x), 6) for x in compositions}
        for x_str, comp_data in structure_info['compositions'].items():
            x = float(x_str)
            if allowed_compositions is not None and round(x, 6) not in allowed_compositions:
                continue
            available = comp_data.get('structures', [])
            for s in available:
                selections.append((s['filepath'], s, 'single_point'))
        if calc_type in ('single_point', 'short_relax'):
            selections = [s for s in selections if s[2] == calc_type]
        if max_structures is not None:
            selections = selections[:max_structures]
        return selections

    # GaN (tag-based) path.
    wanted_ids = set(structure_ids or [])
    wanted_tags = set([t.strip() for t in (tags or []) if t.strip()])

    structures = structure_info.get('structures', [])
    filtered = []
    for s in structures:
        sid = s.get('id') or Path(s.get('filepath', '')).stem
        stags = set(s.get('tags') or [])
        if wanted_ids and sid not in wanted_ids:
            continue
        if wanted_tags and stags.isdisjoint(wanted_tags):
            continue
        filtered.append(s)

    # If the user requested explicit IDs that are not present in structure_info.json,
    # fall back to locating CIF files directly under dft/structures/.
    # This prevents "Selected 0 structures" when structure_info is stale.
    if wanted_ids:
        found_ids = set((s.get('id') or Path(s.get('filepath', '')).stem) for s in filtered)
        missing = [sid for sid in wanted_ids if sid not in found_ids]
        for sid in missing:
            cif_path = STRUCTURES_DIR / f"{sid}.cif"
            if cif_path.exists():
                filtered.append(
                    {
                        "id": sid,
                        "filepath": str(cif_path),
                        "filename": cif_path.name,
                        "tags": None,
                        "supercell_size": None,
                        "notes": "Fallback selection: not present in structure_info.json at selection time.",
                    }
                )
        # Keep stable order: structure_info order first, then fallbacks in requested order.
        if missing:
            print("NOTE: Some requested --structure-ids were not found in structure_info.json.")
            print("      Falling back to direct CIF lookup in dft/structures/:")
            for sid in missing:
                if (STRUCTURES_DIR / f"{sid}.cif").exists():
                    print(f"        + {sid}.cif")

    # stable order: as listed in structure_info.json
    if max_structures is not None:
        filtered = filtered[:max_structures]

    selections = []
    if calc_type == 'all':
        # "all" means SP first, SR second for each selected structure.
        for s in filtered:
            selections.append((s['filepath'], s, 'single_point'))
            selections.append((s['filepath'], s, 'short_relax'))
    else:
        for s in filtered:
            selections.append((s['filepath'], s, calc_type))

    return selections

# =============================================================================
# Main Workflow
# =============================================================================

def run_tier_b_calculations(dry_run=False, gpu=False, calc_type='all',
                            tags=None, structure_ids=None,
                            compositions=None, max_structures=None,
                            maxiter=None, conv_energy=None, conv_density=None,
                            scf_conv_forces=None, conv_eigenstates=None,
                            fmax=None, relax_steps=None,
                            mag_config='none', use_restart=False,
                            reset_state=False):
    """
    Run all Tier B calculations.
    
    Args:
        dry_run: If True, just print what would be done
        gpu: Whether to use GPU acceleration
    """
    print("\n" + "="*60)
    print("TIER B DFT CALCULATIONS")
    print("Single-Point and Short Relaxations")
    print("="*60)
    print("Run arguments:")
    print(f"  gpu: {gpu}")
    print(f"  dry_run: {dry_run}")
    print(f"  calc_type: {calc_type}")
    if tags:
        print(f"  tags filter: {tags}")
    if structure_ids:
        print(f"  structure_ids: {structure_ids}")
    print(f"  max_structures: {max_structures}")
    print(f"  mag_config: {mag_config}")
    print(f"  use_restart: {use_restart}")
    
    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if reset_state:
        reset_tier_b_state()

    run_tag = _run_tag()
    sp_log_file = LOGS_DIR / f'gpaw_tierb_sp__{run_tag}.out'
    sr_log_file = LOGS_DIR / f'gpaw_tierb_sr__{run_tag}.out'
    print(f"  run_tag: {run_tag}")
    print(f"  sp_log: {sp_log_file}")
    print(f"  sr_log: {sr_log_file}")
    
    # Initialize budget tracker
    budget_tracker = DFTBudgetTracker(RESULTS_DIR / 'dft_budget.json')
    budget_tracker.print_status()

    effective_gpu = bool(gpu)
    if effective_gpu:
        print("\nGPU mode requested for Tier B.")
        if not gpaw_gpu_supported():
            raise RuntimeError(
                "Tier B requested with --gpu, but real GPAW GPU backend is not available."
            )

    scf_overrides = {
        'maxiter': maxiter,
        'energy': conv_energy,
        'density': conv_density,
        'forces': scf_conv_forces,
        'eigenstates': conv_eigenstates,
    }
    if any(value is not None for value in scf_overrides.values()):
        print("\nUsing user SCF overrides:")
        for key, value in scf_overrides.items():
            if value is not None:
                print(f"  {key}: {value}")

    if fmax is not None or relax_steps is not None:
        print("\nUsing user ionic-relax overrides:")
        if fmax is not None:
            print(f"  fmax: {fmax}")
        if relax_steps is not None:
            print(f"  relax_steps: {relax_steps}")
    
    # Select structures
    selections = select_structures_for_tierb(
        calc_type=calc_type,
        tags=tags,
        structure_ids=structure_ids,
        compositions=compositions,  # legacy
        max_structures=max_structures,
    )
    print(f"\nSelected {len(selections)} structures for Tier B calculations")

    if dry_run:
        for i, (struct_file, meta, sel_type) in enumerate(selections):
            structure_id = (meta or {}).get('id') or Path(struct_file).stem
            print(f"\n[{i+1}/{len(selections)}] DRY RUN: {Path(struct_file).name}")
            print(f"  Calculation type: {sel_type}")
            print(f"  Structure ID: {structure_id}")
            if (meta or {}).get('tags'):
                print(f"  Tags: {(meta or {}).get('tags')}")
            try:
                atoms = read(struct_file)
                print(f"  Loaded: {len(atoms)} atoms")
            except Exception as e:
                print(f"  ERROR reading structure: {e!r}")
        print("\nDRY RUN complete. No results written.")
        return {"dry_run": True, "n_selected": len(selections)}
    
    # Storage for results for this run
    run_results = {
        'tier': 'B',
        'run_tag': run_tag,
        'started': datetime.now().isoformat(),
        'sp_log_file': str(sp_log_file),
        'sr_log_file': str(sr_log_file),
        'single_point': [],
        'short_relax': [],
    }

    # Cumulative storage (never discard previous entries).
    cumulative_file = RESULTS_DIR / 'tier_b_results.json'
    if cumulative_file.exists():
        try:
            with open(cumulative_file, 'r') as f:
                cumulative_results = json.load(f)
        except Exception:
            cumulative_results = {}
    else:
        cumulative_results = {}
    cumulative_results.setdefault('tier', 'B')
    cumulative_results.setdefault('single_point', [])
    cumulative_results.setdefault('short_relax', [])
    cumulative_results['last_run_tag'] = run_tag
    cumulative_results['last_updated'] = datetime.now().isoformat()
    
    # Process each structure
    for i, (struct_file, meta, calc_type) in enumerate(selections):
        print(f"\n[{i+1}/{len(selections)}] Processing: {Path(struct_file).name}")
        print(f"  Calculation type: {calc_type}")
        structure_id = (meta or {}).get('id') or Path(struct_file).stem
        struct_tags = (meta or {}).get('tags')
        supercell_size = (meta or {}).get('supercell_size')
        if structure_id:
            print(f"  Structure ID: {structure_id}")
        if struct_tags:
            print(f"  Tags: {struct_tags}")
        
        # Check budget
        budget_type = 'tier_b_sp' if calc_type == 'single_point' else 'tier_b_relax'
        if not budget_tracker.can_run(budget_type):
            print(f"  WARNING: {budget_type} budget exhausted, skipping.")
            continue
        
        # Read structure
        try:
            atoms = read(struct_file)
            print(f"  Loaded: {len(atoms)} atoms")
        except Exception as e:
            print(f"  ERROR reading structure: {e}")
            continue
        
        if dry_run:
            print("  DRY RUN: Skipping DFT calculation")
            continue
        
        # Run calculation
        name = f"tierb_{calc_type}_{Path(struct_file).stem}"
        
        try:
            if calc_type == 'single_point':
                results = run_single_point(
                    atoms, name, gpu=effective_gpu, scf_overrides=scf_overrides,
                    mag_config=mag_config, use_restart=use_restart,
                    run_tag=run_tag, sp_log_file=sp_log_file,
                    supercell_size=supercell_size, structure_id=structure_id, tags=struct_tags
                )
                run_results['single_point'].append(results)
            else:
                short_relax_fmax = float(fmax) if fmax is not None else float(TIER_B_CONFIG['short_relax']['fmax'])
                short_relax_steps = int(relax_steps) if relax_steps is not None else int(TIER_B_CONFIG['short_relax']['max_steps'])
                results = run_short_relaxation(
                    atoms, name,
                    fmax=short_relax_fmax,
                    max_steps=short_relax_steps,
                    gpu=effective_gpu,
                    scf_overrides=scf_overrides,
                    mag_config=mag_config,
                    use_restart=use_restart,
                    run_tag=run_tag,
                    sr_log_file=sr_log_file,
                    supercell_size=supercell_size, structure_id=structure_id, tags=struct_tags
                )
                run_results['short_relax'].append(results)
        except Exception as e:
            print(f"  ERROR: Unhandled exception in {calc_type}: {e!r}")
            results = {
                'name': name,
                'status': 'failed',
                'reason': f"Unhandled exception: {e!r}",
                'used_gpu': bool(effective_gpu),
                'structure_id': structure_id,
                'tags': struct_tags,
            }
            if calc_type == 'single_point':
                run_results['single_point'].append(results)
            else:
                run_results['short_relax'].append(results)
        
        # Save results
        results['structure_file'] = struct_file
        if structure_id is not None:
            results.setdefault('structure_id', structure_id)
        if struct_tags is not None:
            results.setdefault('tags', struct_tags)
        
        # Save trajectory snapshot without calculator attached
        atoms_to_save = atoms.copy()
        atoms_to_save.calc = None
        traj_base = TRAJECTORY_DIR / f"{name}.traj"
        traj_file = traj_base if not traj_base.exists() else _versioned_path(traj_base, run_tag)
        try:
            write(traj_file, atoms_to_save)
            results['trajectory_file'] = str(traj_file)
        except Exception as e:
            print(f"  WARNING: Failed to write trajectory: {e!r}")
            results['trajectory_file'] = None
        
        # Record successful calculations in budget tracker.
        if results.get('status') == 'completed':
            if _already_counted_calculation(budget_tracker, budget_type, name):
                print("  NOTE: Calculation already counted previously; skipping duplicate budget increment.")
            else:
                budget_tracker.record_calculation(
                    calc_type=budget_type,
                    structure_id=name,
                    composition=(meta or {}).get('composition_x'),
                    energy=results.get('final_energy') or results.get('energy'),
                    forces_max=results.get('max_force'),
                    status=results['status']
                )
        else:
            print("  NOTE: Not counting failed/skipped run against Tier B budget.")

        # Append to cumulative (never replacing old entries).
        if calc_type == 'single_point':
            cumulative_results['single_point'].append(results)
        else:
            cumulative_results['short_relax'].append(results)

        # Save intermediate cumulative + run snapshot.
        cumulative_results['last_updated'] = datetime.now().isoformat()
        run_file = RESULTS_DIR / f'tier_b_results__{run_tag}.json'
        _atomic_json_dump(cumulative_file, cumulative_results)
        _atomic_json_dump(run_file, run_results)
    
    # Finalize
    run_results['completed'] = datetime.now().isoformat()
    cumulative_results['last_updated'] = datetime.now().isoformat()

    run_file = RESULTS_DIR / f'tier_b_results__{run_tag}.json'
    _atomic_json_dump(cumulative_file, cumulative_results)
    _atomic_json_dump(run_file, run_results)
    print(f"\nResults saved (cumulative): {cumulative_file}")
    print(f"Results saved (this run): {run_file}")
    
    # Print budget status
    budget_tracker.print_status()
    
    # Summary
    n_sp = len([r for r in run_results['single_point'] if r['status'] == 'completed'])
    n_sr = len([r for r in run_results['short_relax'] if r['status'] == 'completed'])
    print(f"\nTier B Summary:")
    print(f"  Single-point completed: {n_sp}")
    print(f"  Short relax completed: {n_sr}")
    print(f"  Total completed: {n_sp + n_sr}")
    
    return run_results

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Tier B DFT calculations')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print what would be done without running DFT')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration')
    parser.add_argument('--calc-type', choices=['single_point', 'short_relax', 'all'],
                       default='all', help='Type of calculation to run')
    parser.add_argument('--tags', nargs='+', default=None,
                       help='Select structures by tags (GaN workflow), e.g. --tags bulk defect')
    parser.add_argument('--structure-ids', nargs='+', default=None,
                       help='Select specific structure IDs from structure_info.json')
    parser.add_argument('--composition', type=float, nargs='+',
                       help='Legacy: composition filter for older oxide workflow (ignored for GaN)')
    parser.add_argument('--max-structures', type=int, default=None,
                       help='Run only the first N selected structures')
    parser.add_argument('--maxiter', type=int, default=None,
                       help='Override SCF maximum iterations')
    parser.add_argument('--conv-energy', type=float, default=None,
                       help='Override SCF convergence threshold for total energy')
    parser.add_argument('--conv-density', type=float, default=None,
                       help='Override SCF convergence threshold for density')
    parser.add_argument('--scf-conv-forces', type=float, default=None,
                       help='Override SCF convergence threshold for forces (rare; usually leave unset)')
    parser.add_argument('--conv-eigenstates', type=float, default=None,
                       help='Override SCF convergence threshold for eigenstates')
    parser.add_argument('--fmax', type=float, default=None,
                       help='Override ionic relaxation force threshold (eV/Ang) for short_relax')
    parser.add_argument('--relax-steps', type=int, default=None,
                       help='Override ionic relaxation max steps for short_relax')
    parser.add_argument('--mag-config', choices=['none', 'fm', 'afm'], default='none',
                       help='Magnetic configuration (GaN default: none)')
    parser.add_argument('--use-restart', action='store_true',
                       help='Continue from existing Tier-B .gpw checkpoints when available')
    parser.add_argument('--reset-state', action='store_true',
                       help='Delete Tier-B outputs/bookkeeping and restart from zero')
    
    args = parser.parse_args()
    print("\nParsed CLI arguments:")
    print(f"  dry_run={args.dry_run}")
    print(f"  gpu={args.gpu}")
    print(f"  calc_type={args.calc_type}")
    print(f"  tags={args.tags}")
    print(f"  structure_ids={args.structure_ids}")
    print(f"  composition={args.composition}")
    print(f"  max_structures={args.max_structures}")
    print(f"  maxiter={args.maxiter}")
    print(f"  conv_energy={args.conv_energy}")
    print(f"  conv_density={args.conv_density}")
    print(f"  scf_conv_forces={args.scf_conv_forces}")
    print(f"  conv_eigenstates={args.conv_eigenstates}")
    print(f"  fmax={args.fmax}")
    print(f"  relax_steps={args.relax_steps}")
    print(f"  mag_config={args.mag_config}")
    print(f"  use_restart={args.use_restart}")
    print(f"  reset_state={args.reset_state}")
    
    run_tier_b_calculations(
        dry_run=args.dry_run,
        gpu=args.gpu,
        calc_type=args.calc_type,
        tags=args.tags,
        structure_ids=args.structure_ids,
        compositions=args.composition,
        max_structures=args.max_structures,
        maxiter=args.maxiter,
        conv_energy=args.conv_energy,
        conv_density=args.conv_density,
        scf_conv_forces=args.scf_conv_forces,
        conv_eigenstates=args.conv_eigenstates,
        fmax=args.fmax,
        relax_steps=args.relax_steps,
        mag_config=args.mag_config,
        use_restart=args.use_restart,
        reset_state=args.reset_state
    )
