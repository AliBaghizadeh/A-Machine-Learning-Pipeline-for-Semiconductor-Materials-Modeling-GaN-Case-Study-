"""
GPAW DFT Parameters for Wurtzite GaN (Minimal-Load)
===================================================

This project has pivoted from Lu/Sc/Fe-oxide to wurtzite GaN for defect and
dislocation-scale modeling (HR-TEM-motivated). The default settings below are
intentionally *lightweight* for a pipeline demo, not publication-grade.

Notes:
- Exchange-correlation: GGA-PBE
- No DFT+U (GaN is non-magnetic in bulk; do not force spin polarization)
- Use modest k-point density for the primitive cell and scale down for supercells
- GPU execution in this repo uses GPAW's "new" GPU path (PW mode) from the tier scripts
"""

import os

# =============================================================================
# Core GPAW Parameters
# =============================================================================

SYSTEM_NAME = "GaN"

DEFAULT_BASE_KPTS = (6, 6, 4)  # primitive GaN: moderate k-mesh; scaled for supercells

GPAW_PARAMS = {
    # Calculation mode (CPU default). Tier scripts overwrite to PW on GPU.
    'mode': 'fd',

    # XC functional
    'xc': 'PBE',
    
    # Brillouin zone sampling (scaled for supercells via get_kpts_for_supercell)
    'kpts': DEFAULT_BASE_KPTS,
    
    # Real-space grid spacing (used for FD/LCAO modes on CPU)
    # GPU code-path overwrites mode to PW and removes 'h' in tier scripts.
    'h': 0.18,
    
    # SCF convergence
    'maxiter': 60,
    'convergence': {
        # "Demo-grade" defaults; can be overridden from CLI.
        'energy': 1e-4,
        'density': 1e-3,
        'eigenstates': 1e-4,
    },
    
    # Smearing: GaN is a semiconductor; keep small FD smearing for robustness.
    'occupations': 'smearing',
    'sigma': 0.05,
    
    # Bulk GaN is non-magnetic.
    'spinpol': False,
    
    # Mixer for SCF
    'mixer': {
        'method': 'Mixer',
        'beta': 0.1,
        'nmaxold': 5,
    },
    
    # Output
    'txt': 'gpaw.out',
    'verbose': 1,
}

# =============================================================================
# Hubbard U Parameters (PBE+U)
# =============================================================================

HUBBARD_U = {
    # No +U for GaN in this minimal pipeline.
}

# Setup for GPAW with +U
def get_gpaw_setup_params():
    """Return GPAW setup strings for DFT+U (GPAW-25 style)."""
    setups = {}
    for element, u_value in HUBBARD_U.items():
        if u_value > 0:
            # GPAW expects setup type strings, e.g. "paw:d,5.0" for Fe 3d.
            setups[element] = f"paw:d,{float(u_value)}"
    return setups

# =============================================================================
# GPU Configuration
# =============================================================================

GPU_CONFIG = {
    'use_gpu': True,
    'gpus': [0],              # Use GPU 0
    'gpu_memory_limit': '14GB',  # Leave 2GB for system
    # Tier scripts will use PW mode on GPU for stability/performance.
    'gpu_mode': 'pw',
}

def setup_gpu_environment():
    """Set environment variables for GPU acceleration"""
    os.environ['GPAW_NEW'] = '1'
    os.environ['GPAW_USE_GPUS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # For parallel GPU calculations
    os.environ['GPAW_GPU_MAX_MEMORY'] = '14000000000'  # 14 GB in bytes
    
    print("GPU environment configured:")
    print(f"  GPAW_NEW = {os.environ.get('GPAW_NEW')}")
    print(f"  GPAW_USE_GPUS = {os.environ.get('GPAW_USE_GPUS')}")
    print(f"  CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# =============================================================================
# Magnetic Configuration
# =============================================================================

MAGNETIC_MOMENTS = {
    'Ga': 0.0,
    'N': 0.0,
}

def set_initial_magnetic_moments(atoms, mode='none', moment=None):
    """
    Set initial magnetic moments on atoms.

    Args:
        atoms: ASE Atoms object
        mode:
          - 'none'/'nonmag': set all initial moments to 0.0
          - 'fm': set all atoms of a given species to +moment (fallback 0.0)
          - 'afm': alternate +moment/-moment for atoms of the first magnetic species
        moment: absolute moment value for fm/afm (defaults to 0.0)
    """
    m = float(0.0 if moment is None else moment)

    mode = (mode or 'none').lower()
    if mode in ('none', 'nonmag', 'nm'):
        atoms.set_initial_magnetic_moments([0.0] * len(atoms))
        return atoms

    if mode == 'fm':
        atoms.set_initial_magnetic_moments([m] * len(atoms))
        return atoms

    if mode == 'afm':
        magmoms = []
        counter = 0
        for _atom in atoms:
            sign = 1.0 if (counter % 2 == 0) else -1.0
            magmoms.append(sign * m)
            counter += 1
        atoms.set_initial_magnetic_moments(magmoms)
        return atoms

    raise ValueError(f"Unknown magmom mode: {mode!r}")

# =============================================================================
# K-point Scaling for Supercells
# =============================================================================

def get_kpts_for_supercell(supercell_size, base_kpts=(4, 4, 2)):
    """
    Scale k-points for supercell calculations.
    Maintain similar k-point density.
    
    Args:
        supercell_size: tuple (nx, ny, nz) - supercell dimensions
        base_kpts: tuple (kx, ky, kz) - base k-point mesh
    
    Returns:
        tuple: scaled k-point mesh
    """
    kx = max(1, int(base_kpts[0]) // int(supercell_size[0]))
    ky = max(1, int(base_kpts[1]) // int(supercell_size[1]))
    kz = max(1, int(base_kpts[2]) // int(supercell_size[2]))
    return (kx, ky, kz)

# =============================================================================
# Relaxation Parameters
# =============================================================================

RELAXATION_PARAMS = {
    # Tier A: Full relaxation
    'tier_a': {
        'fmax': 0.05,          # Max force (eV/Ang)
        'max_steps': 200,      # Max optimization steps
        'cell_relax': True,    # Relax cell + ions
        'optimizer': 'BFGS',   # Optimization algorithm
    },
    
    # Tier B: Short relaxation
    'tier_b_relax': {
        'fmax': 0.10,          # Less strict convergence
        'max_steps': 50,       # Fewer steps
        'cell_relax': False,   # Only ionic relaxation
        'optimizer': 'LBFGS',  # Faster optimizer
    },
    
    # Tier B: Single-point
    'tier_b_sp': {
        'single_point': True,  # No relaxation
    },
}

# =============================================================================
# Material-Specific Parameters
# =============================================================================

# Lattice parameters from CIF files (for reference only)
LATTICE_PARAMS = {
    'GaN': {
        'a': 3.2163,
        'c': 5.2400,
        'spacegroup': 'P63mc',
        'spacegroup_number': 186,
    }
}

# =============================================================================
# Utility Functions
# =============================================================================

def get_gpaw_calculator(task_type='tier_a', supercell_size=(1, 1, 1), gpu=True):
    """
    Create a GPAW calculator with appropriate parameters.

    Args:
        task_type: 'tier_a', 'tier_b_relax', or 'tier_b_sp'
        supercell_size: tuple (nx, ny, nz) for k-point scaling
        gpu: whether to use GPU
    
    Returns:
        GPAW calculator object
    """
    # NOTE: This helper is kept for compatibility. The Tier-A/Tier-B scripts in
    # this repo provide the real, GPU-stable configuration (PW mode).
    from gpaw import GPAW

    params = GPAW_PARAMS.copy()

    # Scale k-points for supercell
    params['kpts'] = get_kpts_for_supercell(supercell_size, base_kpts=tuple(params.get('kpts', DEFAULT_BASE_KPTS)))

    # Add Hubbard U setups (empty for GaN)
    params['setups'] = get_gpaw_setup_params()

    # Configure GPU (best-effort; tier scripts override to PW for stability)
    params['parallel'] = {'gpu': bool(gpu)}

    # Adjust for task type
    if task_type == 'tier_b_sp':
        params['maxiter'] = min(int(params.get('maxiter', 60)), 60)

    return GPAW(**params)

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Test the configuration
    print("=" * 60)
    print("GPAW Configuration for GaN (Minimal-Load)")
    print("=" * 60)
    
    print("\nCore DFT Parameters:")
    for key, value in GPAW_PARAMS.items():
        print(f"  {key}: {value}")
    
    print("\nHubbard U Parameters (eV):")
    for element, u in HUBBARD_U.items():
        print(f"  {element}: {u}")
    
    print("\nGPU Configuration:")
    for key, value in GPU_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nRelaxation Parameters:")
    for tier, params in RELAXATION_PARAMS.items():
        print(f"  {tier}: {params}")
    
    # Setup GPU environment
    print("\n" + "=" * 60)
    setup_gpu_environment()
    print("=" * 60)
