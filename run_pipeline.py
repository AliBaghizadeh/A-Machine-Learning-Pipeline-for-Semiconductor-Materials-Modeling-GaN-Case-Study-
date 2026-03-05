"""
Main Pipeline for GaN MLIP Project
==================================

Orchestrates the complete DFT + MLIP workflow:
1) Generate GaN bulk/defect structures
2) Run lightweight GPAW DFT (Tier B primarily) to generate energies/forces
3) Extract extxyz datasets
4) Train MACE MLIP on GPU
5) Analyze/validate and iterate
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime

# Pipeline Stages
STAGES = {
    'setup': {
        'description': 'Environment setup and validation',
        'script': None,
    },
    'structures': {
        'description': 'Generate structures for DFT',
        'script': 'dft/scripts/structure_generation.py',
    },
    'dft_tier_a': {
        'description': 'Run Tier A DFT calculations (full relaxation)',
        'script': 'dft/scripts/tier_a_relaxation.py',
    },
    'dft_tier_b': {
        'description': 'Run Tier B DFT calculations (single-point/short relax)',
        'script': 'dft/scripts/tier_b_calculations.py',
    },
    'extract_data': {
        'description': 'Extract DFT data for MLIP training',
        'script': 'dft/scripts/extract_dft_data.py',
    },
    'train_mlip': {
        'description': 'Train MACE MLIP model',
        'script': 'mlip/scripts/train_mlip.py',
    },
    'active_learning': {
        'description': 'Run active learning loop',
        'script': 'mlip/scripts/active_learning.py',
    },
    'analysis': {
        'description': 'Structural analysis',
        'script': 'analysis/scripts/structural_analysis.py',
    },
    'validation': {
        'description': 'Validate MLIP against DFT',
        'script': 'analysis/scripts/compare_dft_mlip.py',
    },
}

# Helper Functions
def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(title)
    print("="*60)

def run_command(cmd, dry_run=False):
    """Run a shell command."""
    import subprocess
    
    if dry_run:
        print(f"DRY RUN: {cmd}")
        return 0
    
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode


def get_mpi_prefix(gpu=False):
    """Return MPI launch prefix if configured."""
    raw = os.environ.get('MLIP_MPI_PROCS') or os.environ.get('GPAW_MPI_PROCS')
    if not raw:
        # Project policy: avoid single-CPU runs. Default CPU MPI width to 12.
        if gpu:
            return ""
        raw = "12"
    try:
        nprocs = int(raw)
    except ValueError:
        print(f"WARNING: Invalid MLIP_MPI_PROCS={raw!r}; ignoring MPI setting.")
        return ""
    if nprocs not in (8, 12):
        print(f"WARNING: MLIP_MPI_PROCS={nprocs} not in {{8,12}}; using 12.")
        nprocs = 12
    if gpu and os.environ.get('MLIP_MPI_ALLOW_GPU_MPI', '0') != '1':
        print("NOTE: Ignoring MLIP_MPI_PROCS for GPU run (set MLIP_MPI_ALLOW_GPU_MPI=1 to override).")
        return ""
    launcher = shutil.which('mpiexec') or shutil.which('mpirun')
    if launcher is None:
        print("WARNING: MPI requested but no mpiexec/mpirun found in PATH.")
        return ""
    return f"{launcher} -n {nprocs}"

def check_dependencies():
    """Check that required dependencies are installed."""
    print_header("CHECKING DEPENDENCIES")
    
    dependencies = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'ase': 'ase',
        'torch': 'torch',
        'gpaw': 'gpaw',
    }
    
    missing = []
    
    for name, package in dependencies.items():
        try:
            __import__(name)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("\nAll dependencies installed.")
    return True

def check_gpu():
    """Check GPU availability."""
    print_header("CHECKING GPU")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            # Torch CUDA availability does not guarantee GPAW GPU backend support.
            try:
                from gpaw.gpu import cupy as gpaw_cupy
                if gpaw_cupy.__name__.startswith('gpaw.gpu.cpupy'):
                    print("  GPAW GPU backend: NOT ready (using cpupy fallback)")
                    print("  Tier A/B DFT will run on CPU unless real CuPy is available.")
                else:
                    print("  GPAW GPU backend: ready")
            except Exception:
                print("  GPAW GPU backend: NOT ready")
            return True
        else:
            print("  GPU not available, using CPU")
            return False
    except ImportError:
        print("  PyTorch not installed, cannot check GPU")
        return False

def check_input_files():
    """Check that required input files exist."""
    print_header("CHECKING INPUT FILES")
    
    project_root = Path(__file__).parent
    
    required_files = [
        'cifs/GaN.cif',
        'cifs/GaN_mp-804_conventional_standard.cif',
    ]
    
    missing = []
    
    for file in required_files:
        filepath = project_root / file
        if filepath.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (missing)")
            missing.append(file)
    
    if missing:
        print(f"\nMissing input files: {', '.join(missing)}")
        return False
    
    print("\nAll input files present.")
    return True

# Pipeline Stages
def run_setup(dry_run=False):
    """Run setup checks."""
    print_header("STAGE: SETUP")
    
    print("Checking environment...")
    deps_ok = check_dependencies()
    gpu_ok = check_gpu()
    files_ok = check_input_files()
    
    if deps_ok and files_ok:
        print("\nSetup complete.")
        return True
    else:
        print("\nSetup incomplete. Please resolve issues above.")
        return False

def run_structures(dry_run=False):
    """Generate structures."""
    print_header("STAGE: STRUCTURE GENERATION")
    
    script = STAGES['structures']['script']
    cmd = f"python {script}"
    
    return run_command(cmd, dry_run) == 0

def run_dft_tier_a(dry_run=False, gpu=True, extra_args: str = ""):
    """Run Tier A DFT calculations."""
    print_header("STAGE: DFT TIER A")
    
    script = STAGES['dft_tier_a']['script']
    mpi_prefix = get_mpi_prefix(gpu=gpu)
    cmd = f"{mpi_prefix} python {script} {extra_args}".strip()
    
    if dry_run:
        cmd += " --dry-run"
    
    return run_command(cmd, dry_run) == 0

def run_dft_tier_b(dry_run=False, gpu=False, extra_args: str = ""):
    """Run Tier B DFT calculations."""
    print_header("STAGE: DFT TIER B")
    
    script = STAGES['dft_tier_b']['script']
    mpi_prefix = get_mpi_prefix(gpu=gpu)
    cmd = f"{mpi_prefix} python {script} {extra_args}".strip()
    
    if dry_run:
        cmd += " --dry-run"
    if gpu:
        cmd += " --gpu"
    
    return run_command(cmd, dry_run) == 0

def run_extract_data(dry_run=False):
    """Extract DFT data."""
    print_header("STAGE: EXTRACT DATA")
    
    script = STAGES['extract_data']['script']
    cmd = f"python {script}"
    
    return run_command(cmd, dry_run) == 0

def run_train_mlip(dry_run=False):
    """Train MLIP."""
    print_header("STAGE: TRAIN MLIP")
    
    script = STAGES['train_mlip']['script']
    cmd = f"python {script}"
    
    return run_command(cmd, dry_run) == 0

def run_active_learning(dry_run=False):
    """Run active learning."""
    print_header("STAGE: ACTIVE LEARNING")
    
    script = STAGES['active_learning']['script']
    cmd = f"python {script}"
    
    return run_command(cmd, dry_run) == 0

def run_analysis(dry_run=False):
    """Run structural analysis."""
    print_header("STAGE: ANALYSIS")
    
    script = STAGES['analysis']['script']
    cmd = f"python {script}"
    
    return run_command(cmd, dry_run) == 0

def run_validation(dry_run=False):
    """Run validation."""
    print_header("STAGE: VALIDATION")
    
    script = STAGES['validation']['script']
    cmd = f"python {script}"
    
    return run_command(cmd, dry_run) == 0

# Main Pipeline
def run_pipeline(stages=None, dry_run=False, gpu=True, tier_a_extra: str = "", tier_b_extra: str = ""):
    """
    Run the complete pipeline.
    
    Args:
        stages: List of stages to run (None = all)
        dry_run: If True, print commands without running
        gpu: Use GPU for calculations
    """
    print_header("GaN MLIP PIPELINE")
    print(f"Started: {datetime.now().isoformat()}")
    
    if stages is None:
        stages = list(STAGES.keys())
    
    results = {}
    
    stage_functions = {
        'setup': lambda: run_setup(dry_run),
        'structures': lambda: run_structures(dry_run),
        'dft_tier_a': lambda: run_dft_tier_a(dry_run, gpu, extra_args=tier_a_extra),
        'dft_tier_b': lambda: run_dft_tier_b(dry_run, gpu, extra_args=tier_b_extra),
        'extract_data': lambda: run_extract_data(dry_run),
        'train_mlip': lambda: run_train_mlip(dry_run),
        'active_learning': lambda: run_active_learning(dry_run),
        'analysis': lambda: run_analysis(dry_run),
        'validation': lambda: run_validation(dry_run),
    }
    
    for stage in stages:
        if stage not in STAGES:
            print(f"Unknown stage: {stage}")
            continue
        
        print(f"\nRunning stage: {stage}")
        print(f"Description: {STAGES[stage]['description']}")
        
        try:
            success = stage_functions[stage]()
            results[stage] = 'success' if success else 'failed'
        except Exception as e:
            print(f"Error in stage {stage}: {e}")
            results[stage] = 'error'
        
        if results[stage] != 'success':
            print(f"Stage {stage} failed. Stopping pipeline.")
            break
    
    print_header("PIPELINE COMPLETE")
    print(f"Finished: {datetime.now().isoformat()}")
    
    print("\nResults:")
    for stage, status in results.items():
        symbol = "✓" if status == 'success' else "✗"
        print(f"  {symbol} {stage}: {status}")
    
    return results

# Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GaN MLIP Pipeline')
    
    parser.add_argument('--stages', nargs='+', choices=list(STAGES.keys()),
                       help='Stages to run (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU usage')
    parser.add_argument('--setup-only', action='store_true',
                       help='Run only setup stage')
    parser.add_argument('--tier-a-extra', type=str, default="",
                       help='Extra args appended to dft_tier_a script command (quoted string)')
    parser.add_argument('--tier-b-extra', type=str, default="",
                       help='Extra args appended to dft_tier_b script command (quoted string)')
    
    args = parser.parse_args()
    
    if args.setup_only:
        run_pipeline(['setup'], args.dry_run, not args.no_gpu,
                     tier_a_extra=args.tier_a_extra, tier_b_extra=args.tier_b_extra)
    else:
        run_pipeline(args.stages, args.dry_run, not args.no_gpu,
                     tier_a_extra=args.tier_a_extra, tier_b_extra=args.tier_b_extra)
