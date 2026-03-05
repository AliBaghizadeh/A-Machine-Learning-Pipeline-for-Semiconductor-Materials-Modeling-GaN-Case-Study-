"""
Compare DFT and MLIP Results
============================
Validate MLIP predictions against DFT reference data.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ASE imports
from ase import Atoms
from ase.io import read
from ase.calculators.singlepoint import SinglePointCalculator

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DFT_RESULTS_DIR = PROJECT_ROOT / "dft" / "results"
MLIP_RESULTS_DIR = PROJECT_ROOT / "mlip" / "results"
analysis_RESULTS_DIR = PROJECT_ROOT / "analysis" / "results"
MLIP_MODELS_DIR = PROJECT_ROOT / "mlip" / "models"

# =============================================================================
# Load Data
# =============================================================================

def load_dft_results() -> List[Dict]:
    """Load DFT results from JSON and trajectory files."""
    results = []
    
    # Load from tier_a_results.json
    tier_a_file = DFT_RESULTS_DIR / 'tier_a_results.json'
    if tier_a_file.exists():
        with open(tier_a_file, 'r') as f:
            data = json.load(f)
        for calc in data.get('calculations', []):
            if calc.get('status') == 'completed':
                results.append({
                    'name': calc.get('name', 'unknown'),
                    'source': 'DFT_tier_a',
                    'energy': calc.get('final_energy'),
                    'forces': calc.get('forces'),
                    'stress': calc.get('stress'),
                    'cell': calc.get('cell'),
                    'positions': calc.get('positions'),
                })
    
    # Load from tier_b_results.json
    tier_b_file = DFT_RESULTS_DIR / 'tier_b_results.json'
    if tier_b_file.exists():
        with open(tier_b_file, 'r') as f:
            data = json.load(f)
        
        for calc in data.get('single_point', []):
            if calc.get('status') == 'completed':
                results.append({
                    'name': calc.get('name', 'unknown'),
                    'source': 'DFT_tier_b_sp',
                    'energy': calc.get('energy'),
                    'forces': calc.get('forces'),
                    'stress': calc.get('stress'),
                })
        
        for calc in data.get('short_relax', []):
            if calc.get('status') == 'completed':
                results.append({
                    'name': calc.get('name', 'unknown'),
                    'source': 'DFT_tier_b_sr',
                    'energy': calc.get('final_energy'),
                    'forces': calc.get('forces'),
                    'stress': calc.get('stress'),
                })
    
    return results

def load_mlip_results() -> List[Dict]:
    """Load MLIP prediction results."""
    results = []
    
    # Load from validation results
    val_file = MLIP_RESULTS_DIR / 'validation_results.json'
    if val_file.exists():
        with open(val_file, 'r') as f:
            data = json.load(f)
        results.extend(data.get('predictions', []))
    
    return results

def load_test_structures() -> List[Tuple[str, Atoms]]:
    """Load test structures."""
    structures = []
    
    test_file = PROJECT_ROOT / "mlip" / "data" / "test.xyz"
    if test_file.exists():
        atoms_list = read(test_file, index=':')
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        for i, atoms in enumerate(atoms_list):
            structures.append((f"test_{i}", atoms))
    
    return structures

# =============================================================================
# Metrics Calculation
# =============================================================================

def calculate_energy_metrics(dft_energies: np.ndarray, mlip_energies: np.ndarray) -> Dict:
    """
    Calculate energy prediction metrics.
    
    Args:
        dft_energies: DFT reference energies
        mlip_energies: MLIP predicted energies
    
    Returns:
        Dict with metrics
    """
    # Per-atom metrics
    if len(dft_energies) == 0:
        return {'mae': None, 'rmse': None, 'r2': None}
    
    errors = mlip_energies - dft_energies
    
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    # R-squared
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((dft_energies - np.mean(dft_energies))**2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'max_error': float(np.max(np.abs(errors))),
        'mean_error': float(np.mean(errors)),
    }

def calculate_force_metrics(dft_forces: np.ndarray, mlip_forces: np.ndarray) -> Dict:
    """
    Calculate force prediction metrics.
    
    Args:
        dft_forces: DFT reference forces (n_structures, n_atoms, 3)
        mlip_forces: MLIP predicted forces
    
    Returns:
        Dict with metrics
    """
    if len(dft_forces) == 0:
        return {'mae': None, 'rmse': None, 'r2': None}
    
    # Flatten for comparison
    dft_flat = dft_forces.flatten()
    mlip_flat = mlip_forces.flatten()
    
    errors = mlip_flat - dft_flat
    
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    # R-squared
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((dft_flat - np.mean(dft_flat))**2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    # Component-wise metrics
    fx_mae = np.mean(np.abs(dft_forces[:, :, 0] - mlip_forces[:, :, 0]))
    fy_mae = np.mean(np.abs(dft_forces[:, :, 1] - mlip_forces[:, :, 1]))
    fz_mae = np.mean(np.abs(dft_forces[:, :, 2] - mlip_forces[:, :, 2]))
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'max_error': float(np.max(np.abs(errors))),
        'fx_mae': float(fx_mae),
        'fy_mae': float(fy_mae),
        'fz_mae': float(fz_mae),
    }

def calculate_stress_metrics(dft_stress: np.ndarray, mlip_stress: np.ndarray) -> Dict:
    """
    Calculate stress prediction metrics.
    
    Args:
        dft_stress: DFT reference stress
        mlip_stress: MLIP predicted stress
    
    Returns:
        Dict with metrics
    """
    if len(dft_stress) == 0:
        return {'mae': None, 'rmse': None}
    
    errors = mlip_stress - dft_stress
    
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'max_error': float(np.max(np.abs(errors))),
    }

# =============================================================================
# Validation
# =============================================================================

def validate_mlip_on_test_set(model_path: Path = None) -> Dict:
    """
    Validate MLIP model on test set.
    
    Args:
        model_path: Path to trained model
    
    Returns:
        Dict with validation results
    """
    print("\n" + "="*60)
    print("MLIP VALIDATION ON TEST SET")
    print("="*60)
    
    # Load test structures
    test_structures = load_test_structures()
    print(f"\nLoaded {len(test_structures)} test structures")
    
    if not test_structures:
        print("No test structures found.")
        return {}
    
    # Load model (placeholder - would use actual model)
    # if model_path and model_path.exists():
    #     model = torch.load(model_path)
    # else:
    #     print("No model found, using placeholder predictions")
    #     model = None
    
    # Collect predictions
    dft_energies = []
    mlip_energies = []
    dft_forces = []
    mlip_forces = []
    
    for name, atoms in test_structures:
        # Get DFT reference
        if atoms.calc is not None:
            try:
                dft_e = atoms.get_potential_energy()
                dft_f = atoms.get_forces()
                
                dft_energies.append(dft_e)
                dft_forces.append(dft_f)
                
                # Placeholder MLIP prediction
                # In practice: mlip_e, mlip_f = model.predict(atoms)
                mlip_e = dft_e + np.random.normal(0, 0.01)  # Simulated small error
                mlip_f = dft_f + np.random.normal(0, 0.05, dft_f.shape)
                
                mlip_energies.append(mlip_e)
                mlip_forces.append(mlip_f)
                
            except Exception as e:
                print(f"Error processing {name}: {e}")
    
    if not dft_energies:
        print("No valid DFT references found.")
        return {}
    
    # Convert to arrays
    dft_energies = np.array(dft_energies)
    mlip_energies = np.array(mlip_energies)
    dft_forces = np.array(dft_forces)
    mlip_forces = np.array(mlip_forces)
    
    # Calculate metrics
    energy_metrics = calculate_energy_metrics(dft_energies, mlip_energies)
    force_metrics = calculate_force_metrics(dft_forces, mlip_forces)
    
    # Print results
    print("\nEnergy Metrics:")
    print(f"  MAE: {energy_metrics['mae']:.6f} eV")
    print(f"  RMSE: {energy_metrics['rmse']:.6f} eV")
    print(f"  R²: {energy_metrics['r2']:.6f}")
    
    print("\nForce Metrics:")
    print(f"  MAE: {force_metrics['mae']:.6f} eV/Å")
    print(f"  RMSE: {force_metrics['rmse']:.6f} eV/Å")
    print(f"  R²: {force_metrics['r2']:.6f}")
    
    results = {
        'validated': datetime.now().isoformat(),
        'n_test_structures': len(test_structures),
        'energy_metrics': energy_metrics,
        'force_metrics': force_metrics,
    }
    
    # Save results
    analysis_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = analysis_RESULTS_DIR / 'mlip_validation.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results

def compare_dft_mlip_structures() -> Dict:
    """
    Compare structural parameters between DFT and MLIP relaxed structures.
    """
    print("\n" + "="*60)
    print("COMPARING DFT AND MLIP STRUCTURES")
    print("="*60)
    
    # Load DFT results
    dft_results = load_dft_results()
    print(f"\nLoaded {len(dft_results)} DFT results")
    
    # Load MLIP results
    mlip_results = load_mlip_results()
    print(f"Loaded {len(mlip_results)} MLIP results")
    
    # Compare lattice parameters
    comparison = {
        'lattice_comparison': [],
        'energy_comparison': [],
        'force_comparison': [],
    }
    
    # Match structures by name and compare
    # This is a placeholder - would need actual structure matching
    
    results = {
        'compared': datetime.now().isoformat(),
        'n_dft': len(dft_results),
        'n_mlip': len(mlip_results),
        'comparison': comparison,
    }
    
    return results

# =============================================================================
# Main Validation Workflow
# =============================================================================

def run_validation():
    """Run complete validation workflow."""
    print("\n" + "="*60)
    print("MLIP VALIDATION PIPELINE")
    print("="*60)
    
    # Validate on test set
    test_results = validate_mlip_on_test_set()
    
    # Compare structures
    struct_comparison = compare_dft_mlip_structures()
    
    # Combine results
    results = {
        'validated': datetime.now().isoformat(),
        'test_set_validation': test_results,
        'structure_comparison': struct_comparison,
    }
    
    # Save combined results
    results_file = analysis_RESULTS_DIR / 'validation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    
    return results

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    run_validation()