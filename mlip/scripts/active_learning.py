"""
Active Learning for MLIP
========================
Implement uncertainty-based sampling to maximize DFT efficiency.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dft.config.dft_budget import DFTBudgetTracker

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
MLIP_DATA_DIR = PROJECT_ROOT / "mlip" / "data"
MLIP_MODELS_DIR = PROJECT_ROOT / "mlip" / "models"
MLIP_RESULTS_DIR = PROJECT_ROOT / "mlip" / "results"
DFT_RESULTS_DIR = PROJECT_ROOT / "dft" / "results"
DFT_STRUCTURES_DIR = PROJECT_ROOT / "dft" / "structures"

# Active learning parameters
AL_CONFIG = {
    'n_iterations': 2,                # Number of active learning iterations
    'n_samples_per_iter': 25,         # DFT calculations per iteration
    'uncertainty_threshold': 0.1,     # Energy uncertainty threshold (eV)
    'diversity_weight': 0.5,          # Weight for diversity vs uncertainty
    'ensemble_size': 5,               # Number of models in ensemble
}

# =============================================================================
# Uncertainty Quantification
# =============================================================================

class UncertaintyEstimator:
    """Estimate uncertainty using ensemble of models."""
    
    def __init__(self, models: List):
        """
        Initialize with list of trained models.
        
        Args:
            models: List of trained MLIP models
        """
        self.models = models
        
    def predict_with_uncertainty(self, atoms) -> Tuple[float, float, np.ndarray]:
        """
        Predict energy with uncertainty.
        
        Args:
            atoms: ASE Atoms object
        
        Returns:
            mean_energy: Mean predicted energy
            uncertainty: Standard deviation of predictions
            forces: Mean forces
        """
        if not self.models:
            return 0.0, 1.0, np.zeros((len(atoms), 3))
        
        energies = []
        forces_list = []
        
        for model in self.models:
            # Get prediction from each model
            # In practice: energy, forces = model.predict(atoms)
            # Placeholder:
            energy = np.random.normal(0, 0.1)  # Simulated
            forces = np.random.randn(len(atoms), 3) * 0.1  # Simulated
            energies.append(energy)
            forces_list.append(forces)
        
        mean_energy = np.mean(energies)
        uncertainty = np.std(energies)
        mean_forces = np.mean(forces_list, axis=0)
        
        return mean_energy, uncertainty, mean_forces
    
    def compute_force_uncertainty(self, atoms) -> np.ndarray:
        """
        Compute per-atom force uncertainty.
        
        Args:
            atoms: ASE Atoms object
        
        Returns:
            Force uncertainties for each atom
        """
        forces_list = []
        
        for model in self.models:
            # Placeholder
            forces = np.random.randn(len(atoms), 3) * 0.1
            forces_list.append(forces)
        
        return np.std(forces_list, axis=0)

# =============================================================================
# Diversity Sampling
# =============================================================================

def compute_fingerprint(atoms) -> np.ndarray:
    """
    Compute structural fingerprint for diversity.
    
    Args:
        atoms: ASE Atoms object
    
    Returns:
        Fingerprint vector
    """
    from ase import Atoms
    
    # Simple fingerprint based on composition and cell
    n_atoms = len(atoms)
    numbers = atoms.get_atomic_numbers()
    
    # Count elements
    n_fe = np.sum(numbers == 26)
    n_lu = np.sum(numbers == 71)
    n_sc = np.sum(numbers == 21)
    n_o = np.sum(numbers == 8)
    
    # Cell parameters
    cell = atoms.get_cell()
    a = np.linalg.norm(cell[0])
    b = np.linalg.norm(cell[1])
    c = np.linalg.norm(cell[2])
    volume = atoms.get_volume()
    
    # Composition ratio
    sc_ratio = n_sc / (n_lu + n_sc + 1e-8)
    
    fingerprint = np.array([
        n_atoms, n_fe, n_lu, n_sc, n_o,
        a, b, c, volume, sc_ratio
    ])
    
    return fingerprint

def farthest_point_sampling(fingerprints: np.ndarray, n_samples: int) -> List[int]:
    """
    Select diverse samples using farthest point sampling.
    
    Args:
        fingerprints: Array of fingerprints (n_structures, n_features)
        n_samples: Number of samples to select
    
    Returns:
        Indices of selected structures
    """
    n_structures = len(fingerprints)
    
    if n_structures <= n_samples:
        return list(range(n_structures))
    
    # Normalize fingerprints
    fingerprints = (fingerprints - fingerprints.mean(axis=0)) / (fingerprints.std(axis=0) + 1e-8)
    
    # Start with random point
    selected = [np.random.randint(n_structures)]
    
    for _ in range(n_samples - 1):
        # Compute distances to selected points
        distances = np.zeros(n_structures)
        for i in range(n_structures):
            min_dist = min([np.linalg.norm(fingerprints[i] - fingerprints[j]) 
                          for j in selected])
            distances[i] = min_dist
        
        # Select point with maximum minimum distance
        selected.append(np.argmax(distances))
    
    return selected

def select_structures_for_dft(
    candidates: List,
    uncertainty_estimator: UncertaintyEstimator,
    n_select: int,
    diversity_weight: float = 0.5
) -> Tuple[List, List[float]]:
    """
    Select structures for DFT calculation based on uncertainty and diversity.
    
    Args:
        candidates: List of candidate structures (ASE Atoms)
        uncertainty_estimator: Uncertainty estimator
        n_select: Number of structures to select
        diversity_weight: Weight for diversity (0 = pure uncertainty, 1 = pure diversity)
    
    Returns:
        Selected structures and their uncertainties
    """
    print(f"\nSelecting {n_select} structures from {len(candidates)} candidates...")
    
    # Compute uncertainties
    uncertainties = []
    fingerprints = []
    
    for i, atoms in enumerate(candidates):
        energy, unc, forces = uncertainty_estimator.predict_with_uncertainty(atoms)
        uncertainties.append(unc)
        fingerprints.append(compute_fingerprint(atoms))
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(candidates)}")
    
    uncertainties = np.array(uncertainties)
    fingerprints = np.array(fingerprints)
    
    # Normalize uncertainties
    uncertainties_norm = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min() + 1e-8)
    
    # Select based on combined score
    n_uncertainty = int(n_select * (1 - diversity_weight))
    n_diversity = n_select - n_uncertainty
    
    # Select high uncertainty structures
    uncertainty_indices = np.argsort(uncertainties)[-n_uncertainty:]
    
    # Select diverse structures from remaining
    remaining_indices = [i for i in range(len(candidates)) if i not in uncertainty_indices]
    remaining_fingerprints = fingerprints[remaining_indices]
    diversity_indices_relative = farthest_point_sampling(remaining_fingerprints, n_diversity)
    diversity_indices = [remaining_indices[i] for i in diversity_indices_relative]
    
    # Combine
    selected_indices = list(uncertainty_indices) + diversity_indices
    selected_structures = [candidates[i] for i in selected_indices]
    selected_uncertainties = [uncertainties[i] for i in selected_indices]
    
    print(f"  Selected {len(selected_structures)} structures")
    print(f"  Mean uncertainty: {np.mean(selected_uncertainties):.4f}")
    print(f"  Max uncertainty: {np.max(selected_uncertainties):.4f}")
    
    return selected_structures, selected_uncertainties

# =============================================================================
# Active Learning Loop
# =============================================================================

def run_active_learning_iteration(
    iteration: int,
    candidates: List,
    n_select: int,
    budget_tracker: DFTBudgetTracker
) -> Dict:
    """
    Run one active learning iteration.
    
    Args:
        iteration: Iteration number
        candidates: Candidate structures
        n_select: Number of structures to select for DFT
        budget_tracker: DFT budget tracker
    
    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"ACTIVE LEARNING ITERATION {iteration}")
    print(f"{'='*60}")
    
    results = {
        'iteration': iteration,
        'n_candidates': len(candidates),
        'n_selected': n_select,
        'started': datetime.now().isoformat(),
    }
    
    # Check budget
    remaining = budget_tracker.get_remaining_budget()
    if remaining['active_learning_remaining'] < n_select:
        print(f"WARNING: Active learning budget exhausted!")
        print(f"  Remaining: {remaining['active_learning_remaining']}")
        n_select = remaining['active_learning_remaining']
        
        if n_select == 0:
            results['status'] = 'budget_exhausted'
            return results
    
    # Load trained models (ensemble)
    print("\nLoading model ensemble...")
    models = []
    for i in range(AL_CONFIG['ensemble_size']):
        model_path = MLIP_MODELS_DIR / f'mace_model_{i}.pt'
        if model_path.exists():
            # Load model
            # model = torch.load(model_path)
            # models.append(model)
            pass
    
    if not models:
        print("WARNING: No trained models found. Using random selection.")
        # Random selection as fallback
        indices = np.random.choice(len(candidates), size=min(n_select, len(candidates)), replace=False)
        selected = [candidates[i] for i in indices]
        uncertainties = [1.0] * len(selected)
    else:
        # Uncertainty-based selection
        uncertainty_estimator = UncertaintyEstimator(models)
        selected, uncertainties = select_structures_for_dft(
            candidates, uncertainty_estimator, n_select
        )
    
    # Save selected structures for DFT
    print("\nSaving selected structures...")
    selected_dir = DFT_RESULTS_DIR / 'al_selected'
    selected_dir.mkdir(parents=True, exist_ok=True)
    
    selected_info = []
    for i, (atoms, unc) in enumerate(zip(selected, uncertainties)):
        filename = f'al_iter{iteration}_struct{i:03d}.cif'
        filepath = selected_dir / filename
        
        from ase.io import write
        write(filepath, atoms)
        
        selected_info.append({
            'filename': filename,
            'filepath': str(filepath),
            'uncertainty': float(unc),
        })
    
    results['selected'] = selected_info
    results['status'] = 'completed'
    results['completed'] = datetime.now().isoformat()
    
    # Save results
    results_file = MLIP_RESULTS_DIR / f'al_iteration_{iteration}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nIteration {iteration} complete:")
    print(f"  Selected: {len(selected)} structures")
    print(f"  Results saved to: {results_file}")
    
    return results

def run_active_learning_pipeline():
    """
    Run the complete active learning pipeline.
    """
    print("\n" + "="*60)
    print("ACTIVE LEARNING PIPELINE")
    print("="*60)
    
    # Initialize budget tracker
    budget_tracker = DFTBudgetTracker(DFT_RESULTS_DIR / 'dft_budget.json')
    budget_tracker.print_status()
    
    # Generate candidate structures
    print("\nLoading candidate structures...")
    from ase.io import read
    
    # Load all generated structures
    structure_info_file = DFT_STRUCTURES_DIR / 'structure_info.json'
    if not structure_info_file.exists():
        print("ERROR: No structure info found. Run structure_generation.py first.")
        return
    
    with open(structure_info_file, 'r') as f:
        structure_info = json.load(f)
    
    candidates = []
    for struct_data in structure_info['structures']:
        filepath = struct_data['filepath']
        if Path(filepath).exists():
            atoms = read(filepath)
            candidates.append(atoms)
    
    print(f"Loaded {len(candidates)} candidate structures")
    
    # Run active learning iterations
    all_results = {
        'started': datetime.now().isoformat(),
        'iterations': [],
    }
    
    for iteration in range(1, AL_CONFIG['n_iterations'] + 1):
        results = run_active_learning_iteration(
            iteration=iteration,
            candidates=candidates,
            n_select=AL_CONFIG['n_samples_per_iter'],
            budget_tracker=budget_tracker
        )
        
        all_results['iterations'].append(results)
        
        if results['status'] == 'budget_exhausted':
            print("\nBudget exhausted, stopping active learning.")
            break
        
        # In practice: Run DFT on selected structures, then retrain model
        print(f"\nIteration {iteration} complete.")
        print("Next steps:")
        print("  1. Run DFT on selected structures")
        print("  2. Update training data")
        print("  3. Retrain model")
        print("  4. Run next iteration")
    
    all_results['completed'] = datetime.now().isoformat()
    
    # Save all results
    results_file = MLIP_RESULTS_DIR / 'active_learning_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print("ACTIVE LEARNING COMPLETE")
    print("="*60)
    print(f"Total iterations: {len(all_results['iterations'])}")
    print(f"Results saved to: {results_file}")
    
    return all_results

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Active learning for MLIP')
    parser.add_argument('--iterations', type=int, default=None,
                       help='Number of iterations')
    parser.add_argument('--samples', type=int, default=None,
                       help='Samples per iteration')
    
    args = parser.parse_args()
    
    if args.iterations:
        AL_CONFIG['n_iterations'] = args.iterations
    if args.samples:
        AL_CONFIG['n_samples_per_iter'] = args.samples
    
    run_active_learning_pipeline()