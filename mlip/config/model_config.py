"""
MACE Model Configuration for GaN (Wurtzite, Minimal-Load)
=========================================================

This configuration targets fast iteration on RTX 5080 (16GB VRAM) for
defect/dislocation-scale *structure* modeling using an MLIP.

Important scope note:
- MLIPs do not predict band gaps directly. Use MLIP to relax large structures,
  then run electronic-structure DFT (or higher-level methods) on smaller
  representative cells/snapshots for band gap / optical properties.
"""

# =============================================================================
# Model Architecture
# =============================================================================

MACE_CONFIG = {
    # Model type
    'model': 'MACE',
    'interaction_first': 'RealAgnosticInteractionBlock',
    'interaction': 'RealAgnosticInteractionBlock',
    
    # Radial basis
    'r_max': 5.0,              # Cutoff radius (Angstrom)
    'radial_type': 'bessel',
    'num_radial_basis': 8,
    'num_cutoff_basis': 5,
    
    # Spherical harmonics
    'max_ell': 3,              # Maximum angular momentum (L_max)
    'correlation': 3,          # Body order
    
    # Network architecture
    'num_interactions': 2,     # Number of interaction blocks
    'hidden_irreps': '128x0e + 128x1o + 64x2e + 32x3o',  # Hidden representations
    # Preferred for newer MACE CLI (avoids hidden_irreps channel-mismatch assertion)
    'num_channels': 128,
    'max_L': 3,
    
    # Output
    'num_outputs': 1,          # Single energy output
    'mlp_irreps': '16x0e',     # MLP hidden irreps
    
    # Atomic species (Ga, N)
    'atomic_numbers': [31, 7],
    'avg_num_neighbors': 12,
    
    # Scaling
    'scaling': 'std',
    'atomic_energies': None,   # Will be computed from training data
}

# =============================================================================
# Training Configuration
# =============================================================================

TRAINING_CONFIG = {
    # Optimizer
    'optimizer': 'adam',
    'learning_rate': 1e-4,
    'weight_decay': 1e-8,
    'amsgrad': True,
    
    # Learning rate scheduler
    'scheduler': 'ReduceLROnPlateau',
    'lr_factor': 0.8,
    'lr_patience': 50,
    'lr_min': 1e-6,
    
    # Batch size (limited by 16GB VRAM)
    'batch_size': 4,
    'max_num_atoms': 500,      # Max atoms per batch
    
    # Training
    'max_num_epochs': 500,
    'patience': 25,            # Early stopping patience (minimum-load iteration default)
    
    # Loss weights
    'energy_weight': 1.0,
    'forces_weight': 10.0,
    'stress_weight': 0.1,
    
    # EMA (Exponential Moving Average)
    'ema': True,
    'ema_decay': 0.99,
    
    # Validation
    'valid_batch_size': 8,
    'eval_interval': 10,       # Evaluate every N epochs
    
    # Checkpointing
    'save_all_checkpoints': False,
    'checkpoint_interval': 50,
    
    # Device
    'device': 'cuda',
    # Project rule: never run with 4 workers (too slow). Use 8 or 12.
    'num_workers': 12,
    # Dataloader memory path tuning
    'pin_memory': True,
    # Not currently exposed by mace_run_train CLI (kept for policy tracking).
    'persistent_workers': True,
    'prefetch_factor': 4,
    # Stage training inputs from /mnt/* to fast Linux scratch when possible.
    'io_stage_from_mnt': True,
    'io_stage_dir': '/tmp/mlip_fast_io',
}

# =============================================================================
# GPU Memory Optimization (for 16GB VRAM)
# =============================================================================

GPU_CONFIG = {
    'device': 'cuda:0',
    'max_memory_gb': 14,       # Leave 2GB for system
    
    # Memory optimization strategies
    'gradient_checkpointing': True,
    'mixed_precision': True,   # Use FP16
    'torch_compile': False,    # Disable for stability
    
    # Batch size recommendations for different system sizes
    'batch_size_by_atoms': {
        100: 8,
        200: 4,
        300: 2,
        500: 1,
    },
}

# =============================================================================
# Target Accuracy
# =============================================================================

TARGET_ACCURACY = {
    'energy_mae': 0.005,       # eV/atom
    'forces_mae': 0.05,        # eV/Ang
    'stress_mae': 0.1,         # GPa
    
    'energy_rmse': 0.01,       # eV/atom
    'forces_rmse': 0.1,        # eV/Ang
}

# =============================================================================
# Element Properties
# =============================================================================

ELEMENT_PROPERTIES = {
    'Ga': {
        'atomic_number': 31,
        'mass': 69.723,
        'atomic_energy': None,
    },
    'N': {
        'atomic_number': 7,
        'mass': 14.007,
        'atomic_energy': None,
    },
}

# =============================================================================
# Helper Functions
# =============================================================================

def get_model_config_for_system(n_atoms, gpu_memory_gb=16):
    """
    Adjust model config based on system size and GPU memory.
    
    Args:
        n_atoms: Number of atoms in the system
        gpu_memory_gb: Available GPU memory in GB
    
    Returns:
        Dict with adjusted configuration
    """
    config = MACE_CONFIG.copy()
    training = TRAINING_CONFIG.copy()
    
    # Adjust batch size based on atoms
    if n_atoms > 400:
        training['batch_size'] = 1
    elif n_atoms > 300:
        training['batch_size'] = 2
    elif n_atoms > 200:
        training['batch_size'] = 4
    else:
        training['batch_size'] = 8
    
    # Reduce model size for limited memory
    if gpu_memory_gb < 12:
        config['num_channels'] = 64
        config['max_L'] = 2
        config['num_interactions'] = 1
    
    return {
        'model': config,
        'training': training,
    }

def get_batch_size(n_atoms, max_memory_gb=14):
    """
    Calculate optimal batch size for given number of atoms.
    
    Args:
        n_atoms: Number of atoms
        max_memory_gb: Maximum GPU memory to use (GB)
    
    Returns:
        Recommended batch size
    """
    for atoms, batch_size in sorted(GPU_CONFIG['batch_size_by_atoms'].items(), reverse=True):
        if n_atoms >= atoms:
            return batch_size
    return 8

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import json
    
    print("=" * 60)
    print("MACE Model Configuration")
    print("=" * 60)
    
    print("\nModel Architecture:")
    for key, value in MACE_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nTraining Configuration:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nGPU Configuration:")
    for key, value in GPU_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nTarget Accuracy:")
    for key, value in TARGET_ACCURACY.items():
        print(f"  {key}: {value}")
    
    # Test batch size calculation
    print("\n" + "=" * 60)
    print("Batch Size Recommendations:")
    print("=" * 60)
    for n_atoms in [100, 200, 300, 400, 500]:
        batch_size = get_batch_size(n_atoms)
        print(f"  {n_atoms} atoms: batch_size = {batch_size}")
