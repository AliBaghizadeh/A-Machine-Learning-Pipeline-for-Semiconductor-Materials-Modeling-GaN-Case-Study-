"""
DFT Budget Management (Generic, GaN Pivot)
=========================================

Tracks and enforces the DFT calculation limit across Tier-A/Tier-B/Active Learning.

This module used to be composition-bucketed for Lu/Sc/Fe oxide. For GaN it is
tag/structure driven; the 'composition' field is optional and may be None.
"""

import json
import os
from datetime import datetime
from pathlib import Path

# =============================================================================
# Budget Configuration
# =============================================================================

DFT_BUDGET = {
    'total_max': 200,           # Maximum total DFT calculations
    'tier_a_max': 5,            # Full relaxations (expensive)
    'tier_b_max': 145,          # Short relaxations / single-points
    'active_learning_reserve': 50,  # Reserved for active learning
}

# Current allocation
DFT_ALLOCATION = {
    'tier_a': {
        'count': 5,
        'structures': [
            # Informational only (not enforced). Prefer lightweight GaN bulk + one defect.
            'GaN_bulk_prim',
            'GaN_bulk_sc_2x2x2',
            'GaN_defect_V_N_sc_2x2x2',
        ],
        'purpose': 'Full ionic + cell relaxation for key reference structures',
        'cost_factor': 10,  # Relative computational cost
    },
    
    'tier_b_single_point': {
        'count': 80,
        'distribution': {
            0.0: 5,    # undoped variations
            0.2: 20,   # x=0.2 configurations
            0.3: 20,   # x=0.3 configurations
            0.4: 20,   # x=0.4 configurations
            0.5: 15,   # x=0.5 configurations
        },
        'purpose': 'Single-point calculations for MLIP training data',
        'cost_factor': 1,
    },
    
    'tier_b_short_relax': {
        'count': 40,
        'distribution': {
            0.0: 2,    # undoped variations
            0.2: 10,   # x=0.2 configurations
            0.3: 10,   # x=0.3 configurations
            0.4: 10,   # x=0.4 configurations
            0.5: 8,    # x=0.5 configurations
        },
        'purpose': 'Short ionic relaxations for off-equilibrium data',
        'cost_factor': 3,
    },
    
    'active_learning': {
        'count': 50,
        'iterations': 2,
        'per_iteration': 25,
        'purpose': 'DFT calls guided by MLIP uncertainty',
        'cost_factor': 2,  # Mix of single-point and short relax
    },
}

# =============================================================================
# Budget Tracker Class
# =============================================================================

class DFTBudgetTracker:
    """Track DFT calculations and enforce budget limits."""
    
    def __init__(self, budget_file='dft/results/dft_budget.json'):
        self.budget_file = Path(budget_file)
        self.budget_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load budget state
        if self.budget_file.exists():
            self.load_state()
        else:
            self.initialize_state()
    
    def initialize_state(self):
        """Initialize fresh budget state."""
        self.state = {
            'total_used': 0,
            'tier_a_used': 0,
            'tier_b_sp_used': 0,
            'tier_b_relax_used': 0,
            'active_learning_used': 0,
            'calculations': [],
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
        }
        self.save_state()
    
    def load_state(self):
        """Load budget state from file."""
        with open(self.budget_file, 'r') as f:
            self.state = json.load(f)
    
    def save_state(self):
        """Save budget state to file."""
        self.state['last_updated'] = datetime.now().isoformat()
        with open(self.budget_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def record_calculation(self, calc_type, structure_id, composition, 
                          energy=None, forces_max=None, status='completed'):
        """
        Record a DFT calculation.
        
        Args:
            calc_type: 'tier_a', 'tier_b_sp', 'tier_b_relax', or 'active_learning'
            structure_id: Identifier for the structure
            composition: Sc fraction x (0.0, 0.2, 0.3, 0.4, 0.5)
            energy: Total energy from calculation (eV)
            forces_max: Maximum force (eV/Ang)
            status: 'completed', 'failed', or 'running'
        """
        # Check budget
        if not self.can_run(calc_type):
            raise RuntimeError(f"DFT budget exceeded for {calc_type}!")
        
        # Record calculation
        calc_record = {
            'id': len(self.state['calculations']),
            'type': calc_type,
            'structure': structure_id,
            'composition': composition,
            'energy': energy,
            'forces_max': forces_max,
            'status': status,
            'timestamp': datetime.now().isoformat(),
        }
        
        self.state['calculations'].append(calc_record)
        self.state['total_used'] += 1
        
        if calc_type == 'tier_a':
            self.state['tier_a_used'] += 1
        elif calc_type == 'tier_b_sp':
            self.state['tier_b_sp_used'] += 1
        elif calc_type == 'tier_b_relax':
            self.state['tier_b_relax_used'] += 1
        elif calc_type == 'active_learning':
            self.state['active_learning_used'] += 1
        
        self.save_state()
        
        return calc_record
    
    def can_run(self, calc_type):
        """Check if we can run another calculation of this type."""
        if calc_type == 'tier_a':
            return self.state['tier_a_used'] < DFT_ALLOCATION['tier_a']['count']
        elif calc_type == 'tier_b_sp':
            return self.state['tier_b_sp_used'] < DFT_ALLOCATION['tier_b_single_point']['count']
        elif calc_type == 'tier_b_relax':
            return self.state['tier_b_relax_used'] < DFT_ALLOCATION['tier_b_short_relax']['count']
        elif calc_type == 'active_learning':
            return self.state['active_learning_used'] < DFT_ALLOCATION['active_learning']['count']
        return False
    
    def get_remaining_budget(self):
        """Get remaining budget for each category."""
        return {
            'total_remaining': DFT_BUDGET['total_max'] - self.state['total_used'],
            'tier_a_remaining': DFT_ALLOCATION['tier_a']['count'] - self.state['tier_a_used'],
            'tier_b_sp_remaining': DFT_ALLOCATION['tier_b_single_point']['count'] - self.state['tier_b_sp_used'],
            'tier_b_relax_remaining': DFT_ALLOCATION['tier_b_short_relax']['count'] - self.state['tier_b_relax_used'],
            'active_learning_remaining': DFT_ALLOCATION['active_learning']['count'] - self.state['active_learning_used'],
        }
    
    def get_allocation_for_composition(self, composition):
        """Legacy helper; composition may be None in GaN workflow."""
        comp_calcs = [c for c in self.state['calculations'] if c.get('composition') == composition]
        return {
            'tier_a': sum(1 for c in comp_calcs if c.get('type') == 'tier_a'),
            'tier_b_sp': sum(1 for c in comp_calcs if c.get('type') == 'tier_b_sp'),
            'tier_b_relax': sum(1 for c in comp_calcs if c.get('type') == 'tier_b_relax'),
        }
    
    def print_status(self):
        """Print current budget status."""
        remaining = self.get_remaining_budget()
        
        print("\n" + "=" * 60)
        print("DFT BUDGET STATUS")
        print("=" * 60)
        print(f"\nTotal calculations: {self.state['total_used']} / {DFT_BUDGET['total_max']}")
        print(f"  Remaining: {remaining['total_remaining']}")
        
        print(f"\nTier A (full relax): {self.state['tier_a_used']} / {DFT_ALLOCATION['tier_a']['count']}")
        print(f"Tier B (single-point): {self.state['tier_b_sp_used']} / {DFT_ALLOCATION['tier_b_single_point']['count']}")
        print(f"Tier B (short relax): {self.state['tier_b_relax_used']} / {DFT_ALLOCATION['tier_b_short_relax']['count']}")
        print(f"Active Learning: {self.state['active_learning_used']} / {DFT_ALLOCATION['active_learning']['count']}")
        
        # Composition bucket summary is not meaningful for GaN; keep a small summary if present.
        comps = sorted({c.get('composition') for c in self.state.get('calculations', []) if c.get('composition') is not None})
        if comps:
            print("\nBy composition (legacy):")
            for x in comps:
                counts = self.get_allocation_for_composition(x)
                total = sum(counts.values())
                print(f"  x={x}: {total} calculations (A:{counts['tier_a']}, SP:{counts['tier_b_sp']}, SR:{counts['tier_b_relax']})")
        
        print("\n" + "=" * 60)

# =============================================================================
# Structure Selection Helper
# =============================================================================

def select_structures_for_dft(composition, n_structures, existing_structures):
    """
    Select diverse structures for DFT calculation.
    
    Uses farthest-point sampling to maximize diversity.
    
    Args:
        composition: Sc fraction x
        n_structures: Number of structures to select
        existing_structures: List of (structure_id, features) tuples
    
    Returns:
        List of selected structure IDs
    """
    import numpy as np
    
    if len(existing_structures) <= n_structures:
        return [s[0] for s in existing_structures]
    
    # Extract features
    features = np.array([s[1] for s in existing_structures])
    
    # Normalize features
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    
    # Farthest point sampling
    selected_idx = [0]  # Start with first structure
    
    for _ in range(n_structures - 1):
        # Calculate distances to selected points
        distances = []
        for i in range(len(features)):
            min_dist = min([np.linalg.norm(features[i] - features[j]) 
                          for j in selected_idx])
            distances.append(min_dist)
        
        # Select point with maximum minimum distance
        selected_idx.append(np.argmax(distances))
    
    return [existing_structures[i][0] for i in selected_idx]

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Test the budget tracker
    print("Testing DFT Budget Tracker")
    print("=" * 60)
    
    tracker = DFTBudgetTracker()
    tracker.print_status()
    
    # Simulate some calculations
    print("\nSimulating calculations...")
    
    # Tier A (GaN examples; composition is optional and may be None)
    tracker.record_calculation('tier_a', 'GaN_bulk_sc_2x2x2', None, energy=-100.0, forces_max=0.20)

    # Tier B
    tracker.record_calculation('tier_b_sp', 'GaN_defect_V_N_sc_2x2x2', None, energy=-98.0, forces_max=0.40)
    tracker.record_calculation('tier_b_sp', 'GaN_bulk_sc_2x2x2__rattle00', None, energy=-99.5, forces_max=0.80)
    tracker.record_calculation('tier_b_relax', 'GaN_bulk_sc_2x2x2', None, energy=-101.0, forces_max=0.15)
    
    tracker.print_status()
