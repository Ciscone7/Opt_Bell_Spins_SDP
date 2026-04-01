"""Synergy stopping-criterion scan for Many-Body (Spins) Systems.

This script demonstrates the modularity of the synergy calculation by applying it
to a spin-chain observable (Ground-State Energy).

Usage
-----
    # from project root:
    python -m synergy.spins_synergy --N 6 --k-max 4 --method sa
"""

from __future__ import annotations
import argparse
import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

# Project paths
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from spins.models import heisenberg_hamiltonian_dict
from spins.symmetry import SymmetryManager
from spins.spins_optimize import build_npa_basis_sets, make_spin_objective, run_single_optimization, _default_method_params
from optimize.synergy import calculate_synergy


def run_spins_scan(
    N: int = 6,
    k_max: int = 4,
    method: str = "sa",
    method_params: dict | None = None,
) -> dict:
    """Run synergy scan for a spin-system ground-energy bound."""
    
    # 1. Setup physics problem
    hamiltonian = heisenberg_hamiltonian_dict(N=N, boundary="open")
    symmetry_manager = SymmetryManager.default_for_heisenberg(N=N)
    
    # 2. Build basis candidate sets
    starting_set, adding_set, _ = build_npa_basis_sets(N=N, start_level=1, end_level=2)
    L = len(adding_set)
    
    # 3. Create objective function (actual SDP solve)
    obj_func = make_spin_objective(
        starting_set=starting_set,
        adding_set=adding_set,
        operator=hamiltonian,
        symmetry_manager=symmetry_manager,
        sense="min",
    )
    
    if method_params is None:
        method_params = _default_method_params(method)

    results = {}
    print(f"\nStarting Spins Synergy Scan (N={N}, method={method}, k_max={k_max})")
    print(f"{'k':<5} | {'Bound Energy':<18} | {'Synergy':<15}")
    print("-" * 45)

    # Pre-evaluate f(0) to pass to synergy calculator (efficiency)
    f0 = obj_func(np.zeros(L, dtype=int))
    
    for k in range(0, k_max + 1):
        if k == 0:
            best_val = -f0 # objective is -lb
            best_selection = np.zeros(L, dtype=int)
        else:
            # --- Optimization ---
            res = run_single_optimization(
                obj_func=obj_func,
                L=L,
                k=k,
                seed=42,
                method=method,
                method_params=method_params,
            )
            # best_value in SpinOptimizationResult is already the positive LB
            best_val = res.best_value
            best_selection = res.mask

        # --- Synergy ---
        # Note: obj_func is internal minimizer (-lb). 
        # Synergy needs a consistent objective logic. 
        syn = calculate_synergy(obj_func, best_selection, k, f_zero=f0)

        results[k] = {
            "energy": float(best_val),
            "synergy": float(syn),
            "state": best_selection.tolist()
        }
        print(f"{k:<5} | {best_val:<18.6f} | {syn:<15.6f}")

    return results

def main():
    p = argparse.ArgumentParser(description="Spin system synergy scan")
    p.add_argument("--N", type=int, default=4, help="Number of sites")
    p.add_argument("--k-max", type=int, default=3, help="Max k to scan")
    p.add_argument("--method", type=str, default="sa", choices=["sa", "pt", "bo", "rbm"])
    args = p.parse_args()

    results = run_spins_scan(N=args.N, k_max=args.k_max, method=args.method)
    
    # Save results
    out_dir = _PROJECT_ROOT / "results" / "synergy_spins"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"spins_synergy_{args.method}_{timestamp}.json"
    
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved results to {path}")

if __name__ == "__main__":
    main()
