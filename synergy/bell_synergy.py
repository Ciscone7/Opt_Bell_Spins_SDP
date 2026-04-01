"""Synergy stopping-criterion scan for Bell NPA (I3322).

Usage
-----
    # from project root:
    python -m synergy.bell_synergy --method pt --k-max 21
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Project path setup — allows running as `python -m synergy.opt_synergy_stop`
# from the project root, or directly from any location.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Use the project's own optimization modules
from optimize.montecarlo import parallel_tempering, simulated_annealing
from optimize.rbm import RBMTrainer
from optimize.bayesian import bayesian
from optimize.synergy import calculate_synergy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DATA_DIR     = _PROJECT_ROOT / "data"
_RESULTS_DIR  = _PROJECT_ROOT / "results" / "synergy"
_PLOTS_DIR    = _RESULTS_DIR / "plots"

LOOKUP_TABLE_PATH = _DATA_DIR / "I3322_npa1_to_npa2_alle_relaxations.txt"

N_DEFAULT = 21  # number of monomials in the adding set (I3322 npa1→npa2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class FunctionLoader:
    """Load the precomputed I3322 SDP values from the lookup-table text file.

    File format (one entry per line)::

        <integer_index>, <float_value>

    The integer index encodes the binary selection mask (LSB = first element).
    """

    def __init__(self, filepath: Path | str, N: int = N_DEFAULT) -> None:
        self.N = N
        self.data: dict[int, float] = {}
        filepath = Path(filepath)
        print(f"Loading look-up table from {filepath}...")
        if not filepath.exists():
            print(f"Warning: {filepath} not found — running in simulation mode (random values).")
        else:
            with open(filepath, "r") as fh:
                for line in fh:
                    if line.strip():
                        parts = line.split(",")
                        self.data[int(parts[0].strip())] = float(parts[1].strip())
            print(f"Lookup table loaded: {len(self.data)} entries.")

    def f(self, x: np.ndarray) -> float:
        """Evaluate f(x) for a binary mask of length N."""
        idx = 0
        for i, bit in enumerate(reversed(x)):
            if bit:
                idx += (1 << i)
        return self.data.get(idx, 100.0)  # big penalty if not found (minimisation)


# ---------------------------------------------------------------------------
# Main scan loop
# ---------------------------------------------------------------------------

def run_scan(
    method: str,
    k_max: int,
    N: int = N_DEFAULT,
    use_lookup_table: bool = True,
    method_params: dict | None = None,
) -> dict:
    """Run the synergy scan for k = 0 … k_max.

    Parameters
    ----------
    method:
        ``'pt'``, ``'sa'``, ``'rbm'`` or ``'bo'`` (bayesian).
    k_max:
        Maximum k value to scan.
    N:
        Dimension of the binary vector (number of monomials in adding set).
    use_lookup_table:
        If True (default) use the precomputed lookup table; otherwise fall
        back to random simulation (useful for quick smoke-tests without the
        data file).
    method_params:
        Optional dictionary of hyper-parameters for the chosen method.
    """
    if method_params is None:
        method_params = {}

    if use_lookup_table:
        loader = FunctionLoader(LOOKUP_TABLE_PATH, N=N)
    else:
        # Simulation mode: random objective for testing the workflow
        rng = np.random.default_rng(42)

        class _FakeLoader:
            N = N_DEFAULT
            def f(self, x):
                return float(rng.random())

        loader = _FakeLoader()

    results: dict = {}

    print(f"\nStarting Synergy Scan  (method={method}, k_max={k_max})")
    print(f"{'k':<5} | {'Best f(x)':<18} | {'Synergy':<15}")
    print("-" * 45)

    # k = 0: empty set
    x0 = np.zeros(N, dtype=int)
    f0 = loader.f(x0)
    results[0] = {"cost": f0, "synergy": 0.0, "state": x0.tolist()}
    print(f"{0:<5} | {f0:<18.6f} | {0.0:<15.6f}")

    for k in range(1, k_max + 1):
        # --- optimisation ---
        if method == "pt":
            res = parallel_tempering(
                obj_func=loader.f,
                N=N,
                k=k,
                num_chains=4,
                num_epochs=10,
                steps_per_epoch=150,
                verbose=False,
            )
            best_val = res["best"]["value"]
            best_selection = res["best"]["selection"]

        elif method == "sa":
            res = simulated_annealing(
                obj_func=loader.f,
                N=N,
                k=k,
                steps=method_params.get("sa_steps", 2000),
                T_start=method_params.get("sa_T_start", 10.0),
                alpha=method_params.get("sa_alpha", 0.99),
                verbose=False,
            )
            cand = res.get("best", res.get("last"))
            best_val = cand["value"]
            best_selection = cand["selection"]

        elif method == "rbm":
            trainer = RBMTrainer(
                obj_func=loader.f,
                N=N,
                hamming_weight=k,
                steps=method_params.get("rbm_steps", 200),
                seed=42,
            )
            trainer.train(num_steps=method_params.get("rbm_steps", 200), verbose=False)
            best_val = trainer.current_cost
            best_selection = trainer.current_vec

        elif method == "bo":
            res = bayesian(
                obj_func=loader.f,
                N=N,
                k=k,
                beta=method_params.get("bo_beta", 2.0),
                n_init=method_params.get("bo_n_init", 10),
                n_iter=method_params.get("bo_n_iter", 40),
                candidates_per_iter=method_params.get("bo_candidates_per_iter", 200),
                verbose=False,
            )
            best_val = res["best_value"]
            best_selection = res["best_selection"]

        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'pt', 'sa', 'rbm', or 'bo'.")

        # --- synergy ---
        syn = calculate_synergy(loader.f, best_selection, k)

        results[k] = {
            "cost": float(best_val),
            "synergy": float(syn),
            "state": best_selection.tolist() if isinstance(best_selection, np.ndarray) else list(best_selection),
        }
        print(f"{k:<5} | {best_val:<18.6f} | {syn:<15.6f}")

    return results


# ---------------------------------------------------------------------------
# Save & plot
# ---------------------------------------------------------------------------

def save_and_plot(results: dict, method: str) -> None:
    """Persist results as JSON and produce a dual-axis plot."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = _RESULTS_DIR / f"synergy_stop_{method}_{timestamp}.json"

    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=4, cls=NumpyEncoder)
    print(f"\nResults saved → {json_path}")

    # --- plot ---
    ks    = sorted(int(k) for k in results)
    costs = [results[k]["cost"]    for k in ks]
    syns  = [results[k]["synergy"] for k in ks]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Hamming Distance k", fontsize=12)
    ax1.set_ylabel("Best SDP Bound f(x)", color="tab:blue", fontsize=12)
    ax1.plot(ks, costs, color="tab:blue", marker="o", label="Best f(x)")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Backbone Synergy", color="tab:red", fontsize=12)
    ax2.plot(ks, syns, color="tab:red", marker="s", linestyle="--",
             linewidth=2, label="Synergy")
    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title(f"Synergy Stopping Criterion — I3322, method={method}", fontsize=13)

    plot_path = _PLOTS_DIR / f"synergy_stop_{method}_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved      → {plot_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--method", choices=["pt", "sa", "rbm", "bo"], default="pt",
        help="Optimisation method: pt (Parallel Tempering), sa (Simulated Annealing), rbm (Restricted Boltzmann Machine), or bo (Bayesian Optimization). Default: pt",
    )
    p.add_argument(
        "--k-max", type=int, default=21,
        help="Maximum k to scan (inclusive). Default: 21",
    )
    p.add_argument(
        "--N", type=int, default=N_DEFAULT,
        help=f"Size of the binary search space (adding-set size). Default: {N_DEFAULT}",
    )
    p.add_argument(
        "--no-lookup-table", dest="use_lookup_table",
        action="store_false", default=True,
        help="Disable the lookup table and run in simulation mode (random values).",
    )

    # Method-specific overrides
    p.add_argument("--sa-steps", type=int, help="SA steps")
    p.add_argument("--sa-T-start", type=float, help="SA start temperature")
    p.add_argument("--sa-alpha", type=float, help="SA alpha decay")
    p.add_argument("--rbm-steps", type=int, help="RBM training steps")
    p.add_argument("--bo-beta", type=float, help="Bayesian UCB beta")
    p.add_argument("--bo-n-init", type=int, help="Bayesian initial samples")
    p.add_argument("--bo-n-iter", type=int, help="Bayesian iterations")
    p.add_argument("--bo-candidates-per-iter", type=int, help="Bayesian candidates per iteration")

    args = p.parse_args(argv)

    # Collect method params
    method_params = {
        "sa_steps": args.sa_steps,
        "sa_T_start": args.sa_T_start,
        "sa_alpha": args.sa_alpha,
        "rbm_steps": args.rbm_steps,
        "bo_beta": args.bo_beta,
        "bo_n_init": args.bo_n_init,
        "bo_n_iter": args.bo_n_iter,
        "bo_candidates_per_iter": args.bo_candidates_per_iter,
    }
    # Filter out None values
    method_params = {k: v for k, v in method_params.items() if v is not None}

    results = run_scan(
        method=args.method,
        k_max=args.k_max,
        N=args.N,
        use_lookup_table=args.use_lookup_table,
        method_params=method_params,
    )
    save_and_plot(results, method=args.method)
    return 0


if __name__ == "__main__":
    sys.exit(main())
