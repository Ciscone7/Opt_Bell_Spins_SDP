"""Run optimization sweeps for Bell basis selection and save results.

CLI tool for running Bell NPA basis optimization sweeps.  The core
optimisation logic lives in :mod:`bell.bell_optimize`; this script
adds persistence (artifact management), resumability, and a CLI.

On-disk layout::

    results/bell_optimization_sweep/v1/<run_name>/
        meta.json      <- config, provenance, timestamps
        data.npz       <- run_idx, k, seed, best_value, elapsed_s,
                          n_obj_evals, mask_bits

Examples::

    python -m scripts.bell.optimization_sweep \\
        --m-A 2 --m-B 2 --d-A 2 --d-B 2 \\
        --operator chsh \\
        --start-level 1 --end-level 2 --method sa \\
        --ks 0 1 2 3 --seeds 42 43 44

    python -m scripts.bell.optimization_sweep \\
        --m-A 2 --m-B 2 --d-A 2 --d-B 2 \\
        --operator chsh \\
        --start-level 1 --end-level 3 --method pt \\
        --k-max 10 --num-seeds 5
"""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from artifact_manager import ArtifactManager, RunDir
from bell.bell_logic import (
    BellScenario,
    BellWord,
    _word_sort_key,
    chsh_operator,
)
from bell.bell_optimize import (
    BellOptimizationResult,
    build_bell_basis_sets,
    sweep_k_values,
)


ARTIFACT = "bell_optimization_sweep"


def _require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pandas is required for results_to_dataframe(). "
            "Install with `pip install pandas`."
        ) from e
    return pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unpack_masks(packed: np.ndarray, L: int) -> np.ndarray:
    """Unpack bit-packed masks back to boolean array."""
    if packed.size == 0:
        return np.array([], dtype=bool).reshape(0, L)
    unpacked = [np.unpackbits(row)[:L].astype(bool) for row in packed]
    return np.stack(unpacked, axis=0)


def _stable_bellword_hash(words: List[BellWord], *, n_chars: int = 16) -> str:
    """Fast stable hash for a list of BellWords."""
    h = hashlib.sha256()
    for w in words:
        for x, a in w.alice_seq:
            h.update(struct.pack("<ii", x, a))
        h.update(b"|")
        for y, b in w.bob_seq:
            h.update(struct.pack("<ii", y, b))
        h.update(b";")
    return h.hexdigest()[:n_chars]


def _default_name(
    scenario: BellScenario,
    operator_name: str,
    method: str,
    start_level: int,
    end_level: int,
    feedback: bool,
) -> str:
    parts = [
        f"mA{scenario.m_A}_mB{scenario.m_B}",
        f"dA{scenario.d_A}_dB{scenario.d_B}",
        operator_name,
        f"lv{start_level}to{end_level}",
        method,
    ]
    if feedback:
        parts.append("fb")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Bell operator registry
# ---------------------------------------------------------------------------

_OPERATOR_BUILDERS: Dict[str, Any] = {
    "chsh": chsh_operator,
}


def _build_operator(name: str, scenario: BellScenario) -> Dict[BellWord, float]:
    """Build a named Bell operator for the given scenario."""
    builder = _OPERATOR_BUILDERS.get(name)
    if builder is None:
        raise ValueError(
            f"Unknown operator: {name!r}. "
            f"Available: {', '.join(sorted(_OPERATOR_BUILDERS))}"
        )
    return builder(scenario)


# ---------------------------------------------------------------------------
# Data helpers (flat-table pattern)
# ---------------------------------------------------------------------------

def _load_existing_runs(run: RunDir, L: int) -> Tuple[Dict[Tuple[int, int], int], Dict[str, list]]:
    """Load existing runs from a RunDir.

    Returns:
        completed: dict mapping (k, seed) -> run_idx
        arrays: dict of lists (for appending new rows)
    """
    tbl = run.load_table()
    if not tbl:
        return {}, {}
    completed: Dict[Tuple[int, int], int] = {}
    ks = tbl["k"]
    seeds = tbl["seed"]
    run_idxs = tbl["run_idx"]
    for i in range(len(ks)):
        completed[(int(ks[i]), int(seeds[i]))] = int(run_idxs[i])
    arrays = {key: list(arr) for key, arr in tbl.items()}
    return completed, arrays


def _checkpoint(
    run: RunDir,
    run_idx: List[int],
    k: List[int],
    seed: List[int],
    best_value: List[float],
    elapsed_s: List[float],
    n_obj_evals: List[int],
    mask_bits_list: List[np.ndarray],
) -> None:
    """Atomically save all accumulated results."""
    mb = (
        np.stack(mask_bits_list, axis=0)
        if mask_bits_list
        else np.array([], dtype=np.uint8).reshape(0, 0)
    )
    run.save_table(
        run_idx=np.array(run_idx, dtype=np.int32),
        k=np.array(k, dtype=np.int32),
        seed=np.array(seed, dtype=np.int32),
        best_value=np.array(best_value, dtype=np.float64),
        elapsed_s=np.array(elapsed_s, dtype=np.float64),
        n_obj_evals=np.array(n_obj_evals, dtype=np.int32),
        mask_bits=mb,
    )


# ---------------------------------------------------------------------------
# Main compute and save
# ---------------------------------------------------------------------------

def compute_and_save(
    *,
    scenario: BellScenario,
    operator_name: str,
    start_level: int,
    end_level: int,
    method: str,
    method_params: Dict[str, Any],
    k_values: List[int],
    seeds: List[int],
    mosek_tol: float,
    out_root: Path,
    name: str,
    resume: bool,
    force: bool,
    verbose: bool,
    feedback: bool = False,
) -> str:
    """Run Bell optimization sweep and save results.

    Sets up the :class:`RunDir`, loads any existing checkpoint, then
    delegates the actual sweep to
    :func:`~bell.bell_optimize.sweep_k_values` with an ``on_result``
    callback that persists each new result atomically.

    Returns path string of the run directory.
    """
    starting_set, adding_set, _ = build_bell_basis_sets(
        scenario=scenario,
        start_level=start_level,
        end_level=end_level,
    )
    L = len(adding_set)

    bell_operator = _build_operator(operator_name, scenario)

    if verbose:
        print(
            f"Bell scenario: ({scenario.m_A},{scenario.m_B}), "
            f"d=({scenario.d_A},{scenario.d_B}), operator={operator_name}"
        )
        print(f"Starting set: {len(starting_set)}, Adding set: {L}")
        if feedback:
            print("Feedback mode: ON (warm-start chaining across k values)")

    adding_set_hash = _stable_bellword_hash(adding_set)

    config: Dict[str, Any] = {
        "scenario": {
            "m_A": scenario.m_A,
            "m_B": scenario.m_B,
            "d_A": scenario.d_A,
            "d_B": scenario.d_B,
        },
        "operator": operator_name,
        "start_level": int(start_level),
        "end_level": int(end_level),
        "starting_set_size": len(starting_set),
        "adding_set_size": L,
        "adding_set_hash": adding_set_hash,
        "method": method,
        "method_params": dict(sorted(method_params.items())),
        "mosek_tol": float(mosek_tol),
        "feedback": bool(feedback),
    }

    am = ArtifactManager(out_root)
    run = am.create_run(artifact=ARTIFACT, name=name, config=config)

    # ------------------------------------------------------------------
    # Load existing checkpoint → build existing_results dict
    # ------------------------------------------------------------------
    existing_results: Dict[Tuple[int, int], BellOptimizationResult] = {}
    if not force:
        completed, existing_arrays = (
            _load_existing_runs(run, L) if resume else ({}, {})
        )
        for (kv, s), run_idx in completed.items():
            pos = list(existing_arrays["run_idx"]).index(run_idx)
            mask: Optional[np.ndarray] = None
            if "mask_bits" in existing_arrays:
                packed = existing_arrays["mask_bits"][pos]
                if isinstance(packed, np.ndarray):
                    mask = np.unpackbits(packed)[:L].astype(np.int32)
            if mask is None:
                mask = np.zeros(L, dtype=np.int32)
            existing_results[(kv, s)] = BellOptimizationResult(
                best_value=float(existing_arrays["best_value"][pos]),
                best_indices=sorted(int(i) for i in np.flatnonzero(mask)),
                mask=mask,
                elapsed_s=float(existing_arrays["elapsed_s"][pos]),
                n_obj_evals=int(existing_arrays["n_obj_evals"][pos]),
                method=method,
                k=kv,
                seed=s,
            )

    if verbose:
        all_jobs = [(kv, s) for kv in sorted(k_values) for s in seeds]
        n_todo = sum(1 for j in all_jobs if j not in existing_results)
        print(
            f"Total jobs: {len(all_jobs)}, "
            f"Already done: {len(existing_results)}, "
            f"To compute: {n_todo}"
        )

    if not any(
        (kv, s) not in existing_results
        for kv in k_values for s in seeds
    ):
        if verbose:
            print("All jobs already completed. Nothing to do.")
        return str(run.path)

    # ------------------------------------------------------------------
    # Accumulator + checkpoint callback
    # ------------------------------------------------------------------
    _acc_run_idx: List[int] = []
    _acc_k: List[int] = []
    _acc_seed: List[int] = []
    _acc_best_value: List[float] = []
    _acc_elapsed_s: List[float] = []
    _acc_n_obj_evals: List[int] = []
    _acc_mask_bits: List[np.ndarray] = []

    if not force and resume:
        completed_loaded, arr = _load_existing_runs(run, L)
        if arr:
            _acc_run_idx = list(arr.get("run_idx", []))
            _acc_k = list(arr.get("k", []))
            _acc_seed = list(arr.get("seed", []))
            _acc_best_value = list(arr.get("best_value", []))
            _acc_elapsed_s = list(arr.get("elapsed_s", []))
            _acc_n_obj_evals = list(arr.get("n_obj_evals", []))
            _acc_mask_bits = list(arr.get("mask_bits", []))
    _next_run_idx = (max(int(x) for x in _acc_run_idx) + 1) if _acc_run_idx else 0

    def _on_result(kv: int, s: int, res: BellOptimizationResult) -> None:
        nonlocal _next_run_idx
        packed = np.packbits(res.mask.astype(np.uint8))
        _acc_run_idx.append(_next_run_idx)
        _acc_k.append(kv)
        _acc_seed.append(s)
        _acc_best_value.append(res.best_value)
        _acc_elapsed_s.append(res.elapsed_s)
        _acc_n_obj_evals.append(res.n_obj_evals)
        _acc_mask_bits.append(packed)
        _next_run_idx += 1
        _checkpoint(
            run, _acc_run_idx, _acc_k, _acc_seed,
            _acc_best_value, _acc_elapsed_s,
            _acc_n_obj_evals, _acc_mask_bits,
        )

    # ------------------------------------------------------------------
    # Delegate to sweep_k_values
    # ------------------------------------------------------------------
    run.update_meta(
        k_values_requested=sorted(k_values),
        seeds_requested=sorted(seeds),
        L=int(L),
    )

    try:
        results = sweep_k_values(
            scenario=scenario,
            bell_operator=bell_operator,
            starting_set=starting_set,
            adding_set=adding_set,
            k_values=k_values,
            method=method,
            method_params=method_params,
            mosek_tol=mosek_tol,
            seeds=seeds,
            feedback=feedback,
            existing_results=existing_results,
            on_result=_on_result,
            verbose=verbose,
        )
    finally:
        run.update_meta(
            k_values_present=sorted(set(int(x) for x in _acc_k)),
            seeds_present=sorted(set(int(x) for x in _acc_seed)),
            total_runs=len(_acc_run_idx),
        )

    n_computed = len(_acc_run_idx) - len(existing_results)
    if verbose:
        print(f"\nResults -> {run.path}")
        print(f"Total runs: {len(_acc_run_idx)} (computed this call: {n_computed})")

    return str(run.path)


# ---------------------------------------------------------------------------
# Result loading utilities
# ---------------------------------------------------------------------------

def load_optimization_results(
    run_dir: Path | str,
    unpack_masks: bool = True,
) -> Dict[str, Any]:
    """Load optimization results from a run directory.

    Returns dict with 'meta' and 'data'.
    """
    run_dir = Path(run_dir)
    am = ArtifactManager(run_dir.parents[2])
    run = am.open_run(artifact=ARTIFACT, name=run_dir.name)
    meta = run.load_meta()
    data = run.load_table()
    if unpack_masks and "mask_bits" in data:
        L = meta.get("L", meta.get("adding_set_size"))
        data["masks"] = _unpack_masks(data["mask_bits"], L)
        del data["mask_bits"]
    return {"meta": meta, "data": data}


def results_to_dataframe(run_dir: Path | str) -> "pd.DataFrame":
    """Load results as a pandas DataFrame (without masks)."""
    results = load_optimization_results(run_dir, unpack_masks=False)
    data = results["data"]
    pd = _require_pandas()
    return pd.DataFrame({
        "run_idx": data["run_idx"],
        "k": data["k"],
        "seed": data["seed"],
        "best_value": data["best_value"],
        "elapsed_s": data["elapsed_s"],
        "n_obj_evals": data["n_obj_evals"],
    })


def get_best_per_k(run_dir: Path | str) -> Dict[int, Dict[str, Any]]:
    """Get the best result for each k value."""
    results = load_optimization_results(run_dir, unpack_masks=True)
    data = results["data"]
    best_by_k: Dict[int, Dict[str, Any]] = {}
    for i in range(len(data["k"])):
        k = int(data["k"][i])
        val = float(data["best_value"][i])
        if k not in best_by_k or val > best_by_k[k]["best_value"]:
            best_by_k[k] = {
                "best_value": val,
                "seed": int(data["seed"][i]),
                "mask": data["masks"][i] if "masks" in data else None,
                "run_idx": int(data["run_idx"][i]),
            }
    return best_by_k


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Scenario parameters
    p.add_argument("--m-A", type=int, required=True,
                   help="Number of settings for Alice")
    p.add_argument("--m-B", type=int, required=True,
                   help="Number of settings for Bob")
    p.add_argument("--d-A", type=int, default=2,
                   help="Number of outcomes for Alice (default: 2)")
    p.add_argument("--d-B", type=int, default=2,
                   help="Number of outcomes for Bob (default: 2)")

    # Operator
    p.add_argument("--operator", type=str, default="chsh",
                   choices=list(_OPERATOR_BUILDERS.keys()),
                   help="Bell operator to optimise")

    # Basis parameters
    p.add_argument("--start-level", type=int, default=1,
                   help="NPA level for starting set")
    p.add_argument("--end-level", type=int, default=2,
                   help="NPA level for final set")

    # Optimization method
    p.add_argument("--method", choices=["sa", "pt", "bo", "rbm", "random"],
                   default="sa")

    # SA parameters
    p.add_argument("--sa-steps", type=int, default=100)
    p.add_argument("--sa-T-start", type=float, default=2.0)
    p.add_argument("--sa-alpha", type=float, default=0.95)

    # PT parameters
    p.add_argument("--pt-chains", type=int, default=0,
                   help="Number of chains (0 = auto)")
    p.add_argument("--pt-epochs", type=int, default=5)
    p.add_argument("--pt-steps-per-epoch", type=int, default=40)
    p.add_argument("--pt-T-min", type=float, default=0.1)
    p.add_argument("--pt-T-max", type=float, default=2.0)

    # BO parameters
    p.add_argument("--bo-beta", type=float, default=1.0)
    p.add_argument("--bo-n-init", type=int, default=20)
    p.add_argument("--bo-n-iter", type=int, default=50)
    p.add_argument("--bo-candidates-per-iter", type=int, default=100)

    # RBM parameters
    p.add_argument("--rbm-steps", type=int, default=100)

    # Feedback (warm-start chaining)
    p.add_argument("--feedback", action="store_true", default=False,
                   help="Chain k-values: best mask at k_i warm-starts k_{i+1}")

    # Sweep parameters
    k_group = p.add_mutually_exclusive_group(required=True)
    k_group.add_argument("--ks", nargs="+", type=int, help="Explicit k values")
    k_group.add_argument("--k-max", type=int, help="Sweep k from 0 to k-max")

    seed_group = p.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seeds", nargs="+", type=int, help="Explicit seeds")
    seed_group.add_argument("--num-seeds", type=int,
                            help="Number of seeds (starting from 42)")

    p.add_argument("--mosek-tol", type=float, default=1e-6)

    p.add_argument("--out-root", type=Path,
                   default=Path(__file__).resolve().parents[2] / "results")
    p.add_argument("--name", type=str, default=None,
                   help="Run name (default: auto-generated)")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--verbose", action="store_true", default=True)

    args = p.parse_args(argv)

    scenario = BellScenario(
        m_A=args.m_A, m_B=args.m_B,
        d_A=args.d_A, d_B=args.d_B,
    )

    # Build k values
    k_values = (
        sorted(set(args.ks))
        if args.ks is not None
        else list(range(args.k_max + 1))
    )

    # Build seeds
    seeds = (
        sorted(set(args.seeds))
        if args.seeds is not None
        else list(range(42, 42 + args.num_seeds))
    )

    # Build method params
    if args.method == "sa":
        method_params = {
            "steps": args.sa_steps,
            "T_start": args.sa_T_start,
            "alpha": args.sa_alpha,
        }
    elif args.method == "pt":
        pt_chains = None if int(args.pt_chains) <= 0 else int(args.pt_chains)
        method_params = {
            "num_chains": pt_chains,
            "num_epochs": args.pt_epochs,
            "steps_per_epoch": args.pt_steps_per_epoch,
            "T_min": args.pt_T_min,
            "T_max": args.pt_T_max,
        }
    elif args.method == "bo":
        method_params = {
            "beta": args.bo_beta,
            "n_init": args.bo_n_init,
            "n_iter": args.bo_n_iter,
            "candidates_per_iter": args.bo_candidates_per_iter,
        }
    elif args.method == "rbm":
        method_params = {"steps": args.rbm_steps}
    else:
        method_params = {}

    name = args.name or _default_name(
        scenario, args.operator, args.method,
        args.start_level, args.end_level,
        getattr(args, "feedback", False),
    )

    run_dir = compute_and_save(
        scenario=scenario,
        operator_name=args.operator,
        start_level=args.start_level,
        end_level=args.end_level,
        method=args.method,
        method_params=method_params,
        k_values=k_values,
        seeds=seeds,
        mosek_tol=args.mosek_tol,
        out_root=args.out_root,
        name=name,
        resume=args.resume,
        force=args.force,
        verbose=args.verbose,
        feedback=getattr(args, "feedback", False),
    )

    print(f"Results -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
