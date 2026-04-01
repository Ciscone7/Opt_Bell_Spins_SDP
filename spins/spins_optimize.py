"""
Optimization of spin NPA relaxation basis selection.

Provides model-agnostic tools for selecting the best subset of Pauli
monomials to add to a starting NPA basis, in order to tighten the SDP
relaxation bound on a spin-chain observable.

The main entry points are:

- :func:`optimize_ground_energy` — single (k, seed) run that tightens
  the ground-energy lower bound.
- :func:`sweep_k_values` — sweep over multiple k values and seeds with
  optional warm-start chaining (feedback), resume support, and
  per-result callbacks for atomic persistence.

These mirror the Bell-case API in :mod:`bell.bell_optimize`.

Example::

    from spins.models import heisenberg_hamiltonian_dict
    from spins.spins_optimize import optimize_ground_energy
    from spins.symmetry import SymmetryManager

    H = heisenberg_hamiltonian_dict(N=6, boundary="periodic")
    sym = SymmetryManager.default_for_heisenberg(N=6)

    result = optimize_ground_energy(
        N=6,
        hamiltonian_dict=H,
        symmetry_manager=sym,
        start_level=1,
        end_level=2,
        k=3,
        method="sa",
    )
    print(f"Best lower bound: {result.best_value:.6f}")
"""

from __future__ import annotations

import os
import random
import time
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np

from spins.basis_builder import generate_npa_basis
from spins.pauli_logic import PauliWord
from spins.spins_sdp import Sense, solve_pauli_relaxation
from spins.symmetry import SymmetryManager

# Type alias for operator dicts (PauliWord -> complex coefficient).
Operator = Dict[PauliWord, complex]
Method = Literal["sa", "pt", "bo", "rbm", "random"]


# ---------------------------------------------------------------------------
# Basis helpers
# ---------------------------------------------------------------------------

def build_npa_basis_sets(
    N: int,
    start_level: int,
    end_level: int,
) -> Tuple[List[PauliWord], List[PauliWord], List[PauliWord]]:
    """Return (starting_set, adding_set, final_set) for NPA levels.

    ``starting_set`` contains all words from NPA levels 0 through
    ``start_level``.  ``adding_set`` contains words from levels
    ``start_level + 1`` through ``end_level``.  ``final_set`` is the
    union.

    Args:
        N: Number of spin sites.
        start_level: NPA level for the fixed starting set.
        end_level: NPA level for the full candidate pool.

    Returns:
        Tuple of (starting_set, adding_set, final_set).
    """
    full_basis = generate_npa_basis(N=N, k=end_level)

    starting_set: List[PauliWord] = []
    for level_idx in range(min(start_level + 1, len(full_basis.levels))):
        starting_set.extend(full_basis.levels[level_idx])

    adding_set: List[PauliWord] = []
    for level_idx in range(start_level + 1, min(end_level + 1, len(full_basis.levels))):
        adding_set.extend(full_basis.levels[level_idx])

    final_set = full_basis.words

    return starting_set, adding_set, final_set


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

class SpinOptimizationObjective:
    """Objective function for spin basis optimisation.

    Given a binary selection of words from ``adding_set``, build the
    corresponding basis (``starting_set`` + selected words), solve the
    SDP relaxation, and return a value suitable for *minimisation* by
    external optimisers:

    - ``sense="min"`` → return ``-lb`` (minimising this tightens the
      lower bound).
    - ``sense="max"`` → return ``-ub`` (minimising this tightens the
      upper bound).

    Accepts **either** a dense 0/1 mask of length ``L`` **or** an
    iterable of selected indices (sparse representation).
    """

    def __init__(
        self,
        starting_set: List[PauliWord],
        adding_set: List[PauliWord],
        operator: Operator,
        symmetry_manager: SymmetryManager,
        sense: Sense = "min",
        mosek_tol: float = 1e-9,
    ) -> None:
        self.starting_set = starting_set
        self.adding_set = adding_set
        self.operator = operator
        self.symmetry_manager = symmetry_manager
        self.sense: Sense = sense
        self.mosek_tol = mosek_tol

    def __call__(self, mask_or_indices: Any) -> float:
        if isinstance(mask_or_indices, np.ndarray):
            chosen = [self.adding_set[int(i)] for i in np.flatnonzero(mask_or_indices)]
        else:
            chosen = [self.adding_set[int(i)] for i in mask_or_indices]

        basis = self.starting_set + chosen
        val = solve_pauli_relaxation(
            basis,
            self.operator,
            symmetry_manager=self.symmetry_manager,
            sense=self.sense,
            mosek_tol=self.mosek_tol,
            verbose=False,
        )
        # External optimisers *minimise*.
        # sense="min": we want the highest lb → minimise -lb.
        # sense="max": we want the lowest ub  → minimise -ub.
        return -float(val)


def make_spin_objective(
    starting_set: List[PauliWord],
    adding_set: List[PauliWord],
    operator: Operator,
    symmetry_manager: SymmetryManager,
    sense: Sense = "min",
    mosek_tol: float = 1e-9,
) -> SpinOptimizationObjective:
    """Create an objective function for spin basis optimisation.

    This is a convenience wrapper around :class:`SpinOptimizationObjective`.
    """
    return SpinOptimizationObjective(
        starting_set=starting_set,
        adding_set=adding_set,
        operator=operator,
        symmetry_manager=symmetry_manager,
        sense=sense,
        mosek_tol=mosek_tol,
    )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SpinOptimizationResult:
    """Result of a single spin basis-selection optimisation run.

    Attributes:
        best_value: The best SDP objective value found.  For the
            ground-energy case this is a lower bound (higher = tighter).
        best_indices: Indices into the ``adding_set`` that were selected.
        mask: Dense 0/1 mask of length *L* (the adding-set size).
        elapsed_s: Wall-clock time in seconds.
        n_obj_evals: Number of SDP objective evaluations.
        method: Name of the optimisation method used.
        k: Number of basis elements selected.
        seed: Random seed used.
        raw: Full result dict from the underlying optimiser (optional).
    """
    best_value: float
    best_indices: List[int]
    mask: np.ndarray
    elapsed_s: float
    n_obj_evals: int
    method: str
    k: int
    seed: int
    raw: Optional[Dict[str, Any]] = None


# Callback type: called after each (k, seed) job in sweep_k_values.
OnResultCallback = Callable[[int, int, SpinOptimizationResult], None]


# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

def _default_method_params(method: Method) -> Dict[str, Any]:
    """Sensible defaults for each optimisation method."""
    if method == "sa":
        return {"steps": 100, "T_start": 2.0, "alpha": 0.95}
    elif method == "pt":
        return {
            "num_chains": None,   # auto = num CPUs
            "num_epochs": 10,
            "steps_per_epoch": 50,
            "T_min": 0.01,
            "T_max": 2.0,
        }
    elif method == "bo":
        return {
            "beta": 1.0,
            "n_init": 20,
            "n_iter": 50,
            "candidates_per_iter": 100,
        }
    elif method == "rbm":
        return {"steps": 100}
    elif method == "random":
        return {}
    else:
        return {}


# ---------------------------------------------------------------------------
# Single-run dispatcher
# ---------------------------------------------------------------------------

def run_single_optimization(
    obj_func: SpinOptimizationObjective,
    L: int,
    k: int,
    seed: int,
    method: Method,
    method_params: Dict[str, Any],
    initial_guess: Optional[np.ndarray] = None,
) -> SpinOptimizationResult:
    """Run a single optimisation and return a structured result.

    Args:
        obj_func: Spin objective (mask/indices → float).
        L: Length of adding_set.
        k: Number of words to select (Hamming weight).
        seed: Random seed.
        method: One of ``"sa"``, ``"pt"``, ``"bo"``, ``"rbm"``, ``"random"``.
        method_params: Method-specific hyper-parameters.
        initial_guess: Optional dense mask from a previous k for warm-starting.

    Returns:
        :class:`SpinOptimizationResult`.
    """
    t0 = time.perf_counter()

    if method == "sa":
        from optimize.montecarlo import simulated_annealing

        result = simulated_annealing(
            obj_func=obj_func, N=L, k=k,
            initial_guess=initial_guess,
            steps=method_params.get("steps", 100),
            T_start=method_params.get("T_start", 2.0),
            alpha=method_params.get("alpha", 0.95),
            record_history=False, seed=seed, verbose=False,
            obj_uses_indices=True,
        )
        n_obj_evals = method_params.get("steps", 100) + 1
        best_loss = float(result["best"]["value"])
        best_mask = np.asarray(result["best"]["selection"], dtype=np.int32)
        best_indices = sorted(int(i) for i in np.flatnonzero(best_mask))
        raw = result

    elif method == "pt":
        from optimize.montecarlo import parallel_tempering

        pt_chains = method_params.get("num_chains", None)
        result = parallel_tempering(
            obj_func=obj_func, N=L, k=k,
            num_chains=pt_chains,
            num_epochs=method_params.get("num_epochs", 10),
            steps_per_epoch=method_params.get("steps_per_epoch", 50),
            T_min=method_params.get("T_min", 0.01),
            T_max=method_params.get("T_max", 2.0),
            initial_guess=initial_guess,
            seed=seed, verbose=False, obj_uses_indices=True,
        )
        n_chains_eff = (
            int(pt_chains) if pt_chains and int(pt_chains) > 0
            else (os.cpu_count() or 1)
        )
        n_obj_evals = (
            n_chains_eff
            * method_params.get("num_epochs", 10)
            * method_params.get("steps_per_epoch", 50)
            + n_chains_eff
        )
        best_loss = float(result["best"]["value"])
        best_mask = np.asarray(result["best"]["selection"], dtype=np.int32)
        best_indices = sorted(int(i) for i in np.flatnonzero(best_mask))
        raw = result

    elif method == "bo":
        try:
            from optimize.bayesian import bayesian as bayesian_optimization
        except ImportError as e:
            raise ImportError("Bayesian optimization requires scikit-learn.") from e

        n_init = int(method_params.get("n_init", 20))
        n_iter = int(method_params.get("n_iter", 50))
        bo_result = bayesian_optimization(
            obj_func=obj_func, N=L, k=k,
            beta=float(method_params.get("beta", 1.0)),
            n_init=n_init, n_iter=n_iter,
            candidates_per_iter=int(method_params.get("candidates_per_iter", 100)),
            previous_best=initial_guess,
            seed=seed, verbose=False, obj_uses_indices=True,
        )
        n_obj_evals = n_init + n_iter
        best_loss = float(bo_result["best_value"])
        best_mask = np.asarray(bo_result["best_selection"], dtype=np.int32)
        best_indices = list(map(int, bo_result["best_indices"]))
        raw = bo_result

    elif method == "rbm":
        try:
            from optimize.rbm import RBMTrainer
        except ImportError as e:
            raise ImportError("RBM optimization requires JAX, Equinox, and Optax.") from e

        rbm_steps = int(method_params.get("steps", 100))
        _raw_obj = obj_func

        def _numpy_obj(v: Any) -> float:
            return _raw_obj(np.asarray(v))

        # Adapt initial_guess to exactly k ones (feedback may pass a mask with k-1 ones)
        if initial_guess is not None:
            ones = list(np.flatnonzero(initial_guess))
            if len(ones) > k:
                ones = ones[:k]
            elif len(ones) < k:
                zeros = [i for i in range(L) if i not in set(ones)]
                ones += zeros[:k - len(ones)]
            rbm_initial = np.zeros(L, dtype=np.int32)
            rbm_initial[ones] = 1
        else:
            rbm_initial = None

        trainer = RBMTrainer(
            obj_func=_numpy_obj, N=L, hamming_weight=k,
            steps=rbm_steps,
            seed=seed if seed is not None else 42,
            initial_guess=rbm_initial,
        )
        trainer.train(num_steps=rbm_steps, verbose=False)

        best_mask = np.asarray(trainer.current_vec, dtype=np.int32)
        best_loss = float(trainer.current_cost)
        best_indices = sorted(int(i) for i in np.flatnonzero(best_mask))
        n_obj_evals = rbm_steps + 1
        raw = None

    elif method == "random":
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        mask = np.zeros(L, dtype=np.int32)
        if 0 < k <= L:
            mask[random.sample(range(L), k)] = 1

        loss = obj_func(mask)
        best_loss = float(loss)
        best_mask = mask
        best_indices = sorted(int(i) for i in np.flatnonzero(mask))
        n_obj_evals = 1
        raw = None

    else:
        raise ValueError(f"Unknown method: {method!r}")

    elapsed = time.perf_counter() - t0

    return SpinOptimizationResult(
        best_value=-best_loss,
        best_indices=best_indices,
        mask=best_mask,
        elapsed_s=elapsed,
        n_obj_evals=n_obj_evals,
        method=method,
        k=k,
        seed=seed,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------

def _resolve_basis_sets(
    N: int,
    start_level: int,
    end_level: int,
    starting_set: Optional[List[PauliWord]],
    adding_set: Optional[List[PauliWord]],
) -> Tuple[List[PauliWord], List[PauliWord]]:
    """Build starting/adding sets from NPA levels if not provided."""
    if starting_set is None or adding_set is None:
        s_set, a_set, _ = build_npa_basis_sets(N, start_level, end_level)
        if starting_set is None:
            starting_set = s_set
        if adding_set is None:
            adding_set = a_set
    return starting_set, adding_set


def optimize_ground_energy(
    N: int,
    hamiltonian_dict: Operator,
    symmetry_manager: SymmetryManager,
    *,
    start_level: int = 1,
    end_level: int = 2,
    starting_set: Optional[List[PauliWord]] = None,
    adding_set: Optional[List[PauliWord]] = None,
    k: int,
    method: Method = "sa",
    method_params: Optional[Dict[str, Any]] = None,
    mosek_tol: float = 1e-9,
    seed: int = 42,
    initial_guess: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> SpinOptimizationResult:
    """Tighten the ground-energy lower bound via basis selection.

    Builds a :class:`SpinOptimizationObjective`
    and delegates to :func:`run_single_optimization`.

    Example::

        from spins.models import heisenberg_hamiltonian_dict
        from spins.spins_optimize import optimize_ground_energy
        from spins.symmetry import SymmetryManager

        H = heisenberg_hamiltonian_dict(N=6, boundary="periodic")
        sym = SymmetryManager.default_for_heisenberg(N=6)

        result = optimize_ground_energy(
            N=6, hamiltonian_dict=H, symmetry_manager=sym,
            start_level=1, end_level=2,
            k=3, method="sa",
        )
        print(f"Best lower bound: {result.best_value:.6f}")
    """
    starting_set, adding_set = _resolve_basis_sets(
        N, start_level, end_level, starting_set, adding_set,
    )

    L = len(adding_set)
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if k > L:
        raise ValueError(
            f"k={k} exceeds adding_set size {L}. "
            f"Use k <= {L} or increase end_level."
        )

    if method_params is None:
        method_params = _default_method_params(method)

    if verbose:
        print(f"Spin optimisation: N={N}")
        print(f"  Starting set: {len(starting_set)} words, "
              f"Adding set: {L} words, k={k}")
        print(f"  Method: {method}")

    obj_func = make_spin_objective(
        starting_set=starting_set,
        adding_set=adding_set,
        operator=hamiltonian_dict,
        symmetry_manager=symmetry_manager,
        sense="min",
        mosek_tol=mosek_tol,
    )

    result = run_single_optimization(
        obj_func=obj_func,
        L=L,
        k=k,
        seed=seed,
        method=method,
        method_params=method_params,
        initial_guess=initial_guess,
    )

    if verbose:
        print(f"  Result: SDP lower bound = {result.best_value:.6f}, "
              f"time = {result.elapsed_s:.2f}s, "
              f"evals = {result.n_obj_evals}")

    return result


# Keep old name as an alias for backward compatibility.
optimize_spin_relaxation = optimize_ground_energy


# ---------------------------------------------------------------------------
# Sweep over k values
# ---------------------------------------------------------------------------

def sweep_k_values(
    N: int,
    operator: Operator,
    symmetry_manager: SymmetryManager,
    *,
    start_level: int = 1,
    end_level: int = 2,
    starting_set: Optional[List[PauliWord]] = None,
    adding_set: Optional[List[PauliWord]] = None,
    k_values: List[int],
    method: Method = "sa",
    method_params: Optional[Dict[str, Any]] = None,
    mosek_tol: float = 1e-9,
    seeds: Optional[List[int]] = None,
    feedback: bool = False,
    existing_results: Optional[Dict[Tuple[int, int], SpinOptimizationResult]] = None,
    on_result: Optional[OnResultCallback] = None,
    verbose: bool = True,
) -> Dict[Tuple[int, int], SpinOptimizationResult]:
    """Run ground-energy optimisation across multiple *k* values and seeds.

    Builds a single :class:`SpinOptimizationObjective`
    and runs :func:`run_single_optimization` for every ``(k, seed)``
    combination.

    Args:
        N: Number of spin sites.
        operator: Pauli-operator dictionary for the Hamiltonian.
        symmetry_manager: Symmetry configuration.
        start_level, end_level, starting_set, adding_set, method,
        method_params, mosek_tol:
            Same as :func:`optimize_ground_energy`.
        k_values: List of k values to sweep.
        seeds: List of random seeds (one run per seed per k).
            Defaults to ``[42]``.
        feedback: If ``True``, chain k values per seed: the best mask
            at ``k_i`` is passed as ``initial_guess`` to ``k_{i+1}``.
        existing_results: Already-completed ``(k, seed)`` results to
            skip (e.g. loaded from a previous checkpoint).  Their masks
            are still used for feedback chaining when ``feedback=True``.
        on_result: Optional callback invoked after every *newly*
            completed ``(k, seed)`` job.  Signature:
            ``on_result(k, seed, result) -> None``.
            Use this to persist partial results (e.g. atomic checkpoint
            via a :class:`RunDir`).
        verbose: Print progress.

    Returns:
        Dict mapping ``(k, seed)`` to :class:`SpinOptimizationResult`,
        including both newly computed and ``existing_results``.

    Example::

        from spins.models import heisenberg_hamiltonian_dict
        from spins.spins_optimize import sweep_k_values
        from spins.symmetry import SymmetryManager

        H = heisenberg_hamiltonian_dict(N=6, boundary="periodic")
        sym = SymmetryManager.default_for_heisenberg(N=6)

        results = sweep_k_values(
            N=6, operator=H, symmetry_manager=sym,
            k_values=[0, 1, 2, 3], seeds=[42, 43],
            method="sa", feedback=True,
        )
        for (k, seed), res in sorted(results.items()):
            print(f"  k={k}, seed={seed}: {res.best_value:.6f}")
    """
    # Build basis sets once
    starting_set, adding_set = _resolve_basis_sets(
        N, start_level, end_level, starting_set, adding_set,
    )

    L = len(adding_set)
    if seeds is None:
        seeds = [42]
    if method_params is None:
        method_params = _default_method_params(method)

    # Seed results with anything already done
    results: Dict[Tuple[int, int], SpinOptimizationResult] = {}
    if existing_results:
        results.update(existing_results)

    # Build objective once — shared across all runs (ground-energy: sense="min")
    obj_func = make_spin_objective(
        starting_set=starting_set,
        adding_set=adding_set,
        operator=operator,
        symmetry_manager=symmetry_manager,
        sense="min",
        mosek_tol=mosek_tol,
    )

    sorted_ks = sorted(k_values)
    todo = [(kv, s) for kv in sorted_ks for s in seeds if (kv, s) not in results]

    pbar = tqdm(total=len(todo), desc="Spin optimisation sweep", disable=not verbose)

    def _record(kv: int, s: int, res: SpinOptimizationResult) -> None:
        results[(kv, s)] = res
        if on_result is not None:
            on_result(kv, s, res)

    if feedback:
        for s in seeds:
            prev_mask: Optional[np.ndarray] = None
            for kv in sorted_ks:
                if (kv, s) in results:
                    # Already done — just carry the mask for chaining
                    prev_mask = results[(kv, s)].mask
                    continue
                pbar.set_postfix({"k": kv, "seed": s})
                res = run_single_optimization(
                    obj_func=obj_func, L=L, k=kv, seed=s,
                    method=method, method_params=method_params,
                    initial_guess=prev_mask,
                )
                _record(kv, s, res)
                prev_mask = res.mask
                pbar.update(1)
    else:
        for kv in sorted_ks:
            for s in seeds:
                if (kv, s) in results:
                    continue
                pbar.set_postfix({"k": kv, "seed": s})
                res = run_single_optimization(
                    obj_func=obj_func, L=L, k=kv, seed=s,
                    method=method, method_params=method_params,
                )
                _record(kv, s, res)
                pbar.update(1)

    pbar.close()

    if verbose:
        best_per_k: Dict[int, float] = {}
        for (kv, _s), res in results.items():
            if kv not in best_per_k or res.best_value > best_per_k[kv]:
                best_per_k[kv] = res.best_value
        print("\nBest SDP lower bound per k:")
        for kv in sorted_ks:
            if kv in best_per_k:
                print(f"  k={kv}: {best_per_k[kv]:.6f}")

    return results
