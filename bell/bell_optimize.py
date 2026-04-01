"""
Optimization of Bell NPA relaxation basis selection.

Mirrors the spin-case optimization in :mod:`spins.spins_optimize` but
adapted for the Bell projector algebra defined in :mod:`bell.bell_logic`.

The main entry points are:

- :func:`optimize_bell_relaxation` — single ``(k, seed)`` run that
  tightens (or loosens) the SDP relaxation of a Bell operator.
- :func:`sweep_k_values` — sweep over multiple *k* values and seeds
  with optional warm-start chaining (feedback), resume support, and
  per-result callbacks for atomic persistence.
"""

from __future__ import annotations

import os
import random
import time
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from bell.bell_logic import (
    BellBasis,
    BellOperator,
    BellScenario,
    BellWord,
    Sense,
    compile_moment_matrix_rep,
    expand_bell_operator,
    generate_npa_basis,
    _word_sort_key,
)
from bell.bell_sdp import (
    build_bell_sdp,
)


# ---------------------------------------------------------------------------
# Basis helpers
# ---------------------------------------------------------------------------

def build_bell_basis_sets(
    scenario: BellScenario,
    start_level: int,
    end_level: int,
) -> Tuple[List[BellWord], List[BellWord], List[BellWord]]:
    """Return (starting_set, adding_set, final_set) for Bell NPA levels.

    ``starting_set`` contains all words from NPA levels 0 through
    ``start_level``.  ``adding_set`` contains all words from levels
    ``start_level + 1`` through ``end_level``.  ``final_set`` is the
    union.

    Args:
        scenario: Bell scenario description.
        start_level: NPA level for the fixed starting set.
        end_level: NPA level for the full candidate pool.

    Returns:
        Tuple of (starting_set, adding_set, final_set).
    """
    full_basis = generate_npa_basis(scenario, end_level)

    starting_set: List[BellWord] = []
    for level_idx in range(min(start_level + 1, len(full_basis.levels))):
        starting_set.extend(full_basis.levels[level_idx])

    adding_set: List[BellWord] = []
    for level_idx in range(start_level + 1, min(end_level + 1, len(full_basis.levels))):
        adding_set.extend(full_basis.levels[level_idx])

    final_set = full_basis.words

    return starting_set, adding_set, final_set


def _bell_basis_from_words(
    scenario: BellScenario,
    words: List[BellWord],
) -> BellBasis:
    """Build a minimal :class:`BellBasis` wrapper from an explicit word list.

    This avoids a full NPA-level BFS — we only need the word list, an index
    map, and a dummy ``min_len`` (set to 0 for every word since we don't
    need shortest-path information for the SDP).
    """
    words_sorted = sorted(words, key=_word_sort_key)
    index = {w: i for i, w in enumerate(words_sorted)}
    min_len = {w: 0 for w in words_sorted}

    return BellBasis(
        scenario=scenario,
        k=-1,  # sentinel: not from standard NPA generation
        words=words_sorted,
        levels=[words_sorted],
        min_len=min_len,
        index=index,
    )


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

class BellOptimizationObjective:
    """Objective function for Bell basis optimisation.

    Given a binary selection of words from ``adding_set``, build the
    corresponding basis (``starting_set`` + selected words), compile
    the Bell SDP, and solve it.

    The return value is ``-sdp_value`` when ``sense="max"`` (so that
    external *minimisers* maximise the Bell violation) and ``+sdp_value``
    when ``sense="min"``.

    Accepts **either** a dense 0/1 mask of length ``L`` **or** an iterable
    of selected indices (sparse representation).  The sparse path avoids
    an O(L) scan for every evaluation — important for large candidate pools.
    """

    def __init__(
        self,
        scenario: BellScenario,
        starting_set: List[BellWord],
        adding_set: List[BellWord],
        bell_operator: BellOperator,
        *,
        sense: Sense = "max",
        mosek_tol: float = 1e-6,
    ) -> None:
        self.scenario = scenario
        self.starting_set = starting_set
        self.adding_set = adding_set
        self.bell_operator = bell_operator
        self.sense = sense
        self.mosek_tol = mosek_tol

        # Pre-expand the operator once (completeness substitution)
        self._expanded_op = expand_bell_operator(bell_operator, scenario)

    def __call__(self, mask_or_indices: Any) -> float:
        # Determine selected words
        if isinstance(mask_or_indices, np.ndarray) and mask_or_indices.shape[0] == len(self.adding_set):
            chosen_indices = np.flatnonzero(mask_or_indices)
            chosen = [self.adding_set[int(i)] for i in chosen_indices]
        else:
            # Iterable of indices (sparse representation)
            chosen = [self.adding_set[int(i)] for i in mask_or_indices]

        basis_words = self.starting_set + chosen

        # Build a minimal BellBasis and compile moment matrix
        bell_basis = _bell_basis_from_words(self.scenario, basis_words)
        rep = compile_moment_matrix_rep(bell_basis)

        # Build and solve the SDP
        sdp = build_bell_sdp(rep, self._expanded_op, sense=self.sense)
        problem = sdp.problem

        mosek_params = {
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": self.mosek_tol,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": self.mosek_tol,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": self.mosek_tol,
        }
        problem.solve(solver="MOSEK", verbose=False, mosek_params=mosek_params)

        val = float(problem.value) if problem.value is not None else float("inf")

        # External optimisers are minimisers.
        # For sense="max" (e.g. maximise CHSH violation), minimise -val.
        # For sense="min", minimise val directly.
        if self.sense == "max":
            return -val
        return val


def make_bell_objective(
    scenario: BellScenario,
    starting_set: List[BellWord],
    adding_set: List[BellWord],
    bell_operator: BellOperator,
    *,
    sense: Sense = "max",
    mosek_tol: float = 1e-6,
) -> BellOptimizationObjective:
    """Create an objective function for Bell basis optimisation."""
    return BellOptimizationObjective(
        scenario=scenario,
        starting_set=starting_set,
        adding_set=adding_set,
        bell_operator=bell_operator,
        sense=sense,
        mosek_tol=mosek_tol,
    )


# ---------------------------------------------------------------------------
# Single-run dispatcher
# ---------------------------------------------------------------------------

Method = Literal["sa", "pt", "bo", "rbm", "random"]


@dataclass
class BellOptimizationResult:
    """Result of a single Bell optimisation run.

    Attributes:
        best_value: The best SDP objective value found.
        best_indices: Indices into the ``adding_set`` that were selected.
        mask: Dense 0/1 mask of length *L* (the adding-set size).
        elapsed_s: Wall-clock time in seconds.
        n_obj_evals: Number of SDP objective evaluations.
        method: Name of the optimisation method used.
        k: Number of basis elements selected.
        seed: Random seed used.
        raw: Full result dict from the underlying optimiser (optional).
    """
    best_value: float          # SDP value (positive = tighter bound)
    best_indices: List[int]    # indices into adding_set
    mask: np.ndarray           # dense 0/1 mask of length L
    elapsed_s: float
    n_obj_evals: int
    method: str
    k: int
    seed: int
    raw: Optional[Dict[str, Any]] = None  # full result dict from the optimizer


# Callback type: called after each (k, seed) job in sweep_k_values.
OnResultCallback = Callable[[int, int, BellOptimizationResult], None]


def run_single_optimization(
    obj_func: BellOptimizationObjective,
    L: int,
    k: int,
    seed: int,
    method: Method,
    method_params: Dict[str, Any],
    initial_guess: Optional[np.ndarray] = None,
) -> BellOptimizationResult:
    """Run a single optimisation and return a structured result.

    Args:
        obj_func: Bell objective (mask/indices -> float).
        L: Length of adding_set.
        k: Number of words to select (Hamming weight).
        seed: Random seed.
        method: One of ``"sa"``, ``"pt"``, ``"bo"``, ``"rbm"``, ``"random"``.
        method_params: Method-specific hyper-parameters (see below).
        initial_guess: Optional dense mask from a previous k for warm-starting.

    Returns:
        :class:`BellOptimizationResult`.
    """

    t0 = time.perf_counter()

    if method == "sa":
        from optimize.montecarlo import simulated_annealing

        result = simulated_annealing(
            obj_func=obj_func,
            N=L,
            k=k,
            initial_guess=initial_guess,
            steps=method_params.get("steps", 100),
            T_start=method_params.get("T_start", 2.0),
            alpha=method_params.get("alpha", 0.95),
            record_history=False,
            seed=seed,
            verbose=False,
            return_selection=True,
            obj_uses_indices=True,
        )
        n_obj_evals = method_params.get("steps", 100) + 1
        best_loss = float(result["best"]["value"])
        best_mask = np.asarray(result["best"]["selection"], dtype=np.int32)
        best_indices = sorted(int(i) for i in result["best"].get("indices", np.flatnonzero(best_mask)))
        raw = result

    elif method == "pt":
        from optimize.montecarlo import parallel_tempering

        pt_chains = method_params.get("num_chains", None)
        result = parallel_tempering(
            obj_func=obj_func,
            N=L,
            k=k,
            num_chains=pt_chains,
            num_epochs=method_params.get("num_epochs", 10),
            steps_per_epoch=method_params.get("steps_per_epoch", 50),
            T_min=method_params.get("T_min", 0.01),
            T_max=method_params.get("T_max", 2.0),
            initial_guess=initial_guess,
            seed=seed,
            verbose=False,
            obj_uses_indices=True,
        )

        n_chains_eff = pt_chains if pt_chains and int(pt_chains) > 0 else (os.cpu_count() or 1)
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
        from optimize.bayesian import bayesian as bayesian_optimization

        n_init = int(method_params.get("n_init", 20))
        n_iter = int(method_params.get("n_iter", 50))
        candidates_per_iter = int(method_params.get("candidates_per_iter", 100))
        beta = float(method_params.get("beta", 1.0))

        bo_result = bayesian_optimization(
            obj_func=obj_func,
            N=L,
            k=k,
            beta=beta,
            n_init=n_init,
            n_iter=n_iter,
            candidates_per_iter=candidates_per_iter,
            previous_best=initial_guess,
            seed=seed,
            verbose=False,
            obj_uses_indices=True,
        )
        n_obj_evals = n_init + n_iter
        best_loss = float(bo_result["best_value"])
        best_mask = np.asarray(bo_result["best_selection"], dtype=np.int32)
        best_indices = list(map(int, bo_result["best_indices"]))
        raw = bo_result

    elif method == "rbm":
        from optimize.rbm import RBMTrainer

        rbm_steps = int(method_params.get("steps", 100))

        # Wrap so JAX arrays → numpy before reaching BellOptimizationObjective
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
            obj_func=_numpy_obj,
            N=L,
            hamming_weight=k,
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
            chosen = random.sample(range(L), k)
            mask[chosen] = 1

        loss = obj_func(mask)
        best_loss = float(loss)
        best_mask = mask
        best_indices = sorted(int(i) for i in np.flatnonzero(mask))
        n_obj_evals = 1
        raw = None

    else:
        raise ValueError(f"Unknown method: {method!r}")

    elapsed = time.perf_counter() - t0

    # Convert internal loss back to SDP value.
    # For sense="max" the objective returns -val, so best_sdp_val = -best_loss.
    # For sense="min" the objective returns +val, so best_sdp_val = best_loss.
    if obj_func.sense == "max":
        best_sdp_val = -best_loss
    else:
        best_sdp_val = best_loss

    return BellOptimizationResult(
        best_value=best_sdp_val,
        best_indices=best_indices,
        mask=best_mask,
        elapsed_s=elapsed,
        n_obj_evals=n_obj_evals,
        method=method,
        k=k,
        seed=seed,
        raw=raw,
    )


def optimize_bell_relaxation(
    scenario: BellScenario,
    bell_operator: BellOperator,
    *,
    start_level: int = 1,
    end_level: int = 2,
    starting_set: Optional[List[BellWord]] = None,
    adding_set: Optional[List[BellWord]] = None,
    k: int,
    method: Method = "sa",
    method_params: Optional[Dict[str, Any]] = None,
    sense: Sense = "max",
    mosek_tol: float = 1e-6,
    seed: int = 42,
    initial_guess: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> BellOptimizationResult:
    """Optimise the Bell NPA relaxation by selecting the best *k* words.

    This is the main entry point for Bell basis optimisation.  It creates
    the objective function, delegates to the chosen optimizer, and returns
    a structured result.

    Args:
        scenario: Bell scenario (number of settings/outcomes).
        bell_operator: The Bell operator to optimise (e.g. from
            :func:`chsh_operator`).  Will be expanded (completeness
            substitution) internally.
        start_level: NPA level for the fixed starting set (used only when
            ``starting_set`` / ``adding_set`` are not provided).
        end_level: NPA level for the candidate pool (used only when
            ``starting_set`` / ``adding_set`` are not provided).
        starting_set: Explicit list of words always included.  If ``None``,
            built from ``start_level``.
        adding_set: Explicit candidate pool.  If ``None``, built from
            ``end_level`` minus ``starting_set``.
        k: Number of words to select from ``adding_set`` (Hamming weight).
        method: ``"sa"`` (simulated annealing), ``"pt"`` (parallel
            tempering), ``"bo"`` (Bayesian optimisation), ``"rbm"``
            (RBM REINFORCE), or ``"random"`` (baseline).
        method_params: Hyper-parameters for the chosen method.  Defaults
            are sensible for moderate-sized problems.
        sense: ``"max"`` (maximise violation) or ``"min"``.
        mosek_tol: MOSEK solver tolerance.
        seed: Random seed.
        initial_guess: Optional dense 0/1 mask (length ``L``) from a
            previous run for warm-starting.
        verbose: Print summary when done.

    Returns:
        :class:`BellOptimizationResult` with the best SDP value, selected
        indices, timing, and evaluation count.

    Example::

        from spins_sdp.bell import BellScenario, chsh_operator
        from spins_sdp.bell_optimize import optimize_bell_relaxation

        scenario = BellScenario.symmetric(m=2, d=2)
        op = chsh_operator(scenario)

        result = optimize_bell_relaxation(
            scenario, op,
            start_level=1, end_level=2,
            k=3, method="sa",
        )
        print(f"Best CHSH bound: {result.best_value:.6f}")
    """
    # ---- Build basis sets if not provided ----
    if starting_set is None or adding_set is None:
        s_set, a_set, _ = build_bell_basis_sets(scenario, start_level, end_level)
        if starting_set is None:
            starting_set = s_set
        if adding_set is None:
            adding_set = a_set

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
        print(f"Bell optimisation: scenario=({scenario.m_A},{scenario.m_B}), "
              f"d=({scenario.d_A},{scenario.d_B})")
        print(f"  Starting set: {len(starting_set)} words, "
              f"Adding set: {L} words, k={k}")
        print(f"  Method: {method}, sense={sense}")

    # ---- Build objective ----
    obj_func = make_bell_objective(
        scenario=scenario,
        starting_set=starting_set,
        adding_set=adding_set,
        bell_operator=bell_operator,
        sense=sense,
        mosek_tol=mosek_tol,
    )

    # ---- Run optimisation ----
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
        print(f"  Result: SDP value = {result.best_value:.6f}, "
              f"time = {result.elapsed_s:.2f}s, "
              f"evals = {result.n_obj_evals}")

    return result


def sweep_k_values(
    scenario: BellScenario,
    bell_operator: BellOperator,
    *,
    start_level: int = 1,
    end_level: int = 2,
    starting_set: Optional[List[BellWord]] = None,
    adding_set: Optional[List[BellWord]] = None,
    k_values: List[int],
    method: Method = "sa",
    method_params: Optional[Dict[str, Any]] = None,
    mosek_tol: float = 1e-6,
    seeds: Optional[List[int]] = None,
    feedback: bool = False,
    existing_results: Optional[Dict[Tuple[int, int], BellOptimizationResult]] = None,
    on_result: Optional[OnResultCallback] = None,
    verbose: bool = True,
) -> Dict[Tuple[int, int], BellOptimizationResult]:
    """Run optimisation across multiple *k* values and seeds.

    Args:
        scenario, bell_operator, start_level, end_level, starting_set,
        adding_set, method, method_params, mosek_tol:
            Same as :func:`optimize_bell_relaxation`.
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
        Dict mapping ``(k, seed)`` to :class:`BellOptimizationResult`,
        including both newly computed and ``existing_results``.
    """
    # Build basis sets once
    if starting_set is None or adding_set is None:
        s_set, a_set, _ = build_bell_basis_sets(scenario, start_level, end_level)
        if starting_set is None:
            starting_set = s_set
        if adding_set is None:
            adding_set = a_set

    L = len(adding_set)
    if seeds is None:
        seeds = [42]
    if method_params is None:
        method_params = _default_method_params(method)

    # Seed results with anything already done
    results: Dict[Tuple[int, int], BellOptimizationResult] = {}
    if existing_results:
        results.update(existing_results)

    # Build objective once — shared across all runs (Bell: sense="max")
    obj_func = make_bell_objective(
        scenario=scenario,
        starting_set=starting_set,
        adding_set=adding_set,
        bell_operator=bell_operator,
        sense="max",
        mosek_tol=mosek_tol,
    )

    sorted_ks = sorted(k_values)
    todo = [(kv, s) for kv in sorted_ks for s in seeds if (kv, s) not in results]

    pbar = tqdm(total=len(todo), desc="Bell optimisation sweep", disable=not verbose)

    def _record(kv: int, s: int, res: BellOptimizationResult) -> None:
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
                    obj_func=obj_func,
                    L=L,
                    k=kv,
                    seed=s,
                    method=method,
                    method_params=method_params,
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
                    obj_func=obj_func,
                    L=L,
                    k=kv,
                    seed=s,
                    method=method,
                    method_params=method_params,
                )
                _record(kv, s, res)
                pbar.update(1)

    pbar.close()

    if verbose:
        # Print best per k
        best_per_k: Dict[int, float] = {}
        for (kv, _s), res in results.items():
            if kv not in best_per_k or res.best_value > best_per_k[kv]:
                best_per_k[kv] = res.best_value
        print("\nBest SDP value per k:")
        for kv in sorted_ks:
            if kv in best_per_k:
                print(f"  k={kv}: {best_per_k[kv]:.6f}")

    return results


# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

def _default_method_params(method: Method) -> Dict[str, Any]:
    """Sensible defaults for each method."""
    if method == "sa":
        return {
            "steps": 100,
            "T_start": 2.0,
            "alpha": 0.95,
        }
    elif method == "pt":
        return {
            "num_chains": None,  # auto = num CPUs
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
        return {
            "steps": 20,
        }
    elif method == "random":
        return {}
    else:
        return {}
