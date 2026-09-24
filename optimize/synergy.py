"""Modular synergy calculation for NPA basis optimization.

This module provides generic functions to compute 'backbone decomposition synergy',
enabling it to be used across both Bell inequalities and spin systems.
"""

from __future__ import annotations
from typing import Callable, Any, Optional, List, Tuple, Dict
from itertools import combinations
from math import comb
import numpy as np


def calculate_synergy(
    obj_func: Callable[[np.ndarray], float],
    current_state: np.ndarray,
    k: int,
    f_zero: Optional[float] = None,
    t: Optional[int] = None,
    seed: Optional[int] = None,
) -> float:
    """Calculate the backbone decomposition synergy of a selection.

    Synergy = Phi(S) - mean_{i in T}[ Phi(S \\ {i}) ]
    where Phi(S) = f(∅) - f(S) and T ⊆ S is a random subset of size
    min(t, k) used to approximate the leave-one-out mean.

    Args:
        obj_func: Objective function (binary mask -> float).
            Should return the value to be minimized (e.g., -violation).
        current_state: Binary mask (length N) representing the selection S.
        k: Hamming weight of current_state (number of elements in S).
        f_zero: Optional precomputed value of obj_func(np.zeros(N)).
            Saves one function evaluation if provided.
        t: Maximum number of leave-one-out subsets to evaluate.
            If None or t >= k, all k subsets are evaluated (exact).
            Setting t < k reduces the cost from k+1 to t+1 SDP calls
            at the price of a noisier estimate.
        seed: Random seed used when subsampling leave-one-out indices
            (only relevant when t is not None and t < k).

    Returns:
        The synergy value (float).
    """
    if k <= 0:
        return 0.0

    # Ensure current_state is a numpy array
    current_state = np.asarray(current_state, dtype=int)

    if f_zero is None:
        f_zero = obj_func(np.zeros_like(current_state))

    def phi(vec: np.ndarray) -> float:
        return f_zero - obj_func(vec)

    phi_total = phi(current_state)

    if k == 1:
        # Synergy of a single element is its own improvement.
        # (Though conceptually synergy usually requires k >= 2 interactions).
        return phi_total

    # Find which indices are active (1)
    ones_indices = np.where(current_state == 1)[0]

    # Optionally subsample leave-one-out indices to cap cost at t+1 SDP calls
    if t is not None and t < k:
        rng = np.random.default_rng(seed)
        eval_indices = rng.choice(ones_indices, size=t, replace=False)
    else:
        eval_indices = ones_indices

    # Evaluate Phi for each chosen leave-one-out subset
    sub_phis = []
    for i in eval_indices:
        subset = current_state.copy()
        subset[i] = 0
        sub_phis.append(phi(subset))

    # Synergy is the total improvement minus the average subset improvement.
    # Theoretically >= 0 (adding monomials can only improve the SDP bound),
    # but finite-precision solvers can produce small negative values for
    # large, nearly-redundant bases.  Clamp to zero.
    synergy = max(0.0, phi_total - np.mean(sub_phis))

    return float(synergy)


def estimate_synergy_at_k(
    obj_func: Callable[[np.ndarray], float],
    L: int,
    k: int,
    n_samples: int = 20,
    f_zero: Optional[float] = None,
    t: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Estimate mean synergy at a given k via random subset sampling.

    Instead of optimizing first, draw ``n_samples`` random k-subsets,
    compute their synergy, and return the mean and standard deviation.
    This is a cheap probe of the *intrinsic* cooperative structure at
    each k — useful for identifying k* before committing to optimization.

    Args:
        obj_func: Objective function (binary mask -> float).
        L: Length of the binary mask (adding-set size).
        k: Subset size to probe.
        n_samples: Number of random subsets to draw.
        f_zero: Optional precomputed f(zeros). Saves one SDP call if provided.
        t: Leave-one-out threshold passed to ``calculate_synergy``.
            Caps synergy evaluation at t+1 SDP calls per sample instead of k+1.
        seed: Random seed for reproducibility.

    Returns:
        Tuple (mean_synergy, std_synergy).
    """
    if k <= 0:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)

    if f_zero is None:
        f_zero = obj_func(np.zeros(L, dtype=int))

    synergies = []
    for i in range(n_samples):
        mask = np.zeros(L, dtype=int)
        mask[rng.choice(L, size=k, replace=False)] = 1
        # Each sample gets a deterministic but distinct seed for its leave-one-out draw
        loo_seed = None if seed is None else int(rng.integers(0, 2**31))
        syn = calculate_synergy(obj_func, mask, k, f_zero=f_zero, t=t, seed=loo_seed)
        synergies.append(syn)

    return float(np.mean(synergies)), float(np.std(synergies))


def random_synergy_scan(
    obj_func: Callable[[np.ndarray], float],
    L: int,
    k_values: List[int],
    n_samples: int = 20,
    t: Optional[int] = None,
    f_zero: Optional[float] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """Scan synergy across k values using random sampling (no optimization).

    For each k in ``k_values``, draws ``n_samples`` random k-subsets and
    returns the mean and std of their synergy.  Much cheaper than running
    a full combinatorial optimizer at every k — use this to identify k*
    (the peak), then only optimize at that k.

    Args:
        obj_func: Objective function (binary mask -> float).
        L: Length of the binary mask (adding-set size).
        k_values: List of k values to probe.
        n_samples: Random subsets to draw per k.
        t: Leave-one-out threshold.  Each synergy evaluation uses at most
            t leave-one-out SDP calls instead of k, capping the per-sample
            cost at t+1 SDP calls regardless of k.  None = exact (all k).
        f_zero: Optional precomputed f(zeros).
        seed: Base random seed (each k gets seed + k for independence).
        verbose: Print a summary table while running.

    Returns:
        Dict mapping k -> {"mean": float, "std": float, "n_samples": int}.
    """
    if f_zero is None:
        f_zero = obj_func(np.zeros(L, dtype=int))

    t_eff = t if t is not None else "all"
    if verbose:
        print(f"{'k':<5} | {'Mean synergy':<16} | {'Std':<12} | {'n_samples':<10} | t")
        print("-" * 58)

    results = {0: {"mean": 0.0, "std": 0.0, "n_samples": 0}}
    if verbose:
        print(f"{0:<5} | {0.0:<16.6f} | {0.0:<12.6f} | {0:<10} | -")

    for k in sorted(k_values):
        k_seed = None if seed is None else seed + k
        mean_syn, std_syn = estimate_synergy_at_k(
            obj_func=obj_func,
            L=L,
            k=k,
            n_samples=n_samples,
            f_zero=f_zero,
            t=t,
            seed=k_seed,
        )
        results[k] = {"mean": mean_syn, "std": std_syn, "n_samples": n_samples}
        if verbose:
            print(f"{k:<5} | {mean_syn:<16.6f} | {std_syn:<12.6f} | {n_samples:<10} | {t_eff}")

    return results


def exact_backbone_decomposition(
    obj_func: Callable[[np.ndarray], float],
    current_state: np.ndarray,
    f_zero: Optional[float] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Exact synergy-first backbone decomposition (Varley 2024).

    For a selection S of size k (the ones in ``current_state``), compute the
    α-synergy at every order and the associated partial synergy atoms.

        h^syn_α(S) = min_{a ⊆ S, |a|=α} [ Φ(S) − Φ(S \\ a) ]

        ∂Φ_α(S)   = h^syn_α(S) − h^syn_{α−1}(S)          (with h^syn_0 = 0)

    where Φ(T) = f(∅) − f(T). These atoms are non-negative (monotonicity of
    h^syn_α in α, up to solver noise) and sum to the total improvement:

        Σ_α ∂Φ_α(S) = h^syn_k(S) = Φ(S).

    Cost: 2^k − 1 evaluations of ``obj_func`` on distinct subsets of S.
    Exact only for small k — each intermediate subset is evaluated once and
    cached, then all (k choose α) losses at each order are read from the cache.

    Args:
        obj_func: Objective function (binary mask -> float).
        current_state: Binary mask of length L with k ones marking S.
        f_zero: Optional precomputed obj_func(zeros). Saves one evaluation.
        verbose: Print progress as subsets are evaluated.

    Returns:
        Dict with:
            "h_syn"         : list[float] of length k+1, h^syn_α for α=0..k
            "atoms"         : list[float] of length k,   ∂Φ_α  for α=1..k
            "phi_S"         : float, Φ(S) = sum(atoms)
            "ones_indices"  : np.ndarray, indices (in the full mask) selected by S
            "n_evaluations" : int, number of obj_func calls made
    """
    current_state = np.asarray(current_state, dtype=int)
    ones_indices = np.where(current_state == 1)[0]
    k = int(len(ones_indices))

    if k == 0:
        return {
            "h_syn": [0.0],
            "atoms": [],
            "phi_S": 0.0,
            "ones_indices": ones_indices,
            "n_evaluations": 0,
        }

    if f_zero is None:
        f_zero = obj_func(np.zeros_like(current_state))
        n_eval = 1
    else:
        n_eval = 0

    # Cache Φ(S \ a) keyed by frozenset of removed indices a.
    phi_cache: Dict[frozenset, float] = {}

    # a = ∅  →  Φ(S)
    phi_S = f_zero - obj_func(current_state)
    n_eval += 1
    phi_cache[frozenset()] = phi_S

    # a = S  →  Φ(∅) = 0  (no SDP call needed)
    phi_cache[frozenset(ones_indices.tolist())] = 0.0

    # Enumerate every proper non-empty subset a ⊂ S, cache Φ(S \ a).
    for alpha in range(1, k):
        n_subsets_alpha = comb(k, alpha)
        if verbose:
            print(f"  α={alpha}: evaluating {n_subsets_alpha} subsets "
                  f"(running total {n_eval} of 2^k - 1 = {2**k - 1})")
        for subset_a in combinations(ones_indices.tolist(), alpha):
            a_fs = frozenset(subset_a)
            if a_fs in phi_cache:
                continue
            mask_minus_a = current_state.copy()
            for i in subset_a:
                mask_minus_a[i] = 0
            phi_cache[a_fs] = f_zero - obj_func(mask_minus_a)
            n_eval += 1

    # α-synergy: min loss over size-α subsets of S.
    h_syn: List[float] = [0.0]  # α = 0
    for alpha in range(1, k + 1):
        min_loss = float("inf")
        for subset_a in combinations(ones_indices.tolist(), alpha):
            loss = phi_S - phi_cache[frozenset(subset_a)]
            if loss < min_loss:
                min_loss = loss
        h_syn.append(float(min_loss))

    # Partial atoms by telescoping; clamp tiny negatives from solver noise.
    atoms = [max(0.0, h_syn[alpha] - h_syn[alpha - 1]) for alpha in range(1, k + 1)]

    return {
        "h_syn": h_syn,
        "atoms": atoms,
        "phi_S": float(phi_S),
        "ones_indices": ones_indices,
        "n_evaluations": n_eval,
    }
