import numpy as np
import random
import math as m
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Dict, Optional, Any, Callable, Sequence, Tuple, Set

def _n_choose_k(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    try:
        return int(m.comb(n, k))
    except AttributeError:
        return m.factorial(n) // (m.factorial(n - k) * m.factorial(k))


def _indices_to_mask(indices: Sequence[int], N: int) -> np.ndarray:
    x = np.zeros(N, dtype=np.uint8)
    if indices:
        x[list(indices)] = 1
    return x


def _masks_from_indices_list(idxs_list: Sequence[Sequence[int]], N: int) -> np.ndarray:
    X = np.zeros((len(idxs_list), N), dtype=np.uint8)
    for i, idxs in enumerate(idxs_list):
        if idxs:
            X[i, list(idxs)] = 1
    return X


def generate_index_vectors(N: int, k: int, num_samples: int) -> List[Tuple[int, ...]]:
    """Generate unique random index-tuples of length k (sorted) in [0, N).

    Returns a list of tuples like (i1, i2, ..., ik). This is the sparse
    representation of a Hamming-weight-k mask.
    """
    if N <= 0:
        raise ValueError("N must be positive")
    if k < 0 or k > N:
        raise ValueError("k must satisfy 0 <= k <= N")

    n_choose_k = _n_choose_k(N, k)
    if n_choose_k < num_samples:
        num_samples = n_choose_k

    samples: Set[Tuple[int, ...]] = set()
    while len(samples) < num_samples:
        idxs = tuple(sorted(random.sample(range(N), k)))
        samples.add(idxs)
    return list(samples)

def acquisition_ucb(mean: np.ndarray, std: np.ndarray, beta: float) -> np.ndarray:
    """
    Upper Confidence Bound acquisition function.
    
    Parameters:
    -----------
    mean : np.ndarray
        Predicted mean values.
    std : np.ndarray
        Predicted standard deviation values.
    beta : float
        Exploration-exploitation trade-off parameter.
        
    Returns:
    --------
    np.ndarray
        Acquisition values (lower is better for minimization).
    """
    return mean - beta * std

def bayesian(
    obj_func: Callable[[Any], float],
    N: int,
    k: int,
    beta: float,
    n_init: int,
    n_iter: int,
    candidates_per_iter: int = 100,
    previous_best: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    obj_uses_indices: bool = False,
) -> Dict[str, Any]:
    """
    Perform Bayesian Optimization to minimize an objective function over binary vectors.
    
    Parameters:
    -----------
    obj_func : Callable
        Objective function to minimize. Takes a binary vector and returns a scalar.
    N : int
        Dimension of the binary vector.
    k : int
        Hamming weight constraint (number of ones).
    beta : float
        UCB parameter.
    n_init : int
        Number of initial random samples.
    n_iter : int
        Number of optimization iterations.
    candidates_per_iter : int
        Number of candidate vectors to evaluate per iteration.
    previous_best : Optional[np.ndarray]
        Optional starting point for warm-starting.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, default=True
        If True, shows a progress bar.
        
    Returns:
    --------
    Dict
        Dictionary containing 'best_selection', 'best_value', and 'history'.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Internally store only sparse selections as sorted index-tuples.
    X_idx: List[Tuple[int, ...]] = []
    y: List[float] = []

    def _predict_all_trees(estimators: Sequence[Any], candidates: np.ndarray) -> np.ndarray:
        """Return array shaped (n_trees, n_candidates) with per-tree predictions.

        We parallelize over trees using a thread pool sized to all available cores.
        This mirrors the "pool-of-workers" approach used in parallel tempering,
        but uses threads to avoid pickling large candidate matrices.
        """
        n_workers = os.cpu_count() or 1
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            preds = list(ex.map(lambda t: t.predict(candidates), estimators))
        return np.asarray(preds)

    @lru_cache(maxsize=None)
    def _eval_obj_from_indices(idxs: Tuple[int, ...]) -> float:
        if obj_uses_indices:
            return float(obj_func(list(map(int, idxs))))
        return float(obj_func(_indices_to_mask(idxs, N)))

    def _parse_selection(sel: Any) -> Set[int]:
        """Parse a selection given as dense mask (len N) or indices iterable."""
        arr = np.asarray(sel, dtype=int)
        if arr.ndim != 1:
            raise ValueError("selection must be 1D")
        if arr.shape[0] == N:
            return set(int(i) for i in np.flatnonzero(arr).tolist())
        idxs = set(int(i) for i in arr.tolist())
        if any(i < 0 or i >= N for i in idxs):
            raise ValueError("selection indices out of bounds")
        if len(idxs) != int(arr.shape[0]):
            raise ValueError("selection indices contain duplicates")
        return idxs
    
    if previous_best is not None:
        # Use a few warm-start evaluations near the provided selection.
        # Interpretation: previous_best may have <=k selected items; we sample
        # random completions to size k.
        selected = _parse_selection(previous_best)
        if len(selected) > k:
            raise ValueError(f"previous_best selects {len(selected)} items, but k={k}.")

        n_warm = min(10, max(0, n_init))
        n_init = max(0, n_init - n_warm)

        free = [i for i in range(N) if i not in selected]
        for _ in range(n_warm):
            sel = set(selected)
            if len(sel) < k:
                sel.update(random.sample(free, k - len(sel)))
            idxs = tuple(sorted(sel))
            val = _eval_obj_from_indices(idxs)
            X_idx.append(idxs)
            y.append(val)
            if verbose:
                print("warm start")
            
    # Initial random sampling
    for _ in range(n_init):
        idxs = generate_index_vectors(N, k, 1)[0]
        val = _eval_obj_from_indices(tuple(idxs))
        X_idx.append(idxs)
        y.append(val)

    pbar = tqdm(range(n_iter), desc="Bayesian Optimization", leave=True, disable=not verbose)
    for t in pbar:
        # Surrogate model
        model = RandomForestRegressor(random_state=seed)
        X_dense = _masks_from_indices_list(X_idx, N)
        model.fit(X_dense, y)

        # Generate candidates
        cand_idx = generate_index_vectors(N, k, candidates_per_iter)
        candidates = _masks_from_indices_list(cand_idx, N)

        # Per-tree predictions for uncertainty proxy (tree disagreement)
        # shape: (n_estimators, n_candidates)
        all_preds = _predict_all_trees(model.estimators_, candidates)
        
        # Calculate mean and std across trees
        mu = np.mean(all_preds, axis=0)
        sigma = np.std(all_preds, axis=0)
        
        # Calculate acquisition values
        acq_values = acquisition_ucb(mu, sigma, beta)
        
        # Select best candidate (minimum acquisition value)
        best_idx_candidate = np.argmin(acq_values)
        best_choice = cand_idx[int(best_idx_candidate)]
        
        # Evaluate objective function
        best_y = _eval_obj_from_indices(tuple(best_choice))

        X_idx.append(best_choice)
        y.append(best_y)
        
        current_best = np.min(y)
        pbar.set_postfix({"best_val": f"{current_best:.4f}"})


    best_idx = np.argmin(y)
    best_indices = X_idx[int(best_idx)]
    return {
        "best_indices": list(map(int, best_indices)),
        "best_selection": _indices_to_mask(best_indices, N).astype(int),
        "best_value": y[best_idx],
        "history": {
            "indices": [list(map(int, idxs)) for idxs in X_idx],
            "y": [float(v) for v in y],
        },
    }
