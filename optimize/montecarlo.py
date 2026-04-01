import numpy as np
import random
import multiprocessing as mp
from tqdm import tqdm

from typing import List, Dict, Optional, Any, Callable, Union


_GLOBAL_OBJ_FUNC: Optional[Callable[[Any], float]] = None


def _init_pt_worker(obj_func: Callable[[Any], float]) -> None:
    global _GLOBAL_OBJ_FUNC
    _GLOBAL_OBJ_FUNC = obj_func


def _pt_sa_worker(
    N: int,
    k: int,
    initial_guess: Optional[Union[List[int], np.ndarray]],
    steps: int,
    T_start: float,
    alpha: float,
    record_history: bool,
    seed: Optional[int],
    verbose: bool,
    return_selection: bool,
    obj_uses_indices: bool,
) -> Dict[str, Any]:
    if _GLOBAL_OBJ_FUNC is None:
        raise RuntimeError("PT worker not initialized with objective function")
    return simulated_annealing(
        obj_func=_GLOBAL_OBJ_FUNC,
        N=N,
        k=k,
        initial_guess=initial_guess,
        steps=steps,
        T_start=T_start,
        alpha=alpha,
        record_history=record_history,
        seed=seed,
        verbose=verbose,
        return_selection=return_selection,
        obj_uses_indices=obj_uses_indices,
    )


def simulated_annealing(
    obj_func: Callable[[np.ndarray], float],
    N: int,
    k: int,
    initial_guess: Optional[Union[List[int], np.ndarray]] = None,
    steps: int = 100,
    T_start: float = 1.0,
    alpha: float = 1.0,
    record_history: bool = False,
    seed: Optional[int] = None,
    verbose: bool = True,
    return_selection: bool = True,
    obj_uses_indices: bool = False,
) -> Dict[str, Any]:
    """
    Generic Simulated Annealing optimizer for binary vectors with fixed Hamming weight.
    
    Parameters:
    -----------
    record_history : bool, default=False
        If True, keeps a copy of the vector at every step (High Memory Usage).
        If False, only tracks the best and current state (Low Memory Usage).
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, default=True
        If True, shows a progress bar.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


    def _eval(key: tuple) -> float:
        if obj_uses_indices:
            return obj_func(list(key))
        m = np.zeros(N, dtype=int)
        if key:
            m[list(key)] = 1
        return obj_func(m)

    # --- Initialization Phase ---
    if initial_guess is not None:
        guess = np.asarray(initial_guess, dtype=int)

        # Two supported forms:
        # (A) full mask of length N with 0/1 entries
        # (B) indices list/array of selected positions (length k)
        if guess.ndim != 1:
            raise ValueError("initial_guess must be a 1D array/list")

        if guess.shape[0] == N:
            # Mask form. Fast path: only compute free indices if we must fill.
            selection = guess.copy()
            sel_indices = np.flatnonzero(selection).tolist()
            current_k = len(sel_indices)
            if current_k > k:
                raise ValueError(f"initial_guess has {current_k} ones, but k={k}.")

            needed = k - current_k
            if needed > 0:
                free_indices = np.flatnonzero(selection == 0).tolist()
                sel_indices += random.sample(free_indices, needed)
                selection[sel_indices[-needed:]] = 1
        else:
            # Indices form (preferred for large N): avoids working with a huge mask.
            sel_indices = [int(i) for i in guess.tolist()]
            if any(i < 0 or i >= N for i in sel_indices):
                raise ValueError("initial_guess indices out of bounds")
            if len(set(sel_indices)) != len(sel_indices):
                raise ValueError("initial_guess indices contain duplicates")

            current_k = len(sel_indices)
            if current_k > k:
                raise ValueError(f"initial_guess has {current_k} indices, but k={k}.")

            selection = np.zeros(N, dtype=int)
            if current_k:
                selection[sel_indices] = 1

            needed = k - current_k
            if needed > 0:
                # Fill remaining using rejection sampling.
                selected_set = set(sel_indices)
                while needed > 0:
                    j = int(np.random.randint(0, N))
                    if j in selected_set:
                        continue
                    selected_set.add(j)
                    sel_indices.append(j)
                    selection[j] = 1
                    needed -= 1
    else:
        # Robust random initialization
        selection = np.zeros(N, dtype=int)
        selection[np.random.choice(N, k, replace=False)] = 1
        sel_indices = np.flatnonzero(selection).tolist()

    # Track selected indices for efficient swapping (avoid O(N) scans)
    selected_indices = set(sel_indices)
    
    # Initial evaluation
    current_cost = _eval(tuple(sorted(selected_indices)))
    
    best_selection = selection.copy()
    best_cost = current_cost
    best_selected_indices = set(selected_indices)

    # Only initialize these lists if we intend to fill them
    all_selections = []
    all_costs = []
    
    # Record initial state if requested
    if record_history:
        all_selections.append(selection.copy())
        all_costs.append(current_cost)

    # --- Optimization Loop ---
    if 0 < k < N:
        pbar = tqdm(range(steps), desc="SA Optimization", leave=True, disable=not verbose)
        for step in pbar:
            temperature = T_start * (alpha ** step)

            # Propose Swap
            # Note: converting set to list is O(k), acceptable for moderate k
            out_idx = random.choice(list(selected_indices))
            
            # Rejection sampling is O(1) expected time when k << N.
            while True:
                in_idx = int(np.random.randint(0, N))
                if in_idx not in selected_indices:
                    break

            # Apply Swap
            selection[out_idx], selection[in_idx] = 0, 1
            selected_indices.remove(out_idx)
            selected_indices.add(in_idx)

            # Evaluate
            new_cost = _eval(tuple(sorted(selected_indices)))

            # Acceptance Criterion
            delta = new_cost - current_cost
            
            if delta < 0:
                accept = True
            else:
                if temperature < 1e-10:
                    accept = False
                else:
                    accept = np.exp(-delta / temperature) > np.random.rand()

            if accept:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_selection = selection.copy()
                    best_selected_indices = set(selected_indices)
                    pbar.set_postfix({"loss": f"{best_cost:.4f}"})
            else:
                # Revert Swap
                selection[out_idx], selection[in_idx] = 1, 0
                selected_indices.add(out_idx)
                selected_indices.remove(in_idx)
            
            # --- HISTORY RECORDING (Conditional) ---
            if record_history:
                all_selections.append(selection.copy())
                all_costs.append(current_cost)
                
    elif record_history:
        # Edge case (k=0 or k=N): just record the single state if requested
        all_selections.append(selection.copy())
        all_costs.append(current_cost)

    best_payload: Dict[str, Any] = {
        "value": best_cost,
        "indices": sorted(int(i) for i in best_selected_indices),
    }
    last_payload: Dict[str, Any] = {
        "value": current_cost,
        "indices": sorted(int(i) for i in selected_indices),
    }
    if return_selection:
        best_payload["selection"] = best_selection
        last_payload["selection"] = selection.copy()

    return {
        "best": best_payload,
        "last": last_payload,
        "all": {
            "value": all_costs,
            "selection": all_selections,
        },
    }


def parallel_tempering(
    obj_func: Callable[[np.ndarray], float],
    N: int,
    k: int,
    num_chains: Optional[int] = None,
    num_epochs: int = 50,
    steps_per_epoch: int = 50,
    T_min: float = 0.1,
    T_max: float = 10.0,
    initial_guess: Optional[Union[List[int], np.ndarray]] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    obj_uses_indices: bool = False,
) -> Dict[str, Any]:
    """
    Parallel Tempering / Replica Exchange Monte Carlo for fixed-Hamming-weight binary vectors.

    Requirements:
      - simulated_annealing must return:
          results["last"] = {"selection": x_last, "value": cost_last}
        and optionally:
          results["best"] = {"selection": x_best, "value": cost_best}

    This PT implementation:
      (1) Evolves each replica using its *last/current* state (not best).
      (2) Swaps adjacent replicas with acceptance prob:
            p = min(1, exp((beta_i - beta_j) * (E_j - E_i))).
    """
    if num_chains is None:
        num_chains = mp.cpu_count()
    
    # Seed only the manager RNG; each worker gets its own deterministic seed below
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Temperature ladder: chain 0 hottest, chain -1 coldest (geometric spacing)
    temperatures = np.logspace(np.log10(T_max), np.log10(T_min), num=num_chains)

    if verbose:
        print("--- PT starting ---")
        print(f"Chains: {num_chains}")
        print(f"Temperatures (hot -> cold): {[float(f'{t:.4g}') for t in temperatures]}")

    # Initialize replicas; seed coldest chain from initial_guess if provided
    replicas_idx: List[set[int]] = []
    for c in range(num_chains):
        if c == num_chains - 1 and initial_guess is not None:
            # Use initial_guess for the coldest chain (lowest temperature)
            guess = np.asarray(initial_guess, dtype=int)
            if guess.ndim == 1 and guess.shape[0] == N:
                idxs = set(int(i) for i in np.flatnonzero(guess).tolist())
            else:
                idxs = set(int(i) for i in guess.tolist())
            # If the guess has fewer than k indices, fill randomly
            if len(idxs) < k:
                pool = [i for i in range(N) if i not in idxs]
                extras = np.random.choice(pool, k - len(idxs), replace=False)
                idxs.update(int(i) for i in extras)
        else:
            idxs = set(int(i) for i in np.random.choice(N, k, replace=False).tolist())
        replicas_idx.append(idxs)

    def _mask_from_indices(idxs: set[int]) -> np.ndarray:
        x = np.zeros(N, dtype=int)
        if idxs:
            x[list(idxs)] = 1
        return x

    # Initial costs (serial; can be parallelized if you want)
    if obj_uses_indices:
        replicas_cost = [obj_func(sorted(idxs)) for idxs in replicas_idx]
    else:
        replicas_cost = [obj_func(_mask_from_indices(idxs)) for idxs in replicas_idx]

    # Track best found across all time (optimization metric)
    global_best_cost = float("inf")
    global_best_idx: Optional[set[int]] = None

    # NOTE: `obj_func` can be very large (captures big adding_set for high end-levels).
    # Passing it as an argument in every starmap task has a big overhead.
    # Instead, broadcast it once per worker via initializer.
    with mp.Pool(processes=num_chains, initializer=_init_pt_worker, initargs=(obj_func,)) as pool:
        pbar = tqdm(range(num_epochs), desc="Parallel Tempering", leave=True, disable=not verbose)

        for epoch in pbar:
            # --- Phase 1: run short constant-temperature SA bursts in parallel ---
            tasks = []
            for i in range(num_chains):
                tasks.append(
                    (
                        N,
                        k,
                        sorted(replicas_idx[i]),   # initial_guess as indices
                        steps_per_epoch,
                        temperatures[i],           # T_start (constant in burst)
                        1.0,                       # alpha=1.0 => constant temperature
                        False,                     # record_history
                        None if seed is None else seed + i + epoch * num_chains,
                        False,                     # verbose inside SA
                        False,                     # return_selection
                        obj_uses_indices,
                    )
                )

            results = pool.starmap(_pt_sa_worker, tasks)

            # Update replicas using the LAST state (PT correctness)
            for i in range(num_chains):
                if "last" not in results[i]:
                    raise KeyError("simulated_annealing must return a 'last' field for PT.")

                last = results[i]["last"]
                replicas_idx[i] = set(int(j) for j in (last.get("indices") or []))
                replicas_cost[i] = float(last["value"])

                # Update global best using best if present, otherwise last
                cand = results[i].get("best", last)
                cand_cost = float(cand["value"])
                if cand_cost < global_best_cost:
                    global_best_cost = cand_cost
                    global_best_idx = set(int(j) for j in (cand.get("indices") or []))
                    pbar.set_postfix({"best_loss": f"{global_best_cost:.6f}"})

            # --- Phase 2: attempt swaps between adjacent temperatures ---
            # Alternate pairing for better mixing: (0,1)(2,3)... then (1,2)(3,4)...
            start_idx = epoch % 2

            for i in range(start_idx, num_chains - 1, 2):
                j = i + 1

                Ti, Tj = temperatures[i], temperatures[j]
                beta_i, beta_j = 1.0 / Ti, 1.0 / Tj

                Ei, Ej = replicas_cost[i], replicas_cost[j]

                # Correct log acceptance ratio:
                # exponent = (beta_i - beta_j) * (E_j - E_i)
                exponent = (beta_i - beta_j) * (Ej - Ei)

                # Accept with probability min(1, exp(exponent)), computed stably
                if exponent >= 0:
                    swap_prob = 1.0
                else:
                    swap_prob = float(np.exp(exponent))

                if np.random.rand() < swap_prob:
                    replicas_idx[i], replicas_idx[j] = replicas_idx[j], replicas_idx[i]
                    replicas_cost[i], replicas_cost[j] = replicas_cost[j], replicas_cost[i]

    if verbose:
        print("--- PT finished ---")

    best_mask = None
    if global_best_idx is not None:
        best_mask = _mask_from_indices(global_best_idx)

    return {
        "best": {"value": global_best_cost, "selection": best_mask},
        "temps": temperatures,
        "final_replicas": {
            "selection": [sorted(s) for s in replicas_idx],
            "value": replicas_cost
        },
    }

