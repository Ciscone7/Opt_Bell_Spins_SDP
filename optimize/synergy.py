"""Modular synergy calculation for NPA basis optimization.

This module provides generic functions to compute 'backbone decomposition synergy',
enabling it to be used across both Bell inequalities and spin systems.
"""

from __future__ import annotations
from typing import Callable, Any, Optional
import numpy as np


def calculate_synergy(
    obj_func: Callable[[np.ndarray], float],
    current_state: np.ndarray,
    k: int,
    f_zero: Optional[float] = None,
) -> float:
    """Calculate the backbone decomposition synergy of a selection.

    Synergy = Phi(S) - mean_{i in S}[ Phi(S \\ {i}) ]
    where Phi(S) = f(∅) - f(S) is the improvement over the empty set.

    Args:
        obj_func: Objective function (binary mask -> float).
            Should return the value to be minimized (e.g., -violation).
        current_state: Binary mask (length N) representing the selection S.
        k: Hamming weight of current_state (number of elements in S).
        f_zero: Optional precomputed value of obj_func(np.zeros(N)).
            Saves one function evaluation if provided.

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
    
    # Evaluate Phi for each subset of size k-1
    sub_phis = []
    for i in ones_indices:
        subset = current_state.copy()
        subset[i] = 0
        sub_phis.append(phi(subset))

    # Synergy is the total improvement minus the average subset improvement
    synergy = phi_total - np.mean(sub_phis)
    
    return float(synergy)
