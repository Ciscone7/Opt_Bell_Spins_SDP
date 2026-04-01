"""
SDP assembly for Bell NPA relaxations.

This module provides the tools to build a semidefinite program (SDP) from
a compiled Bell moment-matrix representation.  The algebra (words, bases,
moment-matrix compilation) lives in :mod:`bell.bell_logic`; this module
takes a :class:`~bell.bell_logic.BellMomentMatrixRep` and produces a
CVXPY problem ready to solve.

Public API:
  - :data:`BellOperator` — type alias for Bell operator dictionaries.
  - :data:`Sense` — ``"min"`` or ``"max"``.
  - :class:`BellMomentSDP` — dataclass wrapping the CVXPY problem.
  - :func:`build_moment_matrix_expression` — linear map from moment
    variables to the moment matrix.
  - :func:`compile_operator_linear_form` — project a Bell operator onto
    the moment-variable vector.
  - :func:`build_bell_sdp` — end-to-end SDP construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from bell.bell_logic import (
    BellMomentMatrixRep,
    BellWord,
    _canonicalize,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

BellOperator = Dict[BellWord, float]
"""Mapping from Bell words to real coefficients."""

Sense = Literal["min", "max"]


# ---------------------------------------------------------------------------
# SDP dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BellMomentSDP:
    """Assembled Bell SDP ready for solving."""
    rep: BellMomentMatrixRep
    y: cp.Variable
    M: cp.Expression
    constraints: List[cp.Constraint]
    objective: cp.Expression
    problem: cp.Problem


# ---------------------------------------------------------------------------
# SDP helpers
# ---------------------------------------------------------------------------

def _canonicalize_for_moment(
    w: BellWord,
    rep: BellMomentMatrixRep,
) -> Optional[int]:
    """Return the label index for *w* in *rep*, or ``None`` if absent."""
    c = _canonicalize(w)
    if c in rep.label_index:
        return rep.label_index[c]
    return None


def build_moment_matrix_expression(
    rep: BellMomentMatrixRep,
    y: cp.Variable,
) -> cp.Expression:
    """Build the CVXPY matrix expression ``M = C @ y`` (reshaped).

    Parameters
    ----------
    rep : BellMomentMatrixRep
        Compiled moment-matrix representation from
        :func:`~bell.bell_logic.compile_moment_matrix_rep`.
    y : cp.Variable
        Decision variable of length ``len(rep.labels)``.

    Returns
    -------
    cp.Expression
        Symmetric matrix expression of shape ``(n, n)``.
    """
    n = len(rep.basis)
    m = len(rep.labels)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for j in range(n):
        for i in range(n):
            flat_idx = i + j * n
            k = rep.label_idx[i, j]
            if k >= 0:
                rows.append(flat_idx)
                cols.append(k)
                data.append(1.0)

    C = sp.coo_matrix((data, (rows, cols)), shape=(n * n, m)).tocsr()
    M_vec = cp.Constant(C) @ y
    M = cp.reshape(M_vec, (n, n), order="F")
    return M


def compile_operator_linear_form(
    rep: BellMomentMatrixRep,
    op: BellOperator,
) -> np.ndarray:
    """Project a Bell operator onto the moment-variable vector.

    Returns a coefficient vector ``c`` such that ``c @ y`` equals the
    expectation of *op* under the moment assignment *y*.
    """
    m = len(rep.labels)
    c = np.zeros(m, dtype=float)
    for u, coef in op.items():
        idx = _canonicalize_for_moment(u, rep)
        if idx is None:
            raise KeyError(f"Operator contains label not in rep.labels: {u}")
        c[idx] += coef
    return c


def build_bell_sdp(
    rep: BellMomentMatrixRep,
    objective_op: BellOperator,
    *,
    sense: Sense = "max",
    extra_constraints: Optional[List[cp.Constraint]] = None,
) -> BellMomentSDP:
    """Build the SDP for the Bell NPA hierarchy.

    Because the generator set already excludes the last outcome per setting
    (completeness is baked in via ``E_{d-1|x} = I - sum_{a<d-1} E_{a|x}``),
    the only constraints are:

    - ``y[I] = 1``  (normalisation)
    - ``M >> 0``    (positive semidefiniteness)

    Parameters
    ----------
    rep : BellMomentMatrixRep
        Compiled moment-matrix representation.
    objective_op : BellOperator
        Bell operator to optimise (already expanded via completeness).
    sense : ``"max"`` or ``"min"``
        Optimisation direction.
    extra_constraints : list[cp.Constraint], optional
        Additional CVXPY constraints to impose.

    Returns
    -------
    BellMomentSDP
        Assembled SDP ready for ``problem.solve()``.
    """
    m = len(rep.labels)
    y = cp.Variable(m, name="y")

    M = build_moment_matrix_expression(rep, y)

    constraints: List[cp.Constraint] = []
    constraints.append(y[rep.idx_I] == 1.0)
    constraints.append(M >> 0)

    if extra_constraints:
        constraints.extend(extra_constraints)

    c = compile_operator_linear_form(rep, objective_op)
    obj_expr = c @ y

    objective = cp.Maximize(obj_expr) if sense == "max" else cp.Minimize(obj_expr)
    problem = cp.Problem(objective, constraints)

    return BellMomentSDP(
        rep=rep,
        y=y,
        M=M,
        constraints=constraints,
        objective=obj_expr,
        problem=problem,
    )
