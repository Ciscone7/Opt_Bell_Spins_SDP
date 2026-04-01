"""
Contains exact Hamiltonians (QuTiP `Qobj`) and corresponding Pauli-operator
dictionary builders for each spin model.

    - Transverse-Field Ising Model (TFIM)
    - Heisenberg model (nearest-neighbor)
    - Heisenberg model with second neighbor coupling
"""

import qutip as qt
from typing import Literal

from spins.pauli_logic import Operator, PauliWord


BoundaryType = Literal["open", "periodic"]

Qobj = qt.Qobj

# === General helpers ===

def _single_site_op(N: int, i: int, op: Qobj) -> Qobj:
    """Helper to create an operator acting on site i."""
    op_list = [qt.qeye(2)] * N
    op_list[i] = op

    return qt.tensor(op_list)


# === Transverse-Field Ising Model ===

# QuTip Hamiltonian
def ising_hamiltonian_exact(N: int, J: float = 1.0, h: float = 0.0, k: float = 0.0, boundary: BoundaryType = 'periodic') -> qt.Qobj:
    """
    Construct the Hamiltonian for the transverse field Ising model:
    H = -J sum Z_i Z_{i+1} - h sum X_i - k sum Z_i
    """
    # Initialize Hamiltonian as a zero operator
    H = qt.qzero([2] * N)

    sz = qt.sigmaz()
    sx = qt.sigmax()

    # Define Pauli operators for each site
    for i in range(N):
        # Nearest-neighbor interaction: sigma_z * sigma_z
        if i < N - 1 or boundary == 'periodic':
            j = (i + 1) % N
            if N > 1:
                # Optimization: Construct Z_i Z_j directly via tensor product
                op_list = [qt.qeye(2)] * N
                op_list[i] = sz
                op_list[j] = sz
                H += -J * qt.tensor(op_list)
            else:
                # Edge case N=1: Z_0 * Z_0 = I
                H += -J * qt.tensor([qt.qeye(2)] * N)

        # Transverse field: sigma_x
        H += -h * _single_site_op(N, i, sx)

        # Parallel field: sigma_z
        H += -k * _single_site_op(N, i, sz)

    return H

# Pauli operator dictionary
def ising_hamiltonian_dict(N: int, J: float, h: float, k: float, boundary: BoundaryType) -> Operator:
    """
        Build the 1D Ising Hamiltonian as a Pauli operator dictionary.

        H = -J Σ_i Z_i Z_{i+1} - h Σ_i X_i - k Σ_i Z_i

        Args:
            N: Number of spins
            J: Nearest-neighbor ZZ coupling strength
            h: Transverse field strength (X direction)
            k: Longitudinal field strength (Z direction)
            boundary: "open" or "periodic"

        Returns:
            Operator dictionary mapping PauliWord -> coefficient
    """
    op: Operator = {}

    # -J sum Z_i Z_{i+1}
    for i in range(N):
        if i < N - 1 or boundary == "periodic":
            j = (i + 1) % N
            w = PauliWord(0, (1 << i) | (1 << j))
            op[w] = op.get(w, 0.0) - J

    # -h sum X_i
    for i in range(N):
        w = PauliWord(1 << i, 0)
        op[w] = op.get(w, 0.0) - h

    # -k sum Z_i
    for i in range(N):
        w = PauliWord(0, 1 << i)
        op[w] = op.get(w, 0.0) - k

    return op


# === Heisenberg Model (nearest-neighbor) ===

# QuTip Hamiltonian
def heisenberg_hamiltonian_exact(N: int, boundary: BoundaryType = 'periodic') -> qt.Qobj:
    """
    Construct the Hamiltonian for the Heisenberg chain (case B from paper):
    H = (1/4) sum_i sum_a∈{x,y,z} σ_i^a σ_{i+1}^a
    """
    # Initialize Hamiltonian as a zero operator
    H = qt.qzero([2] * N)

    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()

    # Sum over all sites
    for i in range(N):
        if i < N - 1 or boundary == 'periodic':
            j = (i + 1) % N
            if N > 1:
                # X_i X_{i+1}
                op_list = [qt.qeye(2)] * N
                op_list[i] = sx
                op_list[j] = sx
                H += 0.25 * qt.tensor(op_list)

                # Y_i Y_{i+1}
                op_list = [qt.qeye(2)] * N
                op_list[i] = sy
                op_list[j] = sy
                H += 0.25 * qt.tensor(op_list)

                # Z_i Z_{i+1}
                op_list = [qt.qeye(2)] * N
                op_list[i] = sz
                op_list[j] = sz
                H += 0.25 * qt.tensor(op_list)
            else:
                # Edge case N=1
                H += 0.75 * qt.tensor([qt.qeye(2)] * N)

    return H

# Pauli operator dictionary
def heisenberg_hamiltonian_dict(N: int, boundary: BoundaryType = "periodic") -> Operator:
    """
    Build the 1D Heisenberg Hamiltonian as a Pauli operator dictionary.

    H = (1/4) Σ_i Σ_a∈{x,y,z} σ_i^a σ_{i+1}^a

    This matches the paper's definition (case B).

    Args:
        N: Number of spins
        boundary: "open" or "periodic"

    Returns:
        Operator dictionary mapping PauliWord -> coefficient
    """
    op: Operator = {}

    # (1/4) sum_i sum_a σ_i^a σ_{i+1}^a
    for i in range(N):
        if i < N - 1 or boundary == "periodic":
            j = (i + 1) % N

            # X_i X_{i+1}
            w = PauliWord((1 << i) | (1 << j), 0)
            op[w] = op.get(w, 0.0) + 0.25

            # Y_i Y_{i+1}
            w = PauliWord((1 << i) | (1 << j), (1 << i) | (1 << j))
            op[w] = op.get(w, 0.0) + 0.25

            # Z_i Z_{i+1}
            w = PauliWord(0, (1 << i) | (1 << j))
            op[w] = op.get(w, 0.0) + 0.25

    return op


# === Heisenberg Model with second neighbor coupling ===


# QuTip Hamiltonian
def heisenberg_j2_hamiltonian_exact(N: int, J2: float, boundary: BoundaryType = 'periodic') -> qt.Qobj:
    """
    Construct the Hamiltonian for the Heisenberg chain with second-neighbor couplings (case C):
    H = (1/4) sum_i sum_a∈{x,y,z} [σ_i^a σ_{i+1}^a + J2 σ_i^a σ_{i+2}^a]

    Args:
        N: Number of spins
        J2: Coupling strength for second-neighbor terms
        boundary: 'periodic' or 'open'

    Returns:
        QuTiP Hamiltonian operator
    """
    # Initialize Hamiltonian as a zero operator
    H = qt.qzero([2] * N)

    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()

    # First-neighbor terms: (1/4) sum_i sum_a σ_i^a σ_{i+1}^a
    for i in range(N):
        if i < N - 1 or boundary == 'periodic':
            j = (i + 1) % N
            if N > 1:
                # X_i X_{i+1}
                op_list = [qt.qeye(2)] * N
                op_list[i] = sx
                op_list[j] = sx
                H += 0.25 * qt.tensor(op_list)

                # Y_i Y_{i+1}
                op_list = [qt.qeye(2)] * N
                op_list[i] = sy
                op_list[j] = sy
                H += 0.25 * qt.tensor(op_list)

                # Z_i Z_{i+1}
                op_list = [qt.qeye(2)] * N
                op_list[i] = sz
                op_list[j] = sz
                H += 0.25 * qt.tensor(op_list)

    # Second-neighbor terms: (J2/4) sum_i sum_a σ_i^a σ_{i+2}^a
    for i in range(N):
        if i < N - 2 or boundary == 'periodic':
            j = (i + 2) % N
            if N > 2 or (N == 2 and boundary == 'periodic'):
                # X_i X_{i+2}
                op_list = [qt.qeye(2)] * N
                op_list[i] = sx
                op_list[j] = sx
                H += (J2 * 0.25) * qt.tensor(op_list)

                # Y_i Y_{i+2}
                op_list = [qt.qeye(2)] * N
                op_list[i] = sy
                op_list[j] = sy
                H += (J2 * 0.25) * qt.tensor(op_list)

                # Z_i Z_{i+2}
                op_list = [qt.qeye(2)] * N
                op_list[i] = sz
                op_list[j] = sz
                H += (J2 * 0.25) * qt.tensor(op_list)

    return H

# Pauli operator dictionary
def heisenberg_j2_hamiltonian_dict(N: int, J2: float, boundary: BoundaryType = "periodic") -> Operator:
    """
    Build the 1D Heisenberg Hamiltonian with second-neighbor couplings as a Pauli operator dictionary.

    H = (1/4) Σ_i Σ_a∈{x,y,z} [σ_i^a σ_{i+1}^a + J2 σ_i^a σ_{i+2}^a]

    This matches the paper's definition (case C).

    Args:
        N: Number of spins
        J2: Coupling strength for second-neighbor terms
        boundary: "open" or "periodic"

    Returns:
        Operator dictionary mapping PauliWord -> coefficient
    """
    op: Operator = {}

    # First-neighbor terms: (1/4) sum_i sum_a σ_i^a σ_{i+1}^a
    for i in range(N):
        if i < N - 1 or boundary == "periodic":
            j = (i + 1) % N

            # X_i X_{i+1}
            w = PauliWord((1 << i) | (1 << j), 0)
            op[w] = op.get(w, 0.0) + 0.25

            # Y_i Y_{i+1}
            w = PauliWord((1 << i) | (1 << j), (1 << i) | (1 << j))
            op[w] = op.get(w, 0.0) + 0.25

            # Z_i Z_{i+1}
            w = PauliWord(0, (1 << i) | (1 << j))
            op[w] = op.get(w, 0.0) + 0.25

    # Second-neighbor terms: (J2/4) sum_i sum_a σ_i^a σ_{i+2}^a
    for i in range(N):
        if i < N - 2 or boundary == "periodic":
            j = (i + 2) % N

            # X_i X_{i+2}
            w = PauliWord((1 << i) | (1 << j), 0)
            op[w] = op.get(w, 0.0) + 0.25 * J2

            # Y_i Y_{i+2}
            w = PauliWord((1 << i) | (1 << j), (1 << i) | (1 << j))
            op[w] = op.get(w, 0.0) + 0.25 * J2

            # Z_i Z_{i+2}
            w = PauliWord(0, (1 << i) | (1 << j))
            op[w] = op.get(w, 0.0) + 0.25 * J2

    return op


# =============================================================================
# Common observables (Pauli-dict form)
# =============================================================================

def magnetization_z_dict(N: int) -> Operator:
    r"""Uniform z-magnetization per site: :math:`\frac{1}{N}\sum_i Z_i`."""
    op: Operator = {}
    for i in range(N):
        w = PauliWord(0, 1 << i)
        op[w] = op.get(w, 0.0) + 1.0 / N
    return op


def staggered_magnetization_z_dict(N: int) -> Operator:
    r"""Staggered z-magnetization: :math:`\frac{1}{N}\sum_i (-1)^i Z_i`."""
    op: Operator = {}
    for i in range(N):
        w = PauliWord(0, 1 << i)
        op[w] = op.get(w, 0.0) + ((-1) ** i) / N
    return op


def nearest_neighbor_correlation_dict(
    N: int,
    axis: str = "z",
    boundary: BoundaryType = "periodic",
) -> Operator:
    r"""Average nearest-neighbour correlation along one axis.

    .. math::
        \frac{1}{N_{\text{bonds}}} \sum_{\langle i,j\rangle}
        \sigma_i^a \sigma_j^a

    Args:
        N: Number of sites.
        axis: ``"x"``, ``"y"``, or ``"z"``.
        boundary: ``"open"`` or ``"periodic"``.
    """
    n_bonds = N if boundary == "periodic" else N - 1
    if n_bonds == 0:
        return {}
    op: Operator = {}
    for i in range(N):
        if i < N - 1 or boundary == "periodic":
            j = (i + 1) % N
            if axis == "x":
                w = PauliWord((1 << i) | (1 << j), 0)
            elif axis == "y":
                w = PauliWord((1 << i) | (1 << j), (1 << i) | (1 << j))
            elif axis == "z":
                w = PauliWord(0, (1 << i) | (1 << j))
            else:
                raise ValueError(f"Unknown axis {axis!r}")
            op[w] = op.get(w, 0.0) + 1.0 / n_bonds
    return op


def two_point_correlation_dict(
    N: int,
    i: int,
    j: int,
    axis: str = "z",
) -> Operator:
    r"""Single two-point correlator :math:`\sigma_i^a \sigma_j^a`.

    Args:
        N: Number of sites (used to validate indices).
        i, j: Site indices.
        axis: ``"x"``, ``"y"``, or ``"z"``.
    """
    if not (0 <= i < N and 0 <= j < N):
        raise ValueError(f"Site indices i={i}, j={j} out of range for N={N}")
    if i == j:
        # σ_i^a σ_i^a = I
        return {PauliWord(0, 0): 1.0}
    if axis == "x":
        w = PauliWord((1 << i) | (1 << j), 0)
    elif axis == "y":
        w = PauliWord((1 << i) | (1 << j), (1 << i) | (1 << j))
    elif axis == "z":
        w = PauliWord(0, (1 << i) | (1 << j))
    else:
        raise ValueError(f"Unknown axis {axis!r}")
    return {w: 1.0}


def single_site_pauli_dict(N: int, site: int, axis: str) -> Operator:
    r"""Single Pauli operator :math:`\sigma_{\text{site}}^{\text{axis}}`.

    Args:
        N: Number of sites.
        site: Site index.
        axis: ``"x"``, ``"y"``, or ``"z"``.
    """
    if not (0 <= site < N):
        raise ValueError(f"Site {site} out of range for N={N}")
    if axis == "x":
        w = PauliWord(1 << site, 0)
    elif axis == "y":
        w = PauliWord(1 << site, 1 << site)
    elif axis == "z":
        w = PauliWord(0, 1 << site)
    else:
        raise ValueError(f"Unknown axis {axis!r}")
    return {w: 1.0}
































