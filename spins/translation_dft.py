"""Translation-symmetry DFT block-diagonalization for the moment matrix.

When the monomial basis is closed under the lattice translation σ: j -> j+1 (mod N),
each sign-symmetry sector of the moment matrix is block-circulant and can be
block-diagonalized in translation frequency.  One PSD constraint of size ~N·t
(per sector) becomes up to N smaller constraints — an O(N^2) reduction in the
interior-point cost per iteration.

Sub-N orbits
------------
Not every word has a full-size orbit: words with non-trivial translation
stabiliser (e.g. X1X3 on N=4, or X0X1X2X3 on any N) have orbit size p < N
with p | N.  By representation theory of Z_N (orbit-stabiliser + Frobenius
reciprocity), a period-p orbit contributes to only p of the N frequencies —
namely m ∈ {0, N/p, 2N/p, …, (p−1)N/p}.  The block at frequency m thus
contains *only the orbits whose period p satisfies (N/p) | m*.

Conjugate-frequency dedup
-------------------------
In the real basis every moment y_u = ⟨u⟩ is real, so the entries of c_jk[r]
are real and D_{N-m} = conj(D_m).  Since a Hermitian matrix has the same
spectrum as its conjugate, the PSD constraints for m and N-m are equivalent.
By default we emit constraints only for m ∈ [0, N//2] and let the
real-embedding pick up both members of each conjugate pair.

References
----------
``translation_dft_block_diagonalization_guide.md`` for the original derivation
(full-N orbits, all-frequencies version).  Sub-N orbit treatment and dedup are
described in the module docstrings here and in the code comments.

This module is real-basis only (use_real_basis=True).  The complex/standard
basis is not implemented — in practice the real basis is both tighter and
faster, and there is no need for the complex path.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from spins.pauli_logic import (
    PauliWord,
    multiply_words,
    _real_basis_coeff,
)
from spins.symmetry import SymmetryManager


I_PAULI = PauliWord(0, 0)


@dataclass(frozen=True, slots=True)
class DFTMomentBlockRep:
    """A single frequency block of the translation-DFT-decomposed moment matrix.

    Each entry of the block is a sparse real-linear combination of moment
    variables y[u] (real basis):

        Re(M[i,j]) = sum_k a_data[k] * y[a_cols[k]]   for k with a_rows[k] = j*n + i
        Im(M[i,j]) = sum_k b_data[k] * y[b_cols[k]]   (same Fortran-order indexing)

    Rows of the block correspond to active orbits at this frequency (the
    identity, when present, is just one of them — it is a period-1 orbit).
    """
    basis: List[PauliWord]              # row/column labels (orbit reps active at this m)
    labels: List[PauliWord]              # shared global label list (one entry per SDP variable)
    label_index: Dict[PauliWord, int]    # shared global label -> variable index
    idx_I: int                           # variable index of the identity
    n: int                               # block dimension
    a_rows: np.ndarray                   # int32, length nnz_A  (Fortran-flat entry indices)
    a_cols: np.ndarray                   # int32, length nnz_A  (variable indices)
    a_data: np.ndarray                   # float64, length nnz_A
    b_rows: np.ndarray                   # int32, length nnz_B  (empty if real-symmetric)
    b_cols: np.ndarray                   # int32, length nnz_B
    b_data: np.ndarray                   # float64, length nnz_B

    # Metadata (not used by the SDP assembler; useful for diagnostics)
    sector: Tuple[int, int]              # signature of this block
    frequency: int                       # m in 0..N-1
    periods: Tuple[int, ...]             # period of each orbit (row/col)

    def linear_maps(self, num_y_vars: int):
        """Return (CA, CB, n) — see PauliMomentMatrixRep.linear_maps."""
        CA = sp.coo_matrix(
            (self.a_data, (self.a_rows, self.a_cols)),
            shape=(self.n * self.n, num_y_vars),
        ).tocsr()
        if self.b_data.size == 0:
            return CA, None, self.n
        CB = sp.coo_matrix(
            (self.b_data, (self.b_rows, self.b_cols)),
            shape=(self.n * self.n, num_y_vars),
        ).tocsr()
        return CA, CB, self.n


# --------------------------------------------------------------------------
# Step 1 — orbit decomposition (full-N and sub-N)
# --------------------------------------------------------------------------

def build_orbits(basis: List[PauliWord], N: int) -> List[Tuple[PauliWord, ...]]:
    """Partition the basis into translation orbits.

    Each orbit is returned as a tuple ``(u^{(0)}, u^{(1)}, …, u^{(p−1)})`` where
    ``p`` is the orbit's period (a divisor of ``N``) and ``u^{(r)} = σ^r u^{(0)}``.
    The representative ``u^{(0)}`` is the lexicographically smallest word in
    the orbit (a fixed gauge that doesn't affect PSD-ness).

    The identity, when present, is a period-1 orbit ``(I,)`` and sorts first.

    Orbits are sorted by ``(period, support_size, x_mask, z_mask)`` of the
    representative.
    """
    basis_set = set(basis)
    seen: set = set()
    orbits: List[Tuple[PauliWord, ...]] = []

    for w in basis_set:
        if w in seen:
            continue
        # Walk the orbit by repeated shift; stop the first time we cycle back.
        orbit_words: List[PauliWord] = [w]
        current = w.shift(1, N)
        while current != w:
            orbit_words.append(current)
            current = current.shift(1, N)
        p = len(orbit_words)
        # period p must divide N (orbit-stabiliser); also sanity-check distinctness.
        assert N % p == 0, f"Orbit period {p} does not divide N={N} (word {w!r})"
        assert len(set(orbit_words)) == p, f"Internal: orbit walk produced duplicates ({w!r})"
        # Rotate so the lexmin element is first.
        keys = [(u.x_mask, u.z_mask) for u in orbit_words]
        rep_idx = min(range(p), key=lambda r: keys[r])
        rotated = tuple(orbit_words[(rep_idx + r) % p] for r in range(p))
        orbits.append(rotated)
        seen.update(orbit_words)

    orbits.sort(key=lambda o: (len(o), o[0].support_size(), o[0].x_mask, o[0].z_mask))
    return orbits


def _orbit_active_at(period: int, m: int, N: int) -> bool:
    """A period-p orbit is active at frequency m iff (N/p) divides m.

    Equivalently: m is a multiple of N/p in {0, …, N-1}.
    For p=N (full): always active.  For p=1 (identity-like): only m=0.
    """
    return m % (N // period) == 0


# --------------------------------------------------------------------------
# Step 3 — moment oracle in the real basis
# --------------------------------------------------------------------------

def _moment_real_basis(
    v: PauliWord, w: PauliWord, manager: SymmetryManager
) -> Optional[Tuple[float, PauliWord]]:
    """Real-basis moment oracle: ⟨v† w⟩ = γ · y[c] with γ ∈ {+1, −1}.

    Returns (γ, c) where c is the canonical label, or None if the moment is
    structurally zero (killed by sign symmetry).
    """
    p, u = multiply_words(v, w)
    c = manager.canonicalize(u)
    if c is None:
        return None
    n_y_v = (v.x_mask & v.z_mask).bit_count()
    n_y_w = (w.x_mask & w.z_mask).bit_count()
    n_y_u = (u.x_mask & u.z_mask).bit_count()
    gamma = _real_basis_coeff(n_y_v, n_y_w, p, n_y_u)
    return float(gamma), c


# --------------------------------------------------------------------------
# Steps 4–5 — DFT block assembly for one (sector, frequency)
# --------------------------------------------------------------------------

def _build_dft_block_rep(
    active_orbits: List[Tuple[PauliWord, ...]],
    m: int,
    N: int,
    manager: SymmetryManager,
    label_index: Dict[PauliWord, int],
    labels: List[PauliWord],
    sector: Tuple[int, int],
) -> DFTMomentBlockRep:
    """Assemble D_m for one signature sector, restricted to its active orbits.

    Proper Fourier-basis formula (handles sub-N orbits uniformly):

        D_m[j,k] = √(p_j / p_k) · Σ_{s=0}^{p_k-1} c_jk[s] · e^{-2πi m s / N}

    where c_jk[s] = ⟨(u_j^{(0)})† u_k^{(s)}⟩ and p_j = len(orbit_j).  For full-N
    orbits the √(p_j/p_k) factor is 1 and the sum runs over the full ring.
    The identity-as-period-1 orbit is handled by exactly the same formula
    (no special-casing).
    """
    n = len(active_orbits)
    is_real_block = (m == 0) or (N % 2 == 0 and m == N // 2)

    a_rows: List[int] = []
    a_cols: List[int] = []
    a_data: List[float] = []
    b_rows: List[int] = []
    b_cols: List[int] = []
    b_data: List[float] = []

    periods = tuple(len(o) for o in active_orbits)

    for j_idx, orbit_j in enumerate(active_orbits):
        p_j = periods[j_idx]
        v = orbit_j[0]
        for k_idx, orbit_k in enumerate(active_orbits):
            p_k = periods[k_idx]
            scale = float(np.sqrt(p_j / p_k))

            for s in range(p_k):
                w_word = orbit_k[s]
                res = _moment_real_basis(v, w_word, manager)
                if res is None:
                    continue
                gamma, c = res
                idx = label_index[c]

                phase_s = np.exp(-2j * np.pi * m * s / N)
                re_coef = scale * gamma * phase_s.real
                im_coef = scale * gamma * phase_s.imag   # = -sin(2πms/N) for our convention

                flat = k_idx * n + j_idx   # Fortran-order vec((row=j_idx, col=k_idx))

                if re_coef != 0.0:
                    a_rows.append(flat)
                    a_cols.append(idx)
                    a_data.append(re_coef)
                if not is_real_block and im_coef != 0.0:
                    b_rows.append(flat)
                    b_cols.append(idx)
                    b_data.append(im_coef)

    block_basis = [orbit[0] for orbit in active_orbits]

    return DFTMomentBlockRep(
        basis=block_basis,
        labels=labels,
        label_index=label_index,
        idx_I=label_index[I_PAULI],
        n=n,
        a_rows=np.asarray(a_rows, dtype=np.int32),
        a_cols=np.asarray(a_cols, dtype=np.int32),
        a_data=np.asarray(a_data, dtype=np.float64),
        b_rows=np.asarray(b_rows, dtype=np.int32),
        b_cols=np.asarray(b_cols, dtype=np.int32),
        b_data=np.asarray(b_data, dtype=np.float64),
        sector=sector,
        frequency=m,
        periods=periods,
    )


# --------------------------------------------------------------------------
# Top-level entry point
# --------------------------------------------------------------------------

def build_translation_dft_reps(
    basis: List[PauliWord],
    manager: SymmetryManager,
    *,
    dedup_conjugate_frequencies: bool = True,
) -> Tuple[List[DFTMomentBlockRep], Dict[PauliWord, int]]:
    """Build the list of frequency-block reps for a translation-closed basis.

    Returns ``(reps, global_index)`` matching the signature of
    :func:`spins.spins_sdp.build_block_reps`.

    Parameters
    ----------
    basis :
        Pauli-word basis.  Must be closed under translation
        (``manager.is_translation_closed(basis) is True``).
    manager :
        SymmetryManager — must have ``use_real_basis=True``.
    dedup_conjugate_frequencies :
        When True (default), emit PSD blocks only for m ∈ [0, N//2].  In the
        real basis ``D_{N-m} = conj(D_m)`` so the m > N/2 constraints are
        mathematically equivalent to their m < N/2 conjugates and are
        redundant.  Disable (set False) for diagnostics that need every block
        explicitly (e.g. the eigenvalue-multiset check).
    """
    N = manager.N
    if not manager.use_real_basis:
        raise NotImplementedError(
            "use_translation_dft requires use_real_basis=True. "
            "The complex/standard basis path is intentionally not implemented."
        )
    if not manager.is_translation_closed(basis):
        raise ValueError(
            "use_translation_dft requires the basis to be closed under translation "
            "(every cyclic shift of every word must also be in the basis)."
        )

    orbits = build_orbits(basis, N)

    # Group orbits by signature sector
    sectors: Dict[Tuple[int, int], List[Tuple[PauliWord, ...]]] = {}
    for orbit in orbits:
        sig = orbit[0].signature()
        sectors.setdefault(sig, []).append(orbit)

    # Label collection — walk all (orbit_j, orbit_k) ordered pairs at all s
    # offsets, register every canonical label that appears.
    canonical_set: set = {I_PAULI}
    raw_to_canonical: Dict[PauliWord, PauliWord] = {I_PAULI: I_PAULI}
    for sig, orbit_list in sectors.items():
        for orbit_j in orbit_list:
            v = orbit_j[0]
            for orbit_k in orbit_list:
                for s in range(len(orbit_k)):
                    _, u = multiply_words(v, orbit_k[s])
                    c = manager.canonicalize(u)
                    if c is None:
                        continue
                    raw_to_canonical[u] = c
                    canonical_set.add(c)

    sorted_canons = sorted(
        canonical_set,
        key=lambda u: (u.support_size(), u.x_mask, u.z_mask),
    )
    if sorted_canons[0] != I_PAULI:
        sorted_canons.remove(I_PAULI)
        sorted_canons.insert(0, I_PAULI)
    label_index = {u: i for i, u in enumerate(sorted_canons)}
    global_index = {u: label_index[c] for u, c in raw_to_canonical.items()}

    if dedup_conjugate_frequencies:
        m_range = range(N // 2 + 1)
    else:
        m_range = range(N)

    reps: List[DFTMomentBlockRep] = []
    for sig in sorted(sectors.keys()):
        orbit_list = sectors[sig]
        for m in m_range:
            active = [o for o in orbit_list if _orbit_active_at(len(o), m, N)]
            if not active:
                continue
            rep = _build_dft_block_rep(
                active, m, N, manager, label_index, sorted_canons, sig,
            )
            reps.append(rep)

    return reps, global_index
