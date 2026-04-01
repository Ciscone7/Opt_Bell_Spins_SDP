from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

from spins.pauli_logic import PauliWord, multiply_words, local_pauli, reduce_monomial

@dataclass(frozen=True, slots=True)
class PauliNPABasis:
    """
    Canonical unique basis up to level k (shortest-length representatives).

    words:      all unique words reachable by <= k generator multiplications
                (i.e., the full NPA level k basis = union of levels[0:k+1])
    levels:     levels[ℓ] contains words whose minimal length is exactly ℓ
                (minimal = fewest generator multiplications to reach from identity)
                To recover NPA level m<k basis: concatenate levels[0] through levels[m]
    min_len:    minimal length at which each word appears
    index:      word -> index in `words` (deterministic ordering)
    """
    N: int
    k: int
    words: List[PauliWord]
    levels: List[List[PauliWord]]
    min_len: Dict[PauliWord, int]
    index: Dict[PauliWord, int]
    
    def __repr__(self) -> str:
        """
        Return a string representation showing all basis words.
        Format: PauliNPABasis(N=2, k=1, size=7): [I, X0, Y0, Z0, X1, Y1, Z1]
        """
        words_str = ", ".join(repr(w) for w in self.words)
        return f"PauliNPABasis(N={self.N}, k={self.k}, size={len(self.words)}): [{words_str}]"


def _pauli_atoms_xyz(N: int) -> List[PauliWord]:
    """
    Atomic generators in a fixed deterministic order: x0,y0,z0,x1,y1,z1,...,x(N-1),y(N-1),z(N-1).
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    atoms: List[PauliWord] = []
    for i in range(N):
        bit = 1 << i
        atoms.append(PauliWord(bit, 0))     # X_i
        atoms.append(PauliWord(bit, bit))   # Y_i
        atoms.append(PauliWord(0, bit))     # Z_i
    return atoms

def generate_npa_basis(N: int, k: int) -> PauliNPABasis:
    """
    Generate the canonical unique basis up to level k.

    Level definition:
      Start from identity I.
      At each step, multiply by any single-site generator in {X_i, Y_i, Z_i}.
      Keep each resulting canonical PauliWord the *first* time it is discovered (minimal length).

    Output ordering:
      sort by (min_length, support_size, x_mask, z_mask).

    Returns an NPABasis containing:
      - words: concatenation of all levels (sorted deterministically)
      - levels: list of per-length frontiers (also sorted deterministically)
      - min_len: shortest length for each word
      - index: mapping word -> row/col index
    """
    if k < 0:
        raise ValueError("k must be >= 0.")
    if N <= 0:
        raise ValueError("N must be >= 1.")

    I = PauliWord(0, 0)
    atoms = _pauli_atoms_xyz(N)

    # BFS structures
    levels: List[List[PauliWord]] = [[] for _ in range(k + 1)]
    min_len: Dict[PauliWord, int] = {I: 0}
    frontier: List[PauliWord] = [I]
    levels[0] = [I]

    # BFS by length: each discovered word is assigned its minimal length
    for length in range(1, k + 1):
        next_set: set[PauliWord] = set()
        for w in frontier:
            for a in atoms:
                _, w_new = multiply_words(w, a)   # ignore phase for the basis
                if w_new not in min_len:          # first time discovered => minimal length
                    min_len[w_new] = length
                    next_set.add(w_new)

        # ordering within the level
        next_level = sorted(
            next_set,
            key=lambda ww: (ww.support_size(), ww.x_mask, ww.z_mask),
        )
        levels[length] = next_level
        frontier = next_level

        # early stop if nothing new appears
        if not frontier:
            # trim remaining empty levels for cleanliness
            levels = levels[: length + 1]
            break

    # global ordering
    all_words = list(min_len.keys())
    all_words_sorted = sorted(
        all_words,
        key=lambda w: (min_len[w], w.support_size(), w.x_mask, w.z_mask),
    )

    index = {w: i for i, w in enumerate(all_words_sorted)}

    return PauliNPABasis(
        N=N,
        k=k,
        words=all_words_sorted,
        levels=levels,
        min_len=min_len,
        index=index,
    )

def _paper_default_r(N: int) -> int:
    """
    Paper choice: r = N/2 for N <= 60, and r = 20 for N = 80,100.
    """
    if N<2:
        return 1
    if N <= 60:
        return N // 2
    return 20

# TODO: Add option to choose pbc/open boundary conditions
def generate_heisenberg_paper_basis(
    N: int,
    *,
    r: Optional[int] = None,
    include_degree3: bool = True,
    include_degree4: bool = True
) -> List[PauliWord]:
    """
    Generate the monomial list used in the paper for the 1D Heisenberg chain (PBC).

    Includes (after reduction):
      - 1
      - σ_i^a
      - σ_i^a σ_{i+j}^b for j=1..r
      - σ_i^a σ_{i+1}^b σ_{i+2}^c
      - σ_i^a σ_{i+1}^b σ_{i+2}^c σ_{i+3}^d

    Paper notes:
      - r is chosen as large as memory allows; they use r=N/2 for N<=60, else r=20 for N=80,100.

    Indexing:
      - paper uses i in {1..N}; here we use 0..N-1 with modulo-N arithmetic (PBC).
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if r is None:
        r = _paper_default_r(N)
    if r < 1:
        raise ValueError("r must be >= 1.")
    if r > N - 1:
        r = N - 1  # distances beyond N-1 are redundant under modulo-N

    axes: Tuple[Axis, Axis, Axis] = ("x", "y", "z")

    out: List[PauliWord] = []
    seen: Set[PauliWord] = set()

    def push(w: PauliWord) -> None:
        if w not in seen:
            seen.add(w)
            out.append(w)

    # degree 0: identity
    push(PauliWord(0, 0))

    # degree 1: σ_i^a
    for i in range(N):
        for a in axes:
            push(local_pauli(i, a))

    # degree 2: σ_i^a σ_{i+j}^b, j=1..r (PBC)
    # NOTE: duplicates can occur when N even and j=N/2 (pairs counted twice); `seen` removes them.
    for j in range(1, r + 1):
        for i in range(N):
            i2 = (i + j) % N
            for a in axes:
                for b in axes:
                    w = reduce_monomial([local_pauli(i, a), local_pauli(i2, b)])
                    push(w)

    # degree 3: contiguous triples σ_i^a σ_{i+1}^b σ_{i+2}^c (PBC)
    if include_degree3:
        for i in range(N):
            s1 = i
            s2 = (i + 1) % N
            s3 = (i + 2) % N
            for a in axes:
                for b in axes:
                    for c in axes:
                        w = reduce_monomial([
                            local_pauli(s1, a),
                            local_pauli(s2, b),
                            local_pauli(s3, c),
                        ])
                        push(w)

    # degree 4: contiguous quadruples σ_i^a σ_{i+1}^b σ_{i+2}^c σ_{i+3}^d (PBC)
    if include_degree4:
        for i in range(N):
            s1 = i
            s2 = (i + 1) % N
            s3 = (i + 2) % N
            s4 = (i + 3) % N
            for a in axes:
                for b in axes:
                    for c in axes:
                        for d in axes:
                            w = reduce_monomial([
                                local_pauli(s1, a),
                                local_pauli(s2, b),
                                local_pauli(s3, c),
                                local_pauli(s4, d),
                            ])
                            push(w)

    return out

def generate_heisenberg_j2_basis_weak(
    N: int,
    *,
    r: Optional[int] = None,
    include_degree3: bool = True,
    include_degree4: bool = True
) -> List[PauliWord]:
    """
    Generate the basis for Heisenberg chain with second-neighbor couplings (J_2 <= 1).
    
    Uses the same structure as the standard paper basis:
      - 1
      - σ_i^a
      - σ_i^a σ_{i+j}^b for j=1..r (PBC)
      - σ_i^a σ_{i+1}^b σ_{i+2}^c (contiguous triples)
      - σ_i^a σ_{i+1}^b σ_{i+2}^c σ_{i+3}^d (contiguous quadruples)
    
    This is identical to generate_heisenberg_paper_basis and is included for
    semantic clarity and potential future modifications.
    
    Args:
        N: Number of spins
        r: Max distance for degree-2 terms (default: paper default)
        include_degree3: Whether to include degree-3 terms
        include_degree4: Whether to include degree-4 terms
        
    Returns:
        List of PauliWords forming the basis
    """
    # For J_2 <= 1, use the standard paper basis
    return generate_heisenberg_paper_basis(
        N=N,
        r=r,
        include_degree3=include_degree3,
        include_degree4=include_degree4
    )

def generate_heisenberg_j2_basis_strong(
    N: int,
    *,
    r: Optional[int] = None,
) -> List[PauliWord]:
    """
    Generate the basis for Heisenberg chain with second-neighbor couplings (J_2 > 1).
    
    For strong second-neighbor coupling, use a modified basis that emphasizes
    the j=2 structure:
      - 1
      - σ_i^a
      - σ_i^a σ_{i+j}^b for j=1..r (PBC)
      - σ_i^a σ_{i+2}^b σ_{i+4}^c (spaced triples)
      - σ_i^a σ_{i+1}^b σ_{i+2}^c σ_{i+3}^d (contiguous quadruples)
    
    Args:
        N: Number of spins
        r: Max distance for degree-2 terms (default: paper default)
        
    Returns:
        List of PauliWords forming the basis
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if r is None:
        r = _paper_default_r(N)
    if r < 1:
        raise ValueError("r must be >= 1.")
    if r > N - 1:
        r = N - 1

    axes: Tuple[Axis, Axis, Axis] = ("x", "y", "z")

    out: List[PauliWord] = []
    seen: Set[PauliWord] = set()

    def push(w: PauliWord) -> None:
        if w not in seen:
            seen.add(w)
            out.append(w)

    # degree 0: identity
    push(PauliWord(0, 0))

    # degree 1: σ_i^a
    for i in range(N):
        for a in axes:
            push(local_pauli(i, a))

    # degree 2: σ_i^a σ_{i+j}^b, j=1..r (PBC)
    for j in range(1, r + 1):
        for i in range(N):
            i2 = (i + j) % N
            for a in axes:
                for b in axes:
                    w = reduce_monomial([local_pauli(i, a), local_pauli(i2, b)])
                    push(w)

    # degree 3: spaced triples σ_i^a σ_{i+2}^b σ_{i+4}^c (PBC)
    # This emphasizes the j=2 structure
    for i in range(N):
        s1 = i
        s2 = (i + 2) % N
        s3 = (i + 4) % N
        for a in axes:
            for b in axes:
                for c in axes:
                    w = reduce_monomial([
                        local_pauli(s1, a),
                        local_pauli(s2, b),
                        local_pauli(s3, c),
                    ])
                    push(w)

    # degree 4: contiguous quadruples σ_i^a σ_{i+1}^b σ_{i+2}^c σ_{i+3}^d (PBC)
    for i in range(N):
        s1 = i
        s2 = (i + 1) % N
        s3 = (i + 2) % N
        s4 = (i + 3) % N
        for a in axes:
            for b in axes:
                for c in axes:
                    for d in axes:
                        w = reduce_monomial([
                            local_pauli(s1, a),
                            local_pauli(s2, b),
                            local_pauli(s3, c),
                            local_pauli(s4, d),
                        ])
                        push(w)

    return out


def split_basis_by_signature(basis: List[PauliWord]) -> Dict[tuple[int,int], List[PauliWord]]:
    groups = {
        (0,0): [], # ++
        (0,1): [], # +-
        (1,0): [], # -+
        (1,1): []  # --
    }
    for w in basis:
        sig = w.signature()
        groups[sig].append(w)
    return groups






