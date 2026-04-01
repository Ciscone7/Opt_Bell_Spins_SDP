from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from itertools import permutations

PhaseExp = int  # always interpreted mod 4

Axis = Literal["x", "y", "z"]

# i^p for p in {0,1,2,3}
_I_POW: Final[Tuple[complex, complex, complex, complex]] = (1+0j, 1j, -1+0j, -1j)

# Precompute int real/imag coefficients once
_PHASE_RE: Final[Tuple[int, int, int, int]] = tuple(int(c.real) for c in _I_POW)  # ( 1, 0,-1, 0)
_PHASE_IM: Final[Tuple[int, int, int, int]] = tuple(int(c.imag) for c in _I_POW)  # ( 0, 1, 0,-1)


def _real_basis_coeff(n_y_i: int, n_y_j: int, std_phase: int, n_y_u: int) -> int:
    """Coefficient for moment-matrix entry in the Ỹ=iY basis.

    In this basis the moment matrix is *real-symmetric*.  Each entry is
        M[i,j] = coeff · ỹ_u
    where coeff ∈ {+1, −1} and ỹ_u = ⟨ũ⟩ is a real variable.

    Derivation (D = diag(i^{n_Y(w_k)})):
        M^~ = D* Γ D  →  M^~[i,j] = i^{−n_Y_i + n_Y_j + p − n_Y_u} · ỹ_u
    The exponent is always 0 or 2 (mod 4), giving +1 or −1.

    Parameters
    ----------
    n_y_i : Y-count of basis word w_i
    n_y_j : Y-count of basis word w_j
    std_phase : phase exponent p from standard multiply_words(w_i, w_j)
    n_y_u : Y-count of the product word u = w_i w_j (label, ignoring phase)

    Returns
    -------
    +1 or −1
    """
    exp = (-n_y_i + n_y_j + std_phase - n_y_u) & 3   # mod 4
    # exp must be 0 or 2 for a correct real-basis;
    # 1 or 3 would indicate non-real entries (bug).
    return 1 if exp == 0 else -1


def _real_basis_op_coeff(n_y_u: int) -> float:
    """Operator coefficient correction in the Ỹ=iY basis.

    Standard:  ⟨H⟩ = Σ c_u y_u
    Real basis: ⟨H⟩ = Σ c_u · i^{−n_Y(u)} · ỹ_u = Σ c̃_u · ỹ_u

    For Hamiltonians of physical interest (Heisenberg, Ising, …) every term
    has even n_Y, so i^{−n_Y} ∈ {+1, −1} and c̃_u is real.
    """
    exp = (-n_y_u) & 3   # mod 4
    if exp == 0:
        return 1.0
    if exp == 2:
        return -1.0
    raise ValueError(
        f"Operator term with odd Y-count n_Y={n_y_u} is not supported "
        f"by the real-basis (Ỹ=iY) transformation."
    )


@dataclass(frozen=True, slots=True)
class PauliWord:
    """
    Pauli word on N qubits encoded by two bitmasks.

    Bit k of x_mask is 1 iff there is an X-component on qubit k.
    Bit k of z_mask is 1 iff there is a Z-component on qubit k.

    Local decoding:
      (0,0)->I, (1,0)->X, (0,1)->Z, (1,1)->Y
    """
    x_mask: int
    z_mask: int

    def __post_init__(self) -> None:
        if self.x_mask < 0 or self.z_mask < 0:
            raise ValueError("Bitmasks must be non-negative integers.")
    
    def support_size(self) -> int:
        """Number of non-identity sites."""
        return (self.x_mask | self.z_mask).bit_count()
    
    def signature(self) -> tuple[int, int]:
        """
        Returns (s_xy, s_yz) in {0, 1}.
        0 means parity +1 (even number of flips, invariant).
        1 means parity -1 (odd number of flips, changes sign).
        """
        # S_xy flips X and Y. This corresponds exactly to the x_mask bits.
        s_xy = self.x_mask.bit_count() % 2
        
        # S_yz flips Y and Z. This corresponds exactly to the z_mask bits.
        s_yz = self.z_mask.bit_count() % 2
        
        return (s_xy, s_yz)
    
    def shift(self, k: int, N: int) -> PauliWord:
        """
        Cyclic shift of the Pauli word by k sites on a chain of length N.
        Effectively maps site i -> (i + k) % N.
        """
        if k == 0:
            return self
            
        k = k % N
        if k == 0: # Check again after modulo
            return self

        # We need to rotate the bits within the window of size N.
        # Mask to ensure we don't pick up garbage bits above N
        mask_N = (1 << N) - 1
        
        # Rotation logic: (Left Shift) | (Wrap Around)
        # Note: (x >> (N - k)) handles the wrap-around bits moving to the start
        new_x = ((self.x_mask << k) & mask_N) | (self.x_mask >> (N - k))
        new_z = ((self.z_mask << k) & mask_N) | (self.z_mask >> (N - k))
        
        return PauliWord(new_x, new_z)
    
    def reflect(self, N: int) -> 'PauliWord':
        """Spatial reflection site i -> N - 1 - i."""
        def reverse_bits(n: int, bits: int) -> int:
            result = 0
            for i in range(bits):
                if (n >> i) & 1:
                    result |= (1 << (bits - 1 - i))
            return result
            
        return PauliWord(
            reverse_bits(self.x_mask, N),
            reverse_bits(self.z_mask, N)
        )
    
    def __repr__(self) -> str:
        """
        Return a string representation like 'X0Y1Z2' or 'I' for identity.
        
        Each non-identity operator is represented as a letter (X/Y/Z) followed by the qubit index.
        """
        if self.x_mask == 0 and self.z_mask == 0:
            return "I"
        
        parts = []
        # Find the highest bit set to determine range
        max_bit = max(self.x_mask.bit_length(), self.z_mask.bit_length())
        
        for i in range(max_bit):
            bit = 1 << i
            x_bit = (self.x_mask & bit) != 0
            z_bit = (self.z_mask & bit) != 0
            
            if x_bit and z_bit:
                parts.append(f"Y{i}")
            elif x_bit:
                parts.append(f"X{i}")
            elif z_bit:
                parts.append(f"Z{i}")
        
        return "".join(parts) if parts else "I"


def is_variant_under_sign_symmetries(w: PauliWord) -> bool:
    """
    Returns True if the expectation value <u> must be zero due to 
    Hamiltonian sign symmetries.
    
    Theory: <u> != 0 only if Nx, Ny, and Nz are ALL even numbers.
    """
    nx = (w.x_mask & ~w.z_mask).bit_count()
    ny = (w.x_mask & w.z_mask).bit_count()
    nz = (~w.x_mask & w.z_mask).bit_count()
    
    return (nx % 2 != 0) or (ny % 2 != 0) or (nz % 2 != 0)


@dataclass(frozen=True, slots=True)
class PauliTerm:
    coeff: complex
    word: PauliWord
    
    def __repr__(self) -> str:
        """
        Return a string representation like 'X0Y1', '-Z0Z1', 'i*X0', or '-i*Y1'.
        
        Coefficient is always a phase factor: 1, -1, i, or -i.
        """
        word_str = repr(self.word)
        
        if self.coeff == 1+0j:
            return word_str
        elif self.coeff == -1+0j:
            return f"-{word_str}"
        elif self.coeff == 1j:
            return f"i*{word_str}"
        elif self.coeff == -1j:
            return f"-i*{word_str}"
        else:
            # Fallback (shouldn't happen for proper Pauli terms)
            return f"({self.coeff})*{word_str}"

@dataclass(frozen=True, slots=True)
class PauliMomentMatrixRep:
    """
    Represents M_ij = (a_coef_ij + i b_coef_ij) * y[label_idx_ij],
    where y[...] are real moment variables y_u = <u>.
    """
    basis: List[PauliWord]                 # w_0..w_{n-1}
    labels: List[PauliWord]                # u_0..u_{m-1} (moments needed)
    label_index: Dict[PauliWord, int]      # u -> idx
    label_idx: np.ndarray                  # (n,n) int: which moment label is used at entry (i,j)
    a_coef: np.ndarray                     # (n,n) int8: Re coefficient in { -1,0,1 }
    b_coef: Optional[np.ndarray]           # (n,n) int8: Im coefficient in { -1,0,1 }
    idx_I: int                             # index of identity label

Operator = Dict[PauliWord, complex]

I_PAULI: Final[PauliWord] = PauliWord(0, 0)

def multiply_words(a: PauliWord, b: PauliWord) -> tuple[PhaseExp, PauliWord]:
    """
    Multiply two Pauli words (binary representation).

    Returns:
      (p, c) where
        a * b = i^p * c
      and c is returned as a PauliWord (up to phase).

    Convention:
      P(x,z) = i^{|x&z|} X^x Z^z
    """
    x1, z1 = a.x_mask, a.z_mask
    x2, z2 = b.x_mask, b.z_mask

    # label part is XOR
    x3 = x1 ^ x2
    z3 = z1 ^ z2

    # popcount helpers
    a_xz = (x1 & z1).bit_count()
    b_xz = (x2 & z2).bit_count()
    c_xz = (x3 & z3).bit_count()
    zx   = (z1 & x2).bit_count()   # counts Z from first commuting past X from second

    # phase exponent mod 4:
    # p = |x1&z1| + |x2&z2| - |x3&z3| + 2|z1&x2|   (mod 4)
    p = (a_xz + b_xz - c_xz + 2 * zx) & 3

    return p, PauliWord(x3, z3)


def multiply_terms(t1: PauliTerm, t2: PauliTerm) -> PauliTerm:
    """
    Multiply two Pauli terms (coeff * word), returning a single term in canonical form:
      (c1*w1) * (c2*w2) = (c1*c2*i^p) * w3
    """
    p, w3 = multiply_words(t1.word, t2.word)
    return PauliTerm(t1.coeff * t2.coeff * _I_POW[p], w3)


def reduce_monomial(factors: Sequence[PauliWord]) -> PauliWord:
    """
    Multiply a list of factors and return only the *canonical word label*, discarding the global phase.
    """
    w = I_PAULI  # identity
    for f in factors:
        _, w = multiply_words(w, f)
    return w


def local_pauli(site: int, axis: Axis) -> PauliWord:
    """
    Return the PauliWord representing σ^axis_site on a single site (0-based indexing).
    Encoding:
      X: (x=1,z=0), Z: (x=0,z=1), Y: (x=1,z=1).
    """
    bit = 1 << site
    if axis == "x":
        return PauliWord(bit, 0)
    if axis == "z":
        return PauliWord(0, bit)
    # axis == "y"
    return PauliWord(bit, bit)


def compile_moment_matrix_rep(
    basis: List[PauliWord],
    manager: "SymmetryManager",
    precomputed_label_index: Optional[Dict[PauliWord, int]] = None
) -> PauliMomentMatrixRep:
    """
    Compiles the moment matrix representation M_ij = <w_i^dag w_j>
    respecting all active symmetries in the manager.

    Features:
    - Zeroes out variant moments (Sign Symmetry).
    - Maps moments to canonical variables (Translation/Mirror/Permutation).
    - Drops imaginary part B if use_real_operator is True.
    """
    n = len(basis)
    # The empty label I is always index 0
    I = PauliWord(0, 0)
    
    # Discover Variables (Pass 1)
    # If indices aren't provided, we must find all unique canonical variables
    label_index = precomputed_label_index
    
    if label_index is None:
        label_set = {I}
        for i in range(n):
            wi = basis[i]
            for j in range(i, n):
                wj = basis[j]
                _, u = multiply_words(wi, wj)
                
                # Ask Manager for the canonical form
                # If it returns None, it's a structural zero.
                c = manager.canonicalize(u)
                if c is not None:
                    label_set.add(c)
        
        # Sort for deterministic index
        sorted_labels = sorted(list(label_set), key=lambda x: (x.support_size(), x.x_mask, x.z_mask))
        # Ensure I is 0
        if sorted_labels[0] != I:
            sorted_labels.remove(I)
            sorted_labels.insert(0, I)
        label_index = {l: k for k, l in enumerate(sorted_labels)}
    else:
        # Reconstruct sorted_labels from the map
        sorted_labels = [None] * len(label_index)
        for w, idx in label_index.items():
            sorted_labels[idx] = w

    idx_I = label_index[I]
    
    # Allocate Arrays
    # A = Real part, B = Imaginary part
    # When use_real_basis or use_real_operator is active, B is empty.
    real_matrix = manager.use_real_basis or manager.use_real_operator
    
    label_idx = np.zeros((n, n), dtype=np.int32)
    A = np.zeros((n, n), dtype=np.float64)
    B = np.zeros((0, 0), dtype=np.float64) if real_matrix else np.zeros((n, n), dtype=np.float64)

    # Fill Matrix (Pass 2)
    for i in range(n):
        wi = basis[i]
        
        # Diagonal element (i,i)
        # wi^dag wi = I.  This is always real (1.0).
        label_idx[i, i] = idx_I
        A[i, i] = 1.0
        # B[i,i] remains 0
        
        for j in range(i + 1, n):
            wj = basis[j]
            p, u = multiply_words(wi, wj)
            
            # Canonicalize
            c = manager.canonicalize(u)
            
            if c is None:
                # Structural Zero (killed by symmetry)
                label_idx[i, j] = idx_I
                label_idx[j, i] = idx_I
                # A, B remain 0
                continue
                
            # If valid, look up index
            k = label_index[c]
            label_idx[i, j] = k
            label_idx[j, i] = k
            
            if manager.use_real_basis:
                # Ỹ=iY basis: coefficient is ±1 (real symmetric matrix)
                n_y_i = (wi.x_mask & wi.z_mask).bit_count()
                n_y_j = (wj.x_mask & wj.z_mask).bit_count()
                n_y_u = (u.x_mask & u.z_mask).bit_count()
                coeff = _real_basis_coeff(n_y_i, n_y_j, p, n_y_u)
                A[i, j] = coeff
                A[j, i] = coeff  # A is Symmetric
            else:
                # Standard Pauli basis
                re = _PHASE_RE[p]
                A[i, j] = re
                A[j, i] = re  # A is Symmetric
                
                # Fill B (Imaginary part) - ONLY if needed
                if not manager.use_real_operator:
                    im = _PHASE_IM[p]
                    B[i, j] = im
                    B[j, i] = -im # B is Anti-Symmetric

    return PauliMomentMatrixRep(
        basis=basis,
        labels=sorted_labels,
        label_index=label_index,
        label_idx=label_idx,
        a_coef=A,
        b_coef=B,
        idx_I=idx_I,
    )