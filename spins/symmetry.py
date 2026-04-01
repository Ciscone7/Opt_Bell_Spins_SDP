from __future__ import annotations
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Optional, List, Set, Tuple, Dict, Any

from spins.pauli_logic import PauliWord, is_variant_under_sign_symmetries

@dataclass(frozen=True)
class SymmetryManager:
    """
    Manages the application of symmetries to Pauli strings.
    Acts as a configuration strategy for the SDP builder.
    """
    N: int
    use_rotation: bool = False  # Block diagonalize by signature
    use_sign_symmetry: bool = False      # Zero out 'all-odd' moments
    use_translation: bool = False        # Group by translation orbits
    use_mirror: bool = False             # Group by spatial reflection
    use_permutation: bool = False        # Group by X/Y/Z relabeling
    use_real_operator: bool = False     # Restrict to real-valued moments (loosens the bound)
    use_real_basis: bool = False        # Use Ỹ=iY basis (makes moment matrix real without losing tightness)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization/config hashing."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SymmetryManager":
        """Create from a dictionary."""
        return cls(**d)
    
    @classmethod
    def default_for_heisenberg(cls, N: int) -> "SymmetryManager":
        """Return default symmetry settings optimized for Heisenberg model.
        
        Note: use_real_operator is False because the moment matrix M[i,j] = i^p * y_u
        has complex entries from Pauli multiplication phases, even when the operator
        and ground state are real.  Dropping the imaginary part weakens the PSD constraint.
        """
        return cls(
            N=N,
            use_rotation=True,
            use_sign_symmetry=True,
            use_translation=True,
            use_mirror=True,
            use_permutation=True,
            use_real_operator=False,
            use_real_basis=True,
        )

    @classmethod
    def none(cls, N: int) -> "SymmetryManager":
        """Return a SymmetryManager with no symmetries enabled."""
        return cls(N=N)

    @lru_cache(maxsize=None)
    def canonical_translation(self, w: PauliWord) -> PauliWord:
        """
        Returns the lexicographically smallest PauliWord in the translation orbit.
        (Minimizing indices: shifts operators to 0, 1, ...).
        """
        best_w = w
        best_key = (w.x_mask, w.z_mask)
        
        current = w
        # Check all N-1 other shifts
        for _ in range(self.N - 1):
            current = current.shift(1, self.N)
            # Compare raw masks (much faster than full object comparison)
            curr_key = (current.x_mask, current.z_mask)
            if curr_key < best_key:
                best_key = curr_key
                best_w = current
                
        return best_w

    @lru_cache(maxsize=None)
    def canonicalize(self, w: PauliWord) -> Optional[PauliWord]:
        """
        Returns the unique canonical representative of 'w' under the active symmetries.
        OPTIMIZED: Separates Spatial minimization from S3 minimization.
        """
        # 1. Sign Symmetry (Structural Zero)
        if self.use_sign_symmetry and is_variant_under_sign_symmetries(w):
            return None

        # 2. Spatial Minimization (Translation + Mirror)
        # We find the "Spatial Canonical Form" first.
        # Since support/indices dominate the sort order (2^i), 
        # we can settle spatial position before worrying about S3 types.
        
        spatial_best = w
        
        if self.use_translation:
            # Find min in translation orbit
            spatial_best = self.canonical_translation(w)
            
            if self.use_mirror:
                # Compare against the min of the reflected orbit
                w_refl = w.reflect(self.N)
                refl_best = self.canonical_translation(w_refl)
                
                # Take the absolute minimum of the two branches
                if (refl_best.support_size(), refl_best.x_mask, refl_best.z_mask) < \
                   (spatial_best.support_size(), spatial_best.x_mask, spatial_best.z_mask):
                    spatial_best = refl_best
        
        elif self.use_mirror:
            # Mirror only (no translation loop)
            w_refl = w.reflect(self.N)
            if (w_refl.support_size(), w_refl.x_mask, w_refl.z_mask) < \
               (w.support_size(), w.x_mask, w.z_mask):
                spatial_best = w_refl

        # 3. Permutation Minimization
        # Now that the string is spatially anchored (e.g., at site 0),
        # we rotate X/Y/Z labels to find the final canonical form.
        if self.use_permutation:
            if self.use_real_basis:
                # In the Ỹ=iY basis the Hamiltonian is XX-ỸỸ+ZZ, so only
                # X↔Z is a valid permutation symmetry (not the full S₃).
                return canonicalize_xz_swap(spatial_best)
            return canonicalize_permutation(spatial_best)
        
        return spatial_best


def apply_permutation(w: PauliWord, p: Tuple[int, int, int]) -> PauliWord:
    """
    Apply a permutation p of axes (0,1,2) corresponding to (X,Y,Z).
    p=(1,2,0) means: Old X->Y, Old Y->Z, Old Z->X.
    """
    # 1. Extract local operators
    # We rebuild the masks from scratch
    new_x_mask = 0
    new_z_mask = 0
    
    # Iterate over all sites that have something
    combined_mask = w.x_mask | w.z_mask
    
    if combined_mask == 0:
        return w # Identity is invariant
        
    length = combined_mask.bit_length()
    
    for i in range(length):
        bit = 1 << i
        has_x = (w.x_mask & bit) != 0
        has_z = (w.z_mask & bit) != 0
        
        if not (has_x or has_z):
            continue
            
        # Determine current type: 0=X, 1=Y, 2=Z
        # Based on: X(1,0), Z(0,1), Y(1,1)
        if has_x and has_z:
            current_type = 1 # Y
        elif has_x:
            current_type = 0 # X
        else:
            current_type = 2 # Z
            
        # Map to new type
        new_type = p[current_type]
        
        # Write to new masks
        # Map: 0->(1,0), 1->(1,1), 2->(0,1)
        if new_type == 0: # X
            new_x_mask |= bit
        elif new_type == 1: # Y
            new_x_mask |= bit
            new_z_mask |= bit
        elif new_type == 2: # Z
            new_z_mask |= bit
            
    return PauliWord(new_x_mask, new_z_mask)

@lru_cache(maxsize=None)
def canonicalize_xz_swap(w: PauliWord) -> PauliWord:
    """Return the smaller of w and its X↔Z-swapped version.

    X↔Z swaps x_mask and z_mask (Y sites, which have both bits set, are
    unchanged).  This is the residual permutation symmetry in the Ỹ=iY basis.
    """
    swapped = PauliWord(w.z_mask, w.x_mask)
    w_key = (w.support_size(), w.x_mask, w.z_mask)
    s_key = (swapped.support_size(), swapped.x_mask, swapped.z_mask)
    return swapped if s_key < w_key else w


@lru_cache(maxsize=None)
def canonicalize_permutation(w: PauliWord) -> PauliWord:
    """
    Return the lexicographically smallest PauliWord in the S3 orbit of w.
    """
    # The 6 permutations of (0, 1, 2)
    perms = [
        (0,1,2), (0,2,1),
        (1,0,2), (1,2,0),
        (2,0,1), (2,1,0)
    ]
    
    # We want to minimize the tuple representation (support_size, x_mask, z_mask)
    # This ensures a consistent canonical representative.
    best = w
    best_key = (w.support_size(), w.x_mask, w.z_mask)
    
    for p in perms[1:]: # Skip identity
        cand = apply_permutation(w, p)
        cand_key = (cand.support_size(), cand.x_mask, cand.z_mask)
        if cand_key < best_key:
            best = cand
            best_key = cand_key
            
    return best


