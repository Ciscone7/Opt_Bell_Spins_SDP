from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple
import argparse

from spins import models
from spins.basis_builder import (
    generate_npa_basis,
    generate_heisenberg_paper_basis,
    generate_heisenberg_j2_basis_weak,
    generate_heisenberg_j2_basis_strong,
)

from spins.spins_optimize import build_npa_basis_sets


def time_best_avg(fn, repeats: int) -> Tuple[Any, float, float]:
    """Return (result, best_time, avg_time) over `repeats` runs using perf_counter."""
    times: List[float] = []
    out: Any = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return out, float(min(times)), float(sum(times) / len(times))


def parse_ns_from_args(args: argparse.Namespace) -> List[int]:
    """Parse either --Ns or --N-min/--N-max into a sorted unique int list."""
    if getattr(args, "Ns", None):
        Ns = list(args.Ns)
    else:
        if getattr(args, "N_min", None) is None or getattr(args, "N_max", None) is None:
            raise SystemExit("Provide either --Ns or both --N-min and --N-max")
        if args.N_min > args.N_max:
            raise SystemExit("--N-min must be <= --N-max")
        Ns = list(range(int(args.N_min), int(args.N_max) + 1))

    Ns = [int(n) for n in Ns]
    if any(n <= 0 for n in Ns):
        raise SystemExit("All N must be >= 1")
    return sorted(set(Ns))


def model_params_from_args(args: argparse.Namespace) -> Dict[str, float]:
    """Extract model-specific params from parsed args (ising/heisenberg/heisenberg_j2)."""
    if args.model == "ising":
        return {"J": float(args.J), "h": float(args.h), "k": float(args.k)}
    if args.model == "heisenberg":
        return {}
    if args.model == "heisenberg_j2":
        return {"J2": float(args.J2)}
    raise SystemExit(f"Unknown model: {args.model}")


def hamiltonian_exact_fn(model_name: str):
    """Return the exact (QuTiP) Hamiltonian builder for a model."""
    if model_name == "ising":
        return models.ising_hamiltonian_exact
    if model_name == "heisenberg":
        return models.heisenberg_hamiltonian_exact
    if model_name == "heisenberg_j2":
        return models.heisenberg_j2_hamiltonian_exact
    raise ValueError(f"Unknown model: {model_name}")


def hamiltonian_dict_fn(model_name: str):
    """Return the Pauli-operator dict builder for a model."""
    if model_name == "ising":
        return models.ising_hamiltonian_dict
    if model_name == "heisenberg":
        return models.heisenberg_hamiltonian_dict
    if model_name == "heisenberg_j2":
        return models.heisenberg_j2_hamiltonian_dict
    raise ValueError(f"Unknown model: {model_name}")


# Mapping from CLI name to generator function (no extra args beyond N).
_NAMED_BASIS_GENERATORS = {
    "heisenberg_simple": generate_heisenberg_paper_basis,
    "heisenberg_j2_weak": generate_heisenberg_j2_basis_weak,
    "heisenberg_j2_strong": generate_heisenberg_j2_basis_strong,
}


def build_basis_sets(
    N: int,
    start_level: int,
    end_basis: str = "npa",
    end_level: int = 2,
):
    """Return (starting_set, adding_set, final_set) for an optimization sweep.

    Parameters
    ----------
    N : int
        Number of spin sites.
    start_level : int
        NPA level for the starting (fixed) set.
    end_basis : str
        Either ``"npa"`` (use ``end_level``) or a named basis like
        ``"heisenberg_simple"``, ``"heisenberg_j2_weak"``, etc.
    end_level : int
        NPA level for the final set (only used when ``end_basis="npa"``).
    """
    if end_basis == "npa":
        # Fast path: generate the full NPA basis once (at end_level) and slice levels.
        return build_npa_basis_sets(N=N, start_level=start_level, end_level=end_level)

    # Named-basis path: build the starting set from NPA levels 0..start_level.
    start_npa = generate_npa_basis(N=N, k=start_level)
    starting_set = list(start_npa.words)  # all words up to start_level

    # Named basis: generate the full word list, subtract the starting set.
    gen = _NAMED_BASIS_GENERATORS.get(end_basis)
    if gen is None:
        raise ValueError(
            f"Unknown end_basis={end_basis!r}. "
            f"Choose from: npa, {', '.join(sorted(_NAMED_BASIS_GENERATORS))}"
        )
    final_set = gen(N=N)

    starting_words = set(starting_set)
    adding_set = [w for w in final_set if w not in starting_words]

    return starting_set, adding_set, final_set


def add_symmetry_args(parser: argparse.ArgumentParser) -> None:
    """Add symmetry-related CLI arguments to a parser."""
    sym_group = parser.add_argument_group("symmetry options")
    sym_group.add_argument("--use-rotation", action="store_true", default=False,
                          help="Block diagonalize by signature")
    sym_group.add_argument("--use-sign-symmetry", action="store_true", default=False,
                          help="Zero out variant moments")
    sym_group.add_argument("--use-translation", action="store_true", default=False,
                          help="Group by translation orbits")
    sym_group.add_argument("--use-mirror", action="store_true", default=False,
                          help="Group by spatial reflection")
    sym_group.add_argument("--use-permutation", action="store_true", default=False,
                          help="Group by X/Y/Z relabeling")
    sym_group.add_argument("--use-real-operator", action="store_true", default=False,
                          help="Restrict to real-valued moments (drops Im part — may weaken bound)")
    sym_group.add_argument("--use-real-basis", action="store_true", default=False,
                          help="Use Ỹ=iY basis (real SDP without losing tightness)")
    sym_group.add_argument("--use-all-symmetries", action="store_true", default=False,
                          help="Enable all symmetries (shortcut)")


def symmetry_config_from_args(args: argparse.Namespace) -> Dict[str, bool]:
    """Extract symmetry configuration from parsed args (excludes N)."""
    if getattr(args, "use_all_symmetries", False):
        return {
            "use_rotation": True,
            "use_sign_symmetry": True,
            "use_translation": True,
            "use_mirror": True,
            "use_permutation": True,
            "use_real_operator": False,
            "use_real_basis": True,
        }
    return {
        "use_rotation": getattr(args, "use_rotation", False),
        "use_sign_symmetry": getattr(args, "use_sign_symmetry", False),
        "use_translation": getattr(args, "use_translation", False),
        "use_mirror": getattr(args, "use_mirror", False),
        "use_permutation": getattr(args, "use_permutation", False),
        "use_real_operator": getattr(args, "use_real_operator", False),
        "use_real_basis": getattr(args, "use_real_basis", False),
    }

