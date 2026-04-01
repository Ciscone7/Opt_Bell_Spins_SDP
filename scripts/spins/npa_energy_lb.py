"""Compute a moment-relaxation lower bound for spin-chain models and save results.

Results are saved atomically after each system size N, making the script fully resumable.

On-disk layout::

    spins_sdp/results/spin_moment_energy_lb/v1/<run_name>/
        meta.json      <- config, provenance, timestamps
        data.npz       <- N, basis_size, E_lb, t_best, t_avg

Examples::

    python -m spins_sdp.scripts.npa_energy_lb \
        --model ising --basis npa --npa-level 2 --N-min 5 --N-max 10 \
        --J 1 --h 0.7 --k 0.3 --boundary periodic --name ising_npa2_pbc

    python -m spins_sdp.scripts.npa_energy_lb \
        --model heisenberg --basis heisenberg_simple --Ns 8 10 12 --boundary periodic
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from spins.basis_builder import (
    generate_heisenberg_j2_basis_strong,
    generate_heisenberg_j2_basis_weak,
    generate_heisenberg_paper_basis,
    generate_npa_basis,
)
from artifact_manager import ArtifactManager
from spins_sdp.scripts._common import (
    add_symmetry_args,
    hamiltonian_dict_fn,
    model_params_from_args,
    parse_ns_from_args,
    symmetry_config_from_args,
    time_best_avg,
)
from spins.spins_sdp import solve_pauli_relaxation
from spins.symmetry import SymmetryManager


ARTIFACT = "spin_moment_energy_lb"

FIELDS = {
    "basis_size": np.dtype("int64"),
    "E_lb": np.dtype("float64"),
    "t_best": np.dtype("float64"),
    "t_avg": np.dtype("float64"),
}


def _basis_words(*, basis_name: str, N: int, level: int, boundary: str) -> List[Any]:
    if basis_name in {"heisenberg_simple", "heisenberg_j2_weak", "heisenberg_j2_strong"} and boundary != "periodic":
        raise SystemExit(f"basis={basis_name} requires --boundary periodic")
    if basis_name == "npa":
        return generate_npa_basis(N=N, k=level).words
    if basis_name == "heisenberg_simple":
        return generate_heisenberg_paper_basis(N=N)
    if basis_name == "heisenberg_j2_weak":
        return generate_heisenberg_j2_basis_weak(N=N)
    if basis_name == "heisenberg_j2_strong":
        return generate_heisenberg_j2_basis_strong(N=N)
    raise SystemExit(f"Unknown basis: {basis_name}")


def _default_name(
    model: str, basis: str, npa_level: int, boundary: str,
    params: Dict[str, float],
) -> str:
    """Build a human-readable run name from the config."""
    parts = [model]
    if basis == "npa":
        parts.append(f"npa{npa_level}")
    else:
        parts.append(basis)
    parts.append(boundary)
    for k, v in sorted(params.items()):
        parts.append(f"{k}{v:g}")
    return "_".join(parts)


def compute_and_save(
    *,
    Ns: List[int],
    npa_level: int,
    model_name: str,
    model_params: Dict[str, float],
    basis_name: str,
    boundary: str,
    symmetry_config: Dict[str, bool],
    mosek_tol: float,
    repeats: int,
    out_root: Path,
    name: str,
    resume: bool,
    force: bool,
    verbose: bool,
) -> str:
    config: Dict[str, Any] = {
        "model": model_name,
        "method": "pauli_moment_relaxation",
        "params": dict(model_params),
        "boundary": boundary,
        "basis": basis_name,
        "sense": "min",
        "mosek_tol": float(mosek_tol),
        "repeats": int(repeats),
        "symmetry": symmetry_config,
    }
    if basis_name == "npa":
        config["npa_level"] = int(npa_level)

    am = ArtifactManager(out_root)
    run = am.create_run(artifact=ARTIFACT, name=name, config=config)

    existing = run.load_records(key="N", fields=FIELDS) if resume else {}

    requested = sorted(set(int(n) for n in Ns))
    missing = [n for n in requested if (n not in existing) or force]

    H_dict_fn = hamiltonian_dict_fn(model_name)

    try:
        pbar = tqdm(missing, desc="Moment relaxation sweep")
        for N in pbar:
            pbar.set_postfix({"N": int(N), "done": len(existing), "total": len(requested)})
            basis = _basis_words(basis_name=basis_name, N=N, level=npa_level, boundary=boundary)
            operator = H_dict_fn(N=N, boundary=boundary, **model_params)
            sym_manager = SymmetryManager(N=N, **symmetry_config)

            def run_one() -> float:
                return solve_pauli_relaxation(
                    basis, operator,
                    symmetry_manager=sym_manager,
                    sense="min",
                    mosek_tol=mosek_tol,
                    verbose=verbose,
                )

            val, tb, ta = time_best_avg(run_one, repeats=repeats)
            existing[int(N)] = {
                "basis_size": int(len(basis)),
                "E_lb": float(val),
                "t_best": float(tb),
                "t_avg": float(ta),
            }

            # Atomic checkpoint after each N
            run.save_records(key="N", fields=FIELDS, records=existing)
            run.update_meta(Ns_present=sorted(existing.keys()))
    finally:
        run.update_meta(Ns_present=sorted(existing.keys()))

    return str(run.path)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    n_group = p.add_mutually_exclusive_group(required=True)
    n_group.add_argument("--Ns", nargs="+", type=int, help="Explicit list of N values")
    n_group.add_argument("--N-min", dest="N_min", type=int, help="Minimum N (inclusive)")
    p.add_argument("--N-max", dest="N_max", type=int,
                   help="Maximum N (inclusive, required with --N-min)")

    p.add_argument("--model", choices=["ising", "heisenberg", "heisenberg_j2"],
                   default="ising")
    p.add_argument("--basis",
                   choices=["npa", "heisenberg_simple", "heisenberg_j2_weak",
                            "heisenberg_j2_strong"],
                   default="npa")
    p.add_argument("--npa-level", type=int, default=1,
                   help="NPA/moment level k (used when --basis npa)")

    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--h", type=float, default=0.0)
    p.add_argument("--k", type=float, default=0.0)
    p.add_argument("--J2", type=float, default=0.0)
    p.add_argument("--boundary", choices=["open", "periodic"], default="open")

    p.add_argument("--mosek-tol", type=float, default=1e-9)
    add_symmetry_args(p)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--verbose", action="store_true")

    p.add_argument("--out-root", type=Path,
                   default=Path(__file__).resolve().parents[1] / "results",
                   help="Root directory for results")

    p.add_argument("--name", type=str, default=None,
                   help="Run name (default: auto-generated from config)")

    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action="store_true")

    args = p.parse_args(argv)
    Ns = parse_ns_from_args(args)
    model_params = model_params_from_args(args)
    symmetry_config = symmetry_config_from_args(args)
    name = args.name or _default_name(
        args.model, args.basis, args.npa_level, args.boundary, model_params,
    )

    out_dir = compute_and_save(
        Ns=Ns,
        npa_level=args.npa_level,
        model_name=args.model,
        model_params=model_params,
        basis_name=args.basis,
        boundary=args.boundary,
        symmetry_config=symmetry_config,
        mosek_tol=args.mosek_tol,
        repeats=args.repeats,
        out_root=args.out_root,
        name=name,
        resume=args.resume,
        force=args.force,
        verbose=args.verbose,
    )

    print(f"Results -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
