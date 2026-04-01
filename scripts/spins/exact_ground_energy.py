"""Compute exact ground-state energy for spin-chain models and save results.

Results are saved atomically after each system size N, making the script fully resumable.

On-disk layout::

    spins_sdp/results/spin_exact_ground_energy/v1/<run_name>/
        meta.json      <- config, provenance, timestamps
        data.npz       <- N, dim, E0, t_best, t_avg

Examples::

    python -m spins_sdp.scripts.exact_ground_energy \
        --model ising --N-min 5 --N-max 10 --J 1 --h 0.7 --k 0.3 --boundary periodic

    python -m spins_sdp.scripts.exact_ground_energy \
        --model heisenberg --Ns 8 10 12 --boundary periodic --name heisenberg_pbc
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from artifact_manager import ArtifactManager
from spins_sdp.scripts._common import (
    hamiltonian_exact_fn,
    model_params_from_args,
    parse_ns_from_args,
    time_best_avg,
)


ARTIFACT = "spin_exact_ground_energy"

FIELDS = {
    "dim": np.dtype("int64"),
    "E0": np.dtype("float64"),
    "t_best": np.dtype("float64"),
    "t_avg": np.dtype("float64"),
}


def _default_name(model: str, boundary: str, params: Dict[str, float]) -> str:
    """Build a human-readable run name from the config."""
    parts = [model, boundary]
    for k, v in sorted(params.items()):
        parts.append(f"{k}{v:g}")
    return "_".join(parts)


def compute_and_save(
    *,
    Ns: List[int],
    model_name: str,
    model_params: Dict[str, float],
    boundary: str,
    repeats: int,
    out_root: Path,
    name: str,
    resume: bool,
    force: bool,
) -> str:
    config: Dict[str, Any] = {
        "model": model_name,
        "method": "exact_diagonalization",
        "params": dict(model_params),
        "boundary": boundary,
        "repeats": int(repeats),
    }

    am = ArtifactManager(out_root)
    run = am.create_run(artifact=ARTIFACT, name=name, config=config)

    existing = run.load_records(key="N", fields=FIELDS) if resume else {}

    requested = sorted(set(int(n) for n in Ns))
    missing = [n for n in requested if (n not in existing) or force]

    H_fn = hamiltonian_exact_fn(model_name)
    for N in missing:
        def run_one() -> float:
            H = H_fn(N, boundary=boundary, **model_params)
            evals = H.eigenenergies(eigvals=1)
            return float(np.asarray(evals)[0])

        E0, tb, ta = time_best_avg(run_one, repeats=repeats)
        existing[int(N)] = {
            "dim": int(2**N),
            "E0": float(E0),
            "t_best": float(tb),
            "t_avg": float(ta),
        }

        # Atomic checkpoint after each N
        run.save_records(key="N", fields=FIELDS, records=existing)
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

    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--h", type=float, default=0.0)
    p.add_argument("--k", type=float, default=0.0)
    p.add_argument("--J2", type=float, default=0.0)
    p.add_argument("--boundary", choices=["open", "periodic"], default="periodic")

    p.add_argument("--repeats", type=int, default=1, help="Timing repeats per N")

    p.add_argument("--out-root", type=Path,
                   default=Path(__file__).resolve().parents[1] / "results",
                   help="Root directory for results")

    p.add_argument("--name", type=str, default=None,
                   help="Run name (default: auto-generated from config)")

    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip already-computed N values")
    p.add_argument("--force", action="store_true",
                   help="Recompute requested N even if present")

    args = p.parse_args(argv)
    Ns = parse_ns_from_args(args)
    model_params = model_params_from_args(args)
    name = args.name or _default_name(args.model, args.boundary, model_params)

    out_dir = compute_and_save(
        Ns=Ns,
        model_name=args.model,
        model_params=model_params,
        boundary=args.boundary,
        repeats=args.repeats,
        out_root=args.out_root,
        name=name,
        resume=args.resume,
        force=args.force,
    )

    print(f"Results -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
