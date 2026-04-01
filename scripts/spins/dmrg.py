"""Compute DMRG upper bounds for spin-chain models and save results.

Results are saved atomically after each system size N, making the script
fully resumable.

On-disk layout::

    spins_sdp/results/spin_dmrg_energy_ub/v1/<run_name>/
        meta.json      <- config, provenance, timestamps
        data.npz       <- N, E, e, chi_final, elapsed_s

Examples::

    python -m spins_sdp.scripts.dmrg \
        --model heisenberg --N-min 4 --N-max 12 --chi-max 256 --boundary periodic

    python -m spins_sdp.scripts.dmrg \
        --model ising --Ns 6 8 10 --J 1.0 --h 0.5 --chi-max 128 --boundary periodic

    python -m spins_sdp.scripts.dmrg \
        --model heisenberg_j2 --Ns 8 --J2 0.5 --chi-max 512 --mixer --boundary periodic
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from artifact_manager import ArtifactManager
from spins_sdp.scripts._common import (
    model_params_from_args,
    parse_ns_from_args,
)
from spins_sdp.variational import (
    build_heisenberg_pbc_model,
    build_heisenberg_j1j2_pbc_model,
    build_ising_pbc_model,
    initial_product_state,
    dmrg_upper_bound,
)


ARTIFACT = "spin_dmrg_energy_ub"

# Minimum N for two-site DMRG with PBC (TeNPy requires L > n_active_sites)
MIN_N_DMRG = 4

FIELDS = {
    "E": np.dtype("float64"),
    "e": np.dtype("float64"),
    "chi_final": np.dtype("int64"),
    "elapsed_s": np.dtype("float64"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model(
    model_name: str,
    N: int,
    model_params: Dict[str, float],
    conserve: Optional[str],
):
    """Build a TeNPy model for DMRG (periodic boundary conditions)."""
    if model_name == "heisenberg":
        return build_heisenberg_pbc_model(N, J=model_params.get("J", 1.0), conserve=conserve)
    if model_name == "heisenberg_j2":
        return build_heisenberg_j1j2_pbc_model(
            N, J1=model_params.get("J", 1.0), J2=model_params.get("J2", 0.0), conserve=conserve,
        )
    if model_name == "ising":
        return build_ising_pbc_model(
            N, J=model_params.get("J", 1.0), h=model_params.get("h", 0.0),
            k=model_params.get("k", 0.0), conserve=conserve,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _resolve_conserve(
    model_name: str, conserve_arg: str, model_params: Dict[str, float],
) -> Optional[str]:
    """Resolve the symmetry conservation setting.

    For Ising with transverse field (h != 0), Sz conservation must be disabled.
    """
    if conserve_arg.lower() == "none":
        return None
    conserve = conserve_arg
    if model_name == "ising" and conserve in {"Sz", "best"}:
        if abs(model_params.get("h", 0.0)) > 0:
            return None
    return conserve


def _default_name(
    model: str, model_params: Dict[str, float],
    chi_max: int, conserve: str,
) -> str:
    """Build a human-readable run name from the config."""
    parts = [model, "pbc"]
    for k, v in sorted(model_params.items()):
        parts.append(f"{k}{v:g}")
    parts.append(f"chi{chi_max}")
    if conserve.lower() != "none":
        parts.append(f"cons{conserve}")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def compute_and_save(
    *,
    Ns: List[int],
    model_name: str,
    model_params: Dict[str, float],
    conserve: str,
    init_state: str,
    chi_max: int,
    svd_min: float,
    max_E_err: float,
    dchi: int,
    nsweeps: int,
    mixer: bool,
    combine: bool,
    out_root: Path,
    name: str,
    resume: bool,
    force: bool,
    verbose: bool,
) -> str:
    config: Dict[str, Any] = {
        "model": model_name,
        "method": "dmrg",
        "params": dict(sorted(model_params.items())),
        "boundary": "periodic",
        "conserve": conserve,
        "init_state": init_state,
        "chi_max": int(chi_max),
        "svd_min": float(svd_min),
        "max_E_err": float(max_E_err),
        "dchi": int(dchi),
        "nsweeps": int(nsweeps),
        "mixer": bool(mixer),
        "combine": bool(combine),
    }

    am = ArtifactManager(out_root)
    run = am.create_run(artifact=ARTIFACT, name=name, config=config)

    existing = run.load_records(key="N", fields=FIELDS) if resume else {}

    requested = sorted(set(int(n) for n in Ns))

    # Filter out N values too small for two-site DMRG with PBC
    too_small = [n for n in requested if n < MIN_N_DMRG]
    if too_small and verbose:
        print(f"Warning: Skipping N={too_small} (DMRG with PBC requires N >= {MIN_N_DMRG})")
    requested = [n for n in requested if n >= MIN_N_DMRG]

    if not requested:
        raise ValueError(f"No valid N values. DMRG with PBC requires N >= {MIN_N_DMRG}.")

    missing = [n for n in requested if (n not in existing) or force]

    if verbose:
        print(f"Model: {model_name}, boundary=periodic")
        print(f"DMRG params: chi_max={chi_max}, mixer={mixer}, combine={combine}")
        print(f"Requested N: {requested}")
        print(f"Already computed: {sorted(existing.keys())}")
        print(f"To compute: {missing}")

    conserve_resolved = _resolve_conserve(model_name, conserve, model_params)

    try:
        pbar = tqdm(missing, desc="DMRG sweep", disable=not verbose)
        for N in pbar:
            pbar.set_postfix({"N": int(N), "done": len(existing), "total": len(requested)})

            model = _build_model(model_name, N, model_params, conserve_resolved)
            psi = initial_product_state(model, kind=init_state)

            E, psi_out, info, elapsed_s = dmrg_upper_bound(
                model, psi,
                chi_max=chi_max,
                svd_min=svd_min,
                max_E_err=max_E_err,
                mixer=mixer,
                dchi=dchi,
                nsweeps=nsweeps,
                combine=combine,
            )

            existing[int(N)] = {
                "E": float(E),
                "e": float(E) / N,
                "chi_final": int(max(psi_out.chi)) if hasattr(psi_out.chi, '__iter__') else int(psi_out.chi),
                "elapsed_s": float(elapsed_s),
            }

            # Atomic checkpoint after each N
            run.save_records(key="N", fields=FIELDS, records=existing)
            run.update_meta(Ns_present=sorted(existing.keys()))
    finally:
        run.update_meta(Ns_present=sorted(existing.keys()))

    if verbose:
        print(f"\nResults saved to: {run.path}")
        print(f"Total N values: {len(existing)}")

    return str(run.path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # N values
    n_group = p.add_mutually_exclusive_group(required=True)
    n_group.add_argument("--Ns", nargs="+", type=int, help="Explicit list of N values")
    n_group.add_argument("--N-min", dest="N_min", type=int, help="Minimum N (inclusive)")
    p.add_argument("--N-max", dest="N_max", type=int,
                   help="Maximum N (inclusive, required with --N-min)")

    # Model selection
    p.add_argument("--model", choices=["ising", "heisenberg", "heisenberg_j2"],
                   default="heisenberg")

    # Model params
    p.add_argument("--J", type=float, default=1.0, help="Coupling strength J")
    p.add_argument("--h", type=float, default=0.0, help="Ising transverse field h")
    p.add_argument("--k", type=float, default=0.0, dest="k_ising",
                   help="Ising longitudinal field k")
    p.add_argument("--J2", type=float, default=0.0,
                   help="Heisenberg J2 next-nearest neighbor coupling")

    # Symmetry and initial state
    p.add_argument("--conserve", type=str, default="Sz",
                   help="Symmetry conservation: 'Sz' for U(1) or 'None' to disable")
    p.add_argument("--init", type=str, default="neel", choices=["neel", "all_up"],
                   help="Initial product state")

    # DMRG accuracy/performance
    p.add_argument("--chi-max", type=int, default=256, help="Maximum bond dimension")
    p.add_argument("--svd-min", type=float, default=1e-10, help="SVD cutoff")
    p.add_argument("--max-E-err", type=float, default=1e-10,
                   help="Energy convergence threshold")
    p.add_argument("--dchi", type=int, default=64,
                   help="Bond dimension increment per sweep")
    p.add_argument("--nsweeps", type=int, default=2,
                   help="Number of sweeps per chi increment")
    p.add_argument("--mixer", action="store_true",
                   help="Enable DMRG mixer (helps avoid local minima)")
    p.add_argument("--combine", action="store_true",
                   help="Use combined two-site update")

    # Output
    p.add_argument("--out-root", type=Path,
                   default=Path(__file__).resolve().parents[1] / "results",
                   help="Root directory for results")

    p.add_argument("--name", type=str, default=None,
                   help="Run name (default: auto-generated from config)")

    # Resume/force
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action="store_true", help="Recompute all N values")
    p.add_argument("--verbose", action="store_true", default=True)

    args = p.parse_args(argv)
    Ns = parse_ns_from_args(args)

    # Build model params — handle k/k_ising naming
    if args.model == "ising":
        model_params = {"J": float(args.J), "h": float(args.h), "k": float(args.k_ising)}
    elif args.model == "heisenberg":
        model_params = {"J": float(args.J)}
    elif args.model == "heisenberg_j2":
        model_params = {"J": float(args.J), "J2": float(args.J2)}
    else:
        model_params = model_params_from_args(args)

    name = args.name or _default_name(
        args.model, model_params, args.chi_max, args.conserve,
    )

    run_dir = compute_and_save(
        Ns=Ns,
        model_name=args.model,
        model_params=model_params,
        conserve=args.conserve,
        init_state=args.init,
        chi_max=args.chi_max,
        svd_min=args.svd_min,
        max_E_err=args.max_E_err,
        dchi=args.dchi,
        nsweeps=args.nsweeps,
        mixer=args.mixer,
        combine=args.combine,
        out_root=args.out_root,
        name=name,
        resume=args.resume,
        force=args.force,
        verbose=args.verbose,
    )

    print(f"Results -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
