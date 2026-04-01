"""Bell NPA optimisation — I3322 scenario (3 settings, 2 outcomes, bipartite).

This script is the SpinsSDP-framework equivalent of fra_Opt_Bell's
``run_3322_experiment.py``.  It performs the same optimisation task
(selecting k monomials from NPA level-1→2 to minimise the SDP bound
on the I3322 expression) but uses the clean SpinsSDP abstractions:

  - :mod:`bell.bell_logic` for word algebra and NPA basis generation.
  - :mod:`bell.bell_optimize` for the SDP objective and sweep logic.
  - :mod:`artifact_manager` for structured, resumable result storage
    (``meta.json`` + ``data.npz`` instead of HDF5).

On-disk layout::

    results/bell_optimization_sweep/v1/<run_name>/
        meta.json   ← config, provenance, timestamps
        data.npz    ← run_idx, k, seed, best_value, elapsed_s,
                       n_obj_evals, mask_bits

Usage examples::

    # Run Parallel Tempering on I3322, all k from 0 to 21, 30 seeds
    python -m scripts_fra.run_3322_fra \\
        --method pt \\
        --k-max 21 --num-seeds 30

    # Simulated Annealing, specific k values, specific seeds
    python -m scripts_fra.run_3322_fra \\
        --method sa --sa-steps 200 --sa-T-start 0.3 \\
        --ks 5 10 15 21 --seeds 42 43 44

    # Bayesian optimisation, test mode (1 seed, k={0,1})
    python -m scripts_fra.run_3322_fra \\
        --method bo --ks 0 1 --seeds 42 --resume

    # Resume a previous run
    python -m scripts_fra.run_3322_fra \\
        --method pt --k-max 21 --num-seeds 30 --resume

Notes
-----
Physics constants (hardcoded, matching fra_Opt_Bell convention):
    m_A = m_B = 3 (three measurement settings per party)
    d_A = d_B = 2 (binary outcomes → one projector per setting)

NPA hierarchy: start_level=1, end_level=2 (same as fra_Opt_Bell).

The I3322 operator is built algebraically from :func:`i3322_operator`
below, which produces the same expression as the JSON file
``data/I_3322_std.json`` used by fra_Opt_Bell.
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Allow running as "python scripts_fra/run_3322_fra.py" from the project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from artifact_manager import ArtifactManager, RunDir
from bell.bell_logic import (
    BellScenario,
    BellWord,
    BellOperator,
    alice_projector,
    bob_projector,
    multiply_words,
    expand_bell_operator,
    generate_npa_basis,
    _word_sort_key,
)
from bell.bell_optimize import (
    BellOptimizationResult,
    build_bell_basis_sets,
    sweep_k_values,
)

# ---------------------------------------------------------------------------
# Physics constants — match fra_Opt_Bell's hardcoded values
# ---------------------------------------------------------------------------
M_A = 3  # settings for Alice
M_B = 3  # settings for Bob
D_A = 2  # outcomes for Alice
D_B = 2  # outcomes for Bob

SCENARIO = BellScenario(m_A=M_A, m_B=M_B, d_A=D_A, d_B=D_B)

# NPA levels (matching fra_Opt_Bell config.json defaults)
START_LEVEL = 1
END_LEVEL   = 2

ARTIFACT = "bell_optimization_sweep"

# ---------------------------------------------------------------------------
# Lookup-table mode (mirrors fra_Opt_Bell's LOOK_UP_TABLE flag)
# ---------------------------------------------------------------------------
# When True the precomputed table replaces per-call MOSEK SDP solves,
# making each objective evaluation ~1000x faster.  Requires the file at
# LOOKUP_TABLE_PATH to be present (copied from fra_Opt_Bell/data/).
LOOKUP_TABLE = True

LOOKUP_TABLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "I3322_npa1_to_npa2_alle_relaxations.txt"
)


# ---------------------------------------------------------------------------
# I3322 operator
# ---------------------------------------------------------------------------

def i3322_operator(scenario: BellScenario) -> BellOperator:
    """Build the I3322 Bell operator in the SpinsSDP BellWord algebra.

    The I3322 inequality (Collins & Gisin 2004) for a (3,3,2,2) scenario:

        I3322 = <A1B1> + <A1B2> + <A2B1> + <A2B2>
                + <A2B3> - <A1B3> - <A3B1> + <A3B2>
                - <A2> - <B1> - 2<B2>

    written in terms of projective operators E_{0|x} (Alice) and F_{0|y}
    (Bob) using the transformation:

        <E_{a|x}> = P(a|x)
        <E_{0|x} F_{0|y}> = P(ab=00|xy)
        <A_x> = 2 P(0|x) - 1  →  P(0|x) = (<A_x> + 1) / 2    (ignored
        here — we work directly with projector expectations).

    In the (d=2) binary outcome convention used by SpinsSDP, the single
    independent projector per setting is ``E_{0|x}`` (outcome a=0).
    The fra_Opt_Bell JSON uses the same convention via its string labels:
    ``"A1"`` ↔ E_{0|0},  ``"A1B1"`` ↔ E_{0|0} F_{0|0}, etc. (1-indexed).

    The expansion via :func:`expand_bell_operator` handles the completeness
    substitution E_{1|x} = I - E_{0|x} automatically.

    Returns
    -------
    BellOperator
        Expanded operator ready for SDP assembly.
    """
    # Projectors: E_{0|x} for x in {0,1,2} and F_{0|y} for y in {0,1,2}
    # (0-indexed settings, 0-indexed outcomes)
    def A(x: int) -> BellWord:  # Alice, setting x, outcome 0
        return alice_projector(scenario, x, 0)

    def B(y: int) -> BellWord:  # Bob, setting y, outcome 0
        return bob_projector(scenario, y, 0)

    def AB(x: int, y: int) -> Optional[BellWord]:
        return multiply_words(A(x), B(y))

    # I3322 in terms of projector expectations P(0|x), P(00|xy):
    #   I3322 =  P(00|01) + P(00|02) + P(00|11) + P(00|12)
    #          - P(00|03) + P(00|13) - P(00|21) + P(00|22)
    #          - P(0|1)   - P(0|0)[Bob] - 2*P(0|1)[Bob]
    # Using 0-indexed (x,y): original 1-indexed A_i,B_j → (i-1, j-1)
    # I3322 (fra_Opt_Bell JSON):
    #   "A1B1": +1  →  AB(0,0): +1
    #   "A1B2": +1  →  AB(0,1): +1
    #   "A2B1": +1  →  AB(1,0): +1
    #   "A2B2": +1  →  AB(1,1): +1
    #   "A1B3": -1  →  AB(0,2): -1
    #   "A2B3": +1  →  AB(1,2): +1
    #   "A3B1": -1  →  AB(2,0): -1
    #   "A3B2": +1  →  AB(2,1): +1
    #   "A2":   -1  →  A(1):   -1
    #   "B1":   -1  →  B(0):   -1
    #   "B2":   -2  →  B(1):   -2

    op: BellOperator = {}

    def add(word: BellWord, coef: float) -> None:
        if word is not None:
            op[word] = op.get(word, 0.0) + coef

    # Joint terms
    add(AB(0, 0), +1.0)
    add(AB(0, 1), +1.0)
    add(AB(1, 0), +1.0)
    add(AB(1, 1), +1.0)
    add(AB(0, 2), -1.0)
    add(AB(1, 2), +1.0)
    add(AB(2, 0), -1.0)
    add(AB(2, 1), +1.0)

    # Marginal terms
    add(A(1), -1.0)
    add(B(0), -1.0)
    add(B(1), -2.0)

    # Expand (completeness substitution for last-outcome projectors)
    return expand_bell_operator(op, scenario)


# ---------------------------------------------------------------------------
# Lookup-table objective (mirrors fra_Opt_Bell's FunctionLoader)
# ---------------------------------------------------------------------------

class FunctionLoader:
    """Load the precomputed I3322 SDP values from a text file.

    The file has rows of the form ``<index>, <value>`` where ``index``
    is the integer encoding of the binary selection mask (bit i = 1 means
    adding_set[i] is included) and ``value`` is the corresponding SDP
    objective value.

    This avoids calling MOSEK for every objective evaluation — each lookup
    is O(1) via a dictionary.
    """

    def __init__(self, filepath: str | Path, N: int = 21) -> None:
        self.N = N
        self.data: dict = {}
        print(f"Loading lookup table from {filepath}...")
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    parts = line.split(",")
                    idx = int(parts[0].strip())
                    val = float(parts[1].strip())
                    self.data[idx] = val
        print(f"Lookup table loaded: {len(self.data)} entries.")

    def __call__(self, mask_or_indices) -> float:
        """Evaluate using a binary mask (numpy array) or iterable of indices."""
        import numpy as _np
        if isinstance(mask_or_indices, _np.ndarray) and mask_or_indices.shape[0] == self.N:
            bits = mask_or_indices
        else:
            bits = _np.zeros(self.N, dtype=_np.int32)
            for i in mask_or_indices:
                bits[int(i)] = 1
        # Encode mask to integer index (bit 0 = least-significant)
        idx = 0
        for i, bit in enumerate(reversed(bits)):
            if bit:
                idx += (1 << i)
        # Lookup returns the SDP value; objective minimises so return as-is
        # (sweep_k_values uses sense='max' internally, so we negate below)
        val = self.data.get(idx, 0.0)
        # The lookup table stores the raw SDP value (lower = tighter bound).
        # BellOptimizationObjective with sense='max' returns -val;
        # here we replicate that convention so the objective is comparable.
        return -val


# ---------------------------------------------------------------------------
# Persistence helpers (mirrors scripts/bell/optimization_sweep.py)
# ---------------------------------------------------------------------------

import hashlib, struct


def _stable_bellword_hash(words: List[BellWord], *, n_chars: int = 16) -> str:
    h = hashlib.sha256()
    for w in words:
        for x, a in w.alice_seq:
            h.update(struct.pack("<ii", x, a))
        h.update(b"|")
        for y, b in w.bob_seq:
            h.update(struct.pack("<ii", y, b))
        h.update(b";")
    return h.hexdigest()[:n_chars]


def _default_name(method: str, feedback: bool) -> str:
    scenario = SCENARIO
    parts = [
        f"I3322",
        f"mA{scenario.m_A}_mB{scenario.m_B}",
        f"dA{scenario.d_A}_dB{scenario.d_B}",
        f"lv{START_LEVEL}to{END_LEVEL}",
        method,
    ]
    if feedback:
        parts.append("fb")
    return "_".join(parts)


def _unpack_masks(packed: np.ndarray, L: int) -> np.ndarray:
    if packed.size == 0:
        return np.array([], dtype=bool).reshape(0, L)
    return np.stack([np.unpackbits(row)[:L].astype(bool) for row in packed])


def _load_existing_runs(run: RunDir, L: int):
    tbl = run.load_table()
    if not tbl:
        return {}, {}
    completed = {}
    for i in range(len(tbl["k"])):
        completed[(int(tbl["k"][i]), int(tbl["seed"][i]))] = int(tbl["run_idx"][i])
    arrays = {key: list(arr) for key, arr in tbl.items()}
    return completed, arrays


def _checkpoint(run: RunDir, run_idx, k, seed, best_value, elapsed_s,
                n_obj_evals, mask_bits_list) -> None:
    mb = (
        np.stack(mask_bits_list, axis=0)
        if mask_bits_list
        else np.array([], dtype=np.uint8).reshape(0, 0)
    )
    run.save_table(
        run_idx=np.array(run_idx, dtype=np.int32),
        k=np.array(k, dtype=np.int32),
        seed=np.array(seed, dtype=np.int32),
        best_value=np.array(best_value, dtype=np.float64),
        elapsed_s=np.array(elapsed_s, dtype=np.float64),
        n_obj_evals=np.array(n_obj_evals, dtype=np.int32),
        mask_bits=mb,
    )


# ---------------------------------------------------------------------------
# Main compute-and-save
# ---------------------------------------------------------------------------

def compute_and_save(
    *,
    method: str,
    method_params: Dict[str, Any],
    k_values: List[int],
    seeds: List[int],
    mosek_tol: float,
    out_root: Path,
    name: str,
    resume: bool,
    force: bool,
    verbose: bool,
    feedback: bool = False,
    use_lookup_table: bool = True,
) -> str:
    """Run I3322 optimisation sweep and save results.

    Equivalent to ``compute_and_save`` in
    :mod:`scripts.bell.optimization_sweep`, but hardwired for the I3322
    scenario (3,3,2,2) and uses :func:`i3322_operator` instead of CHSH.
    """
    scenario = SCENARIO
    starting_set, adding_set, _ = build_bell_basis_sets(
        scenario=scenario,
        start_level=START_LEVEL,
        end_level=END_LEVEL,
    )
    L = len(adding_set)

    bell_operator = i3322_operator(scenario)

    # ------------------------------------------------------------------
    # Lookup-table objective override
    # ------------------------------------------------------------------
    # When use_lookup_table=True we replace the BellOptimizationObjective
    # (which calls MOSEK for every SDP evaluation) with a simple dict lookup
    # over the precomputed table.  We wrap it in a lightweight callable that
    # mimics the interface expected by sweep_k_values / run_single_optimization.
    lookup_obj = None
    if use_lookup_table:
        if not LOOKUP_TABLE_PATH.exists():
            print(
                f"WARNING: Lookup table not found at {LOOKUP_TABLE_PATH}.\n"
                f"         Falling back to MOSEK SDP evaluation."
            )
        else:
            lookup_obj = FunctionLoader(LOOKUP_TABLE_PATH, N=L)
            if verbose:
                print("Using precomputed lookup table (fast mode).")

    if verbose:
        print(
            f"I3322 scenario: ({scenario.m_A},{scenario.m_B}), "
            f"d=({scenario.d_A},{scenario.d_B})"
        )
        print(f"Starting set: {len(starting_set)}, Adding set: {L}")
        print(f"Method: {method}, k values: {sorted(k_values)}, seeds: {sorted(seeds)}")
        if feedback:
            print("Feedback mode: ON (warm-start chaining across k values)")

    adding_set_hash = _stable_bellword_hash(adding_set)

    config: Dict[str, Any] = {
        "expression": "I3322",
        "scenario": {
            "m_A": scenario.m_A, "m_B": scenario.m_B,
            "d_A": scenario.d_A, "d_B": scenario.d_B,
        },
        "start_level": START_LEVEL,
        "end_level": END_LEVEL,
        "starting_set_size": len(starting_set),
        "adding_set_size": L,
        "adding_set_hash": adding_set_hash,
        "method": method,
        "method_params": dict(sorted(method_params.items())),
        "mosek_tol": float(mosek_tol),
        "feedback": bool(feedback),
        "lookup_table": bool(lookup_obj is not None),
    }

    am = ArtifactManager(out_root)
    run = am.create_run(artifact=ARTIFACT, name=name, config=config)

    # ------------------------------------------------------------------
    # Load existing checkpoint
    # ------------------------------------------------------------------
    existing_results: Dict = {}
    if not force:
        completed, existing_arrays = (
            _load_existing_runs(run, L) if resume else ({}, {})
        )
        for (kv, s), run_idx in completed.items():
            pos = list(existing_arrays["run_idx"]).index(run_idx)
            mask: Optional[np.ndarray] = None
            if "mask_bits" in existing_arrays:
                packed = existing_arrays["mask_bits"][pos]
                if isinstance(packed, np.ndarray):
                    mask = np.unpackbits(packed)[:L].astype(np.int32)
            if mask is None:
                mask = np.zeros(L, dtype=np.int32)
            existing_results[(kv, s)] = BellOptimizationResult(
                best_value=float(existing_arrays["best_value"][pos]),
                best_indices=sorted(int(i) for i in np.flatnonzero(mask)),
                mask=mask,
                elapsed_s=float(existing_arrays["elapsed_s"][pos]),
                n_obj_evals=int(existing_arrays["n_obj_evals"][pos]),
                method=method,
                k=kv,
                seed=s,
            )

    if verbose:
        all_jobs = [(kv, s) for kv in sorted(k_values) for s in seeds]
        n_todo = sum(1 for j in all_jobs if j not in existing_results)
        print(
            f"Total jobs: {len(all_jobs)}, "
            f"Already done: {len(existing_results)}, "
            f"To compute: {n_todo}"
        )

    if not any(
        (kv, s) not in existing_results
        for kv in k_values for s in seeds
    ):
        if verbose:
            print("All jobs already completed. Nothing to do.")
        return str(run.path)

    # ------------------------------------------------------------------
    # Accumulator + checkpoint callback
    # ------------------------------------------------------------------
    _acc_run_idx: List[int] = []
    _acc_k: List[int] = []
    _acc_seed: List[int] = []
    _acc_best_value: List[float] = []
    _acc_elapsed_s: List[float] = []
    _acc_n_obj_evals: List[int] = []
    _acc_mask_bits: List[np.ndarray] = []

    if not force and resume:
        _, arr = _load_existing_runs(run, L)
        if arr:
            _acc_run_idx   = list(arr.get("run_idx", []))
            _acc_k         = list(arr.get("k", []))
            _acc_seed      = list(arr.get("seed", []))
            _acc_best_value = list(arr.get("best_value", []))
            _acc_elapsed_s  = list(arr.get("elapsed_s", []))
            _acc_n_obj_evals = list(arr.get("n_obj_evals", []))
            _acc_mask_bits  = list(arr.get("mask_bits", []))

    _next_run_idx = (max(int(x) for x in _acc_run_idx) + 1) if _acc_run_idx else 0

    def _on_result(kv: int, s: int, res: BellOptimizationResult) -> None:
        nonlocal _next_run_idx
        packed = np.packbits(res.mask.astype(np.uint8))
        _acc_run_idx.append(_next_run_idx)
        _acc_k.append(kv)
        _acc_seed.append(s)
        _acc_best_value.append(res.best_value)
        _acc_elapsed_s.append(res.elapsed_s)
        _acc_n_obj_evals.append(res.n_obj_evals)
        _acc_mask_bits.append(packed)
        _next_run_idx += 1
        _checkpoint(
            run, _acc_run_idx, _acc_k, _acc_seed,
            _acc_best_value, _acc_elapsed_s,
            _acc_n_obj_evals, _acc_mask_bits,
        )

    run.update_meta(
        k_values_requested=sorted(k_values),
        seeds_requested=sorted(seeds),
        L=int(L),
    )

    try:
        if lookup_obj is not None:
            # Pass the lookup objective directly via sweep_k_values' obj_func.
            # We do this by monkey-patching the sense attribute so that
            # run_single_optimization can use it directly without MOSEK.
            # lookup_obj already returns -val (maximisation convention).
            from bell.bell_optimize import (
                BellOptimizationObjective,
                run_single_optimization,
            )
            from tqdm import tqdm

            # Attach minimal attributes so run_single_optimization is happy
            lookup_obj.sense = "max"  # type: ignore[attr-defined]
            lookup_obj.adding_set = adding_set  # type: ignore[attr-defined]

            sorted_ks = sorted(k_values)
            todo = [
                (kv, s)
                for kv in sorted_ks
                for s in seeds
                if (kv, s) not in existing_results
            ]
            pbar = tqdm(total=len(todo), desc="I3322 lookup sweep",
                        disable=not verbose)
            for kv, s in todo:
                from bell.bell_optimize import _default_method_params
                mp = method_params or _default_method_params(method)
                prev_mask = None
                if feedback and (kv - 1, s) in existing_results:
                    prev_mask = existing_results[(kv - 1, s)].mask
                res = run_single_optimization(
                    obj_func=lookup_obj,
                    L=L,
                    k=kv,
                    seed=s,
                    method=method,
                    method_params=mp,
                    initial_guess=prev_mask,
                )
                existing_results[(kv, s)] = res
                _on_result(kv, s, res)
                pbar.update(1)
            pbar.close()
        else:
            sweep_k_values(
                scenario=scenario,
                bell_operator=bell_operator,
                starting_set=starting_set,
                adding_set=adding_set,
                k_values=k_values,
                method=method,
                method_params=method_params,
                mosek_tol=mosek_tol,
                seeds=seeds,
                feedback=feedback,
                existing_results=existing_results,
                on_result=_on_result,
                verbose=verbose,
            )
    finally:
        run.update_meta(
            k_values_present=sorted(set(int(x) for x in _acc_k)),
            seeds_present=sorted(set(int(x) for x in _acc_seed)),
            total_runs=len(_acc_run_idx),
        )

    n_computed = len(_acc_run_idx) - len(existing_results)
    if verbose:
        print(f"\nResults -> {run.path}")
        print(f"Total runs: {len(_acc_run_idx)} (computed this call: {n_computed})")

    return str(run.path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    from scripts_fra import load_config  # shared helper

    # Built-in script defaults (lowest priority)
    _DEFAULTS: Dict[str, Any] = {
        "method":      "pt",
        "k_max":       21,
        "num_seeds":   30,
        "mosek_tol":   1e-6,
        "feedback":    False,
        "resume":      True,
        "verbose":     True,
        "lookup_table": LOOKUP_TABLE,
        "out_root":    str(Path(__file__).resolve().parents[1] / "results"),
        "sa":  {"steps": 200, "T_start": 0.3, "alpha": 0.95},
        "pt":  {"num_chains": 0, "num_epochs": 10, "steps_per_epoch": 20,
                "T_min": 0.1, "T_max": 2.0},
        "bo":  {"beta": 2.0, "n_init": 10, "n_iter": 40,
                "candidates_per_iter": 200},
        "rbm": {"steps": 200},
    }

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config file (optional — values it provides act as defaults)
    p.add_argument(
        "--config", metavar="PATH", default=None,
        help=(
            "Path to a JSON config file.  All parameters can be specified "
            "there; any CLI flag provided alongside --config will override "
            "the corresponding config value."
        ),
    )

    # Method
    p.add_argument("--method", choices=["sa", "pt", "bo", "rbm", "random"],
                   default=None, help="Optimisation method")

    # SA parameters (None = take from config / built-in default)
    p.add_argument("--sa-steps",   type=int,   default=None)
    p.add_argument("--sa-T-start", type=float, default=None)
    p.add_argument("--sa-alpha",   type=float, default=None)

    # PT parameters
    p.add_argument("--pt-chains",          type=int,   default=None,
                   help="Number of chains (0 = auto = num CPUs)")
    p.add_argument("--pt-epochs",          type=int,   default=None)
    p.add_argument("--pt-steps-per-epoch", type=int,   default=None)
    p.add_argument("--pt-T-min",           type=float, default=None)
    p.add_argument("--pt-T-max",           type=float, default=None)

    # BO parameters
    p.add_argument("--bo-beta",                type=float, default=None)
    p.add_argument("--bo-n-init",              type=int,   default=None)
    p.add_argument("--bo-n-iter",              type=int,   default=None)
    p.add_argument("--bo-candidates-per-iter", type=int,   default=None)

    # RBM parameters
    p.add_argument("--rbm-steps", type=int, default=None)

    # Feedback
    p.add_argument("--feedback", action=argparse.BooleanOptionalAction, default=None,
                   help="Chain k-values: best mask at k_i warm-starts k_{i+1}")

    # k sweep — optional (may come from config)
    k_group = p.add_mutually_exclusive_group(required=False)
    k_group.add_argument("--ks",    nargs="+", type=int, help="Explicit k values")
    k_group.add_argument("--k-max", type=int,
                         help="Sweep all k from 0 to k-max (inclusive)")

    # Seed sweep — optional (may come from config)
    seed_group = p.add_mutually_exclusive_group(required=False)
    seed_group.add_argument("--seeds",     nargs="+", type=int,
                            help="Explicit list of seeds")
    seed_group.add_argument("--num-seeds", type=int,
                            help="Number of seeds starting from 42")

    # Output
    p.add_argument("--mosek-tol", type=float, default=None)
    p.add_argument("--out-root",  type=Path,  default=None)
    p.add_argument("--name",      type=str,   default=None,
                   help="Run name (default: auto-generated)")
    p.add_argument("--resume",    action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--force",     action="store_true", default=False)
    p.add_argument("--verbose",   action=argparse.BooleanOptionalAction, default=None)

    # Lookup table
    p.add_argument(
        "--no-lookup-table",
        dest="lookup_table",
        action="store_false",
        default=None,
        help=(
            "Disable the precomputed lookup table and use live MOSEK SDP "
            "evaluation instead."
        ),
    )
    p.add_argument(
        "--lookup-table",
        dest="lookup_table",
        action="store_true",
        help="Enable the precomputed lookup table (default when available).",
    )

    args = p.parse_args(argv)

    # ------------------------------------------------------------------
    # Merge: built-in defaults  ←  config file  ←  CLI flags
    # ------------------------------------------------------------------
    cfg: Dict[str, Any] = dict(_DEFAULTS)
    if args.config:
        file_cfg = load_config(args.config)
        cfg.update(file_cfg)

    # Helper: use CLI value when explicitly provided, else fall back to cfg
    def _get(cli_val, key: str, *, transform=None):
        val = cli_val if cli_val is not None else cfg.get(key)
        return transform(val) if (transform and val is not None) else val

    method       = _get(args.method,       "method")
    feedback     = _get(args.feedback,     "feedback")
    resume       = _get(args.resume,       "resume")
    verbose      = _get(args.verbose,      "verbose")
    mosek_tol    = _get(args.mosek_tol,    "mosek_tol",    transform=float)
    out_root     = Path(_get(args.out_root, "out_root"))
    use_lut      = _get(args.lookup_table, "lookup_table")
    run_name     = args.name  # only ever set by CLI

    if verbose and args.config:
        print(f"Config file: {args.config}")

    # Method-specific params: merge config sub-dict with CLI overrides
    mp_cfg = dict(cfg.get(method, {}))

    if method == "sa":
        method_params = {
            "steps":   mp_cfg.get("steps",   200),
            "T_start": mp_cfg.get("T_start", 0.3),
            "alpha":   mp_cfg.get("alpha",   0.95),
        }
        if args.sa_steps   is not None: method_params["steps"]   = args.sa_steps
        if args.sa_T_start is not None: method_params["T_start"] = args.sa_T_start
        if args.sa_alpha   is not None: method_params["alpha"]   = args.sa_alpha
    elif method == "pt":
        method_params = {
            "num_chains":      mp_cfg.get("num_chains",      0),
            "num_epochs":      mp_cfg.get("num_epochs",      10),
            "steps_per_epoch": mp_cfg.get("steps_per_epoch", 20),
            "T_min":           mp_cfg.get("T_min",           0.1),
            "T_max":           mp_cfg.get("T_max",           2.0),
        }
        if args.pt_chains          is not None: method_params["num_chains"]      = args.pt_chains
        if args.pt_epochs          is not None: method_params["num_epochs"]      = args.pt_epochs
        if args.pt_steps_per_epoch is not None: method_params["steps_per_epoch"] = args.pt_steps_per_epoch
        if args.pt_T_min           is not None: method_params["T_min"]           = args.pt_T_min
        if args.pt_T_max           is not None: method_params["T_max"]           = args.pt_T_max
        nc = method_params["num_chains"]
        method_params["num_chains"] = None if (nc is None or int(nc) <= 0) else int(nc)
    elif method == "bo":
        # init_train_size is the fra_Opt_Bell compat name for n_init
        n_init_default = mp_cfg.get("n_init") or mp_cfg.get("init_train_size", 10)
        # steps is the fra_Opt_Bell compat name for n_iter
        n_iter_default = mp_cfg.get("n_iter") or mp_cfg.get("steps", 40)
        method_params = {
            "beta":               mp_cfg.get("beta",               2.0),
            "n_init":             n_init_default,
            "n_iter":             n_iter_default,
            "candidates_per_iter": mp_cfg.get("candidates_per_iter", 200),
        }
        if args.bo_beta                is not None: method_params["beta"]               = args.bo_beta
        if args.bo_n_init              is not None: method_params["n_init"]             = args.bo_n_init
        if args.bo_n_iter              is not None: method_params["n_iter"]             = args.bo_n_iter
        if args.bo_candidates_per_iter is not None: method_params["candidates_per_iter"] = args.bo_candidates_per_iter
    elif method == "rbm":
        method_params = {
            "steps":   mp_cfg.get("steps",   200),
            "T_start": mp_cfg.get("T_start", 0.3),
            "alpha":   mp_cfg.get("alpha",   None),
        }
        if args.rbm_steps is not None: method_params["steps"] = args.rbm_steps
    else:
        method_params = {}

    # ------------------------------------------------------------------
    # Build k values  (CLI > config > default)
    # ------------------------------------------------------------------
    if args.ks is not None:
        k_values = sorted(set(args.ks))
    elif args.k_max is not None:
        k_values = list(range(args.k_max + 1))
    elif "ks" in cfg:
        k_values = sorted(set(cfg["ks"]))
    elif "k_values_to_test" in cfg:          # fra_Opt_Bell compat key
        k_values = sorted(set(cfg["k_values_to_test"]))
    elif "k_max" in cfg:
        k_values = list(range(int(cfg["k_max"]) + 1))
    else:
        print("ERROR: no k values specified. Use --ks, --k-max, "
              "or set them in the config file.", file=__import__("sys").stderr)
        return 1

    # ------------------------------------------------------------------
    # Build seeds  (CLI > config > default)
    # ------------------------------------------------------------------
    if args.seeds is not None:
        seeds = sorted(set(args.seeds))
    elif args.num_seeds is not None:
        seeds = list(range(42, 42 + args.num_seeds))
    elif "seeds" in cfg:
        seeds = sorted(set(cfg["seeds"]))
    elif "num_seeds" in cfg:
        base = int(cfg.get("seed", 42))
        seeds = list(range(base, base + int(cfg["num_seeds"])))
    else:
        print("ERROR: no seeds specified. Use --seeds, --num-seeds, "
              "or set them in the config file.", file=__import__("sys").stderr)
        return 1

    name = run_name or _default_name(method, feedback)

    run_dir = compute_and_save(
        method=method,
        method_params=method_params,
        k_values=k_values,
        seeds=seeds,
        mosek_tol=mosek_tol,
        out_root=out_root,
        name=name,
        resume=resume,
        force=args.force,
        verbose=verbose,
        feedback=feedback,
        use_lookup_table=use_lut,
    )

    print(f"Results -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
