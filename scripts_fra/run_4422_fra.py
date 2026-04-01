"""Bell NPA optimisation — I4422 campaign (4 settings, 2 outcomes, bipartite).

This script is the SpinsSDP-framework equivalent of fra_Opt_Bell's
``run_4422_campaign.py``.  It runs a batch campaign over multiple 4422
Bell expressions loaded from a text file, performing NPA basis selection
optimisation on each expression using the clean SpinsSDP abstractions:

  - :mod:`bell.bell_logic` for word algebra and NPA basis generation.
  - :mod:`bell.bell_optimize` for the SDP objective and sweep logic.
  - :mod:`artifact_manager` for structured, resumable result storage
    (``meta.json`` + ``data.npz`` instead of HDF5).

On-disk layout (one run directory per expression)::

    results/bell_optimization_sweep/v1/<run_name>_expr{idx}/
        meta.json   ← config, provenance, timestamps
        data.npz    ← run_idx, k, seed, best_value, elapsed_s,
                       n_obj_evals, mask_bits

Physics constants (hardcoded, matching fra_Opt_Bell convention):
    m_A = m_B = 4 (four measurement settings per party)
    d_A = d_B = 2 (binary outcomes → one projector per setting)

NPA hierarchy: start_level=1, end_level=2.

Input file format
-----------------
Same as fra_Opt_Bell's ``all_4422_bell_expressions_MYORDER.txt``:
a text file where each row contains space-separated floating-point
coefficients.  The number of coefficients equals the size of the
``local1`` basis minus the identity (i.e. the projector labels in
Alice's and Bob's measurement bases + joint projectors).  The coefficient
at position ``i`` multiplies the corresponding moment label.

Usage examples::

    # Run PT on all 4422 expressions, all k from 0 to N, 10 seeds
    python -m scripts_fra.run_4422_fra \\
        --input data/all_4422_bell_expressions_MYORDER.txt \\
        --method pt --k-max-auto --num-seeds 10

    # Run SA on expressions 0 and 1 only, k={0,5,10}, 5 seeds
    python -m scripts_fra.run_4422_fra \\
        --input data/all_4422_bell_expressions_MYORDER.txt \\
        --indices 0 1 --method sa \\
        --ks 0 5 10 --num-seeds 5

    # Resume a previous run (skip already completed jobs)
    python -m scripts_fra.run_4422_fra \\
        --input data/all_4422_bell_expressions_MYORDER.txt \\
        --method pt --k-max-auto --num-seeds 10 --resume
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Allow running as "python scripts_fra/run_4422_fra.py" from the project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from artifact_manager import ArtifactManager, RunDir
from bell.bell_logic import (
    BellScenario,
    BellWord,
    BellOperator,
    IDENTITY,
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

REFERENCE_VALUES = [
    1.2071, 1.25, 1.2361, 1.2596, 1.4365, 1.197, 2.6214, 1.4353, 1.2321, 3.5412,
    4.8785, 3.6056, 2.5, 1.238, 1.056, 2.4365, 1.4495, 2.4548, 3.4207, 2.4617,
    2.6139, 3.6384, 3.6188, 1.25, 2.4103, 1.25, 1.2407, 3.6714, 2.1812, 3.4307,
    2.3056, 1.4145, 2.4459, 1.5923, 2.4414, 2.4693, 1.4332, 2.4158, 2.3871, 3.6402,
    2.6035, 3.627, 3.5902, 3.609, 3.6186, 3.5996, 3.5823, 2.616, 3.6151, 2.393,
    2.7576, 3.5283, 3.6023, 2.3909, 2.5356, 2.5, 2.3956, 2.3423, 2.7308, 2.6731,
    3.6678, 3.6244, 3.614, 3.6651, 3.6849, 3.6692, 2.6248, 2.5686, 3.6943, 3.6933,
    3.6706, 5.8156, 5.8176, 3.9643, 4.7878, 4.7596, 4.8291, 4.75, 4.8382, 4.8024,
    4.8406, 2.7652, 3.8556, 4.5674, 4.6742, 4.6862, 2.6012, 4.8398, 4.8814, 3.4288,
    2.4075, 3.5971, 2.427, 2.2133, 2.3642, 2.2657, 2.2459, 2.2229, 5.9457, 5.9539,
    5.9627, 5.934, 3.7572, 4.7645, 4.75, 3.75, 4.745, 3.6133, 2.6275, 3.6093,
    3.6135, 4.0523, 4.9051, 4.9167, 5.0179, 3.8134, 6.0135, 3.8264, 4.8704, 4.941,
    4.5441, 3.6147, 3.5007, 3.5971, 5.9717, 5.9676, 6.0036, 3.9417, 5.9721, 4.8265,
    4.7994, 4.8053, 3.7066, 4.7491, 4.7249, 4.8089, 4.7399, 6.1497, 4.9763, 6.0156,
    5.8489, 6.0, 6.0742, 5.111, 3.8195, 6.0296, 6.0096, 3.4917, 3.5629, 8.2993,
    5.0648, 4.0999, 3.9802, 4.9295, 6.902, 5.067, 5.9418, 3.7931, 5.7018, 6.0246,
    6.0653, 6.0189, 6.052, 6.0076, 6.0641, 3.3738, 3.4198, 5.1205, 7.2675, 8.325,
    5.1584, 8.4377, 7.5876, 10.7261
]

M_A = 4  # settings for Alice
M_B = 4  # settings for Bob
D_A = 2  # outcomes for Alice
D_B = 2  # outcomes for Bob

SCENARIO = BellScenario(m_A=M_A, m_B=M_B, d_A=D_A, d_B=D_B)

# NPA levels
START_LEVEL = 1
END_LEVEL   = 2

ARTIFACT = "bell_optimization_sweep"


# ---------------------------------------------------------------------------
# Operator construction from coefficient vector
# ---------------------------------------------------------------------------

def _local1_labels(scenario: BellScenario) -> List[BellWord]:
    """Build the ordered list of moment labels at NPA 'local1' level.

    'local1' is the level used by fra_Opt_Bell to assign coefficients:
    it contains the identity, all single-party projectors, and all
    joint projectors E_{0|x} F_{0|y}.

    This matches the ``behaviour.generate_npa("local1")`` logic from
    fra_Opt_Bell's ``bell_utils.py`` (which returns a list of strings
    starting with the identity, then Alice projectors, then Bob
    projectors, then joint projectors A_i B_j).
    """
    labels: List[BellWord] = [IDENTITY]

    # Alice single-party projectors E_{0|x} for x in 0..m_A-1
    for x in range(scenario.m_A):
        labels.append(alice_projector(scenario, x, 0))

    # Bob single-party projectors F_{0|y} for y in 0..m_B-1
    for y in range(scenario.m_B):
        labels.append(bob_projector(scenario, y, 0))

    # Joint projectors E_{0|x} F_{0|y}
    for x in range(scenario.m_A):
        for y in range(scenario.m_B):
            w = multiply_words(alice_projector(scenario, x, 0),
                               bob_projector(scenario, y, 0))
            if w is not None:
                labels.append(w)

    return labels


def build_operator_from_coeffs(
    coeffs: List[float],
    labels: List[BellWord],
    scenario: BellScenario,
) -> BellOperator:
    """Build a BellOperator from a coefficient vector.

    Parameters
    ----------
    coeffs :
        Coefficient for each label in ``labels[1:]`` (the identity is
        excluded — it contributes a constant and does not affect the SDP
        optimum).
    labels :
        Ordered list of BellWords (first element is identity).
    scenario :
        Bell scenario (needed for completeness expansion).

    Returns
    -------
    BellOperator
        Expanded operator (identity-subtracted, completeness-substituted)
        ready for SDP assembly.
    """
    # labels[0] is identity; coefficients apply to labels[1:]
    op: BellOperator = {}
    for i, coef in enumerate(coeffs):
        if coef == 0.0:
            continue
        word = labels[i + 1]  # skip identity
        op[word] = op.get(word, 0.0) + coef

    return expand_bell_operator(op, scenario)


def load_expressions(filepath: str, labels: List[BellWord]) -> List[BellOperator]:
    """Load all Bell expressions from the 4422 campaign input file.

    Each row of the file contains ``len(labels) - 1`` space-separated
    coefficients (the identity label is omitted from the file, matching
    fra_Opt_Bell's ``create_dict_list`` which assigns labels[1:] to the
    per-row values).

    Parameters
    ----------
    filepath :
        Path to the text file (e.g. ``all_4422_bell_expressions_MYORDER.txt``).
    labels :
        Ordered BellWord labels (from :func:`_local1_labels`).
        Length must equal number of file columns + 1 (for identity).

    Returns
    -------
    list[BellOperator]
        Expanded Bell operators, one per row.
    """
    scenario = SCENARIO
    operators: List[BellOperator] = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # fra_Opt_Bell assigns labels[1:] (len = len(labels)-1)
            n_labels = len(labels) - 1
            if len(parts) < n_labels:
                continue  # skip malformed rows
            coeffs = [float(v) for v in parts[:n_labels]]
            op = build_operator_from_coeffs(coeffs, labels, scenario)
            operators.append(op)

    return operators


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

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


def _checkpoint(
    run: RunDir,
    run_idx, k, seed, best_value, elapsed_s, n_obj_evals, mask_bits_list,
) -> None:
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
# Per-expression sweep
# ---------------------------------------------------------------------------

def run_expression(
    *,
    expr_idx: int,
    bell_operator: BellOperator,
    starting_set: List[BellWord],
    adding_set: List[BellWord],
    method: str,
    method_params: Dict[str, Any],
    k_values: List[int],
    seeds: List[int],
    mosek_tol: float,
    out_root: Path,
    run_name: str,
    resume: bool,
    force: bool,
    verbose: bool,
    feedback: bool,
) -> str:
    """Run and save the optimisation sweep for a single Bell expression.

    Parameters
    ----------
    expr_idx :
        0-based index of this expression in the input file.
    bell_operator :
        Pre-built, expanded BellOperator for this expression.
    starting_set, adding_set :
        Fixed NPA basis sets (shared across all expressions).
    run_name :
        Base run name; the expression index is appended automatically:
        ``f"{run_name}_expr{expr_idx+1}"``.

    Returns
    -------
    str
        Path of the run directory.
    """
    scenario = SCENARIO
    L = len(adding_set)
    expr_run_name = f"{run_name}_expr{expr_idx + 1}"

    adding_set_hash = _stable_bellword_hash(adding_set)

    config: Dict[str, Any] = {
        "expression": f"4422_expr_{expr_idx + 1}",
        "expression_idx": expr_idx,
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
    }

    am = ArtifactManager(out_root)
    run = am.create_run(artifact=ARTIFACT, name=expr_run_name, config=config)

    # ------------------------------------------------------------------
    # Load existing checkpoint
    # ------------------------------------------------------------------
    existing_results: Dict[Tuple[int, int], BellOptimizationResult] = {}
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

    if not any(
        (kv, s) not in existing_results
        for kv in k_values for s in seeds
    ):
        if verbose:
            print(f"  Expression {expr_idx + 1}: all jobs done, skipping.")
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
            _acc_run_idx    = list(arr.get("run_idx", []))
            _acc_k          = list(arr.get("k", []))
            _acc_seed       = list(arr.get("seed", []))
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

    return str(run.path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_run_name(method: str, feedback: bool) -> str:
    scenario = SCENARIO
    parts = [
        "I4422_campaign",
        f"mA{scenario.m_A}_mB{scenario.m_B}",
        f"dA{scenario.d_A}_dB{scenario.d_B}",
        f"lv{START_LEVEL}to{END_LEVEL}",
        method,
    ]
    if feedback:
        parts.append("fb")
    return "_".join(parts)


def _cli_parser_4422(add_required_groups: bool = True) -> argparse.ArgumentParser:
    """Build the argument parser for run_4422_fra.

    When ``add_required_groups=False`` the k-sweep and seed groups are added
    as optional so that values can come from a config file instead.
    """
    DEFAULT_INPUT = str(
        Path(__file__).resolve().parents[1]
        / "data" / "all_4422_bell_expressions_MYORDER.txt"
    )

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

    # Input file
    p.add_argument("--input", dest="input_file", default=None,
                   help="Path to 4422 Bell expressions file")

    # Expression indices to run (default: all)
    p.add_argument("--indices", nargs="+", type=int, default=None,
                   help="0-based indices of expressions to process (default: all)")

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
    k_group.add_argument("--ks",         nargs="+", type=int,
                         help="Explicit k values")
    k_group.add_argument("--k-max",      type=int,
                         help="Sweep k from 0 to k-max (inclusive)")
    k_group.add_argument("--k-max-auto", action="store_true", default=False,
                         help="Sweep k from 0 to len(adding_set) automatically")

    # Seed sweep — optional (may come from config)
    seed_group = p.add_mutually_exclusive_group(required=False)
    seed_group.add_argument("--seeds",     nargs="+", type=int)
    seed_group.add_argument("--num-seeds", type=int,
                            help="Number of seeds starting from 42")

    # Output
    p.add_argument("--mosek-tol", type=float, default=None)
    p.add_argument("--out-root",  type=Path,  default=None)
    p.add_argument("--name",      type=str,   default=None,
                   help="Base run name (expression index appended automatically)")
    p.add_argument("--resume",    action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--force",     action="store_true", default=False)
    p.add_argument("--verbose",   action=argparse.BooleanOptionalAction, default=None)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    from scripts_fra import load_config  # shared helper

    # Built-in script defaults (lowest priority)
    _DEFAULTS: Dict[str, Any] = {
        "input_file": str(
            Path(__file__).resolve().parents[1]
            / "data" / "all_4422_bell_expressions_MYORDER.txt"
        ),
        "method":     "pt",
        "start_level": START_LEVEL,
        "end_level":   END_LEVEL,
        "k_max_auto":  True,
        "num_seeds":   10,
        "mosek_tol":   1e-6,
        "feedback":    False,
        "resume":      True,
        "verbose":     True,
        "error_to_stop": 0.1,
        "out_root":    str(Path(__file__).resolve().parents[1] / "results"),
        "sa":  {"steps": 200, "T_start": 1.0, "alpha": 0.95},
        "pt":  {"num_chains": 0, "num_epochs": 5, "steps_per_epoch": 40,
                "T_min": 0.1, "T_max": 2.0},
        "bo":  {"beta": 2.0, "n_init": 5, "n_iter": 10,
                "candidates_per_iter": 100},
        "rbm": {"steps": 200},
    }

    p = _cli_parser_4422()
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

    method       = _get(args.method,    "method")
    feedback     = _get(args.feedback,  "feedback")
    resume       = _get(args.resume,    "resume")
    verbose      = _get(args.verbose,   "verbose")
    mosek_tol    = _get(args.mosek_tol, "mosek_tol", transform=float)
    out_root     = Path(_get(args.out_root, "out_root"))
    input_file   = _get(args.input_file, "input_file")
    run_name     = args.name
    start_level  = int(cfg.get("start_level", START_LEVEL))
    end_level    = int(cfg.get("end_level",   END_LEVEL))

    # error_to_stop: None → no early stopping, float in (0,1] → stop k-sweep
    # when relative error vs reference value drops below threshold
    _ets_raw = cfg.get("error_to_stop", 0.1)
    error_to_stop: Optional[float] = (
        None if _ets_raw is None
        else max(0.0, min(1.0, float(_ets_raw)))
    )

    # Method-specific params: merge config sub-dict with CLI overrides
    mp_cfg = dict(cfg.get(method, {}))

    if method == "sa":
        method_params = {
            "steps":   _get(args.sa_steps,   "steps",   transform=int) or mp_cfg.get("steps",   200),
            "T_start": _get(args.sa_T_start,  "T_start", transform=float) or mp_cfg.get("T_start", 1.0),
            "alpha":   _get(args.sa_alpha,    "alpha",   transform=float) or mp_cfg.get("alpha",   0.95),
        }
        if args.sa_steps   is not None: method_params["steps"]   = args.sa_steps
        if args.sa_T_start is not None: method_params["T_start"] = args.sa_T_start
        if args.sa_alpha   is not None: method_params["alpha"]   = args.sa_alpha
    elif method == "pt":
        method_params = {
            "num_chains":      mp_cfg.get("num_chains",      0),
            "num_epochs":      mp_cfg.get("num_epochs",      5),
            "steps_per_epoch": mp_cfg.get("steps_per_epoch", 40),
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
        n_init_default = mp_cfg.get("n_init") or mp_cfg.get("init_train_size", 5)
        # steps is the fra_Opt_Bell compat name for n_iter
        n_iter_default = mp_cfg.get("n_iter") or mp_cfg.get("steps", 10)
        method_params = {
            "beta":               mp_cfg.get("beta",               2.0),
            "n_init":             n_init_default,
            "n_iter":             n_iter_default,
            "candidates_per_iter": mp_cfg.get("candidates_per_iter", 100),
        }
        if args.bo_beta                is not None: method_params["beta"]               = args.bo_beta
        if args.bo_n_init              is not None: method_params["n_init"]             = args.bo_n_init
        if args.bo_n_iter              is not None: method_params["n_iter"]             = args.bo_n_iter
        if args.bo_candidates_per_iter is not None: method_params["candidates_per_iter"] = args.bo_candidates_per_iter
    elif method == "rbm":
        method_params = {
            "steps":   mp_cfg.get("steps",   200),
            "T_start": mp_cfg.get("T_start", 1.0),
            "alpha":   mp_cfg.get("alpha",   None),
        }
        if args.rbm_steps is not None: method_params["steps"] = args.rbm_steps
    else:
        method_params = {}

    # ------------------------------------------------------------------
    # Build NPA basis (shared across all expressions)
    # ------------------------------------------------------------------
    scenario = SCENARIO
    starting_set, adding_set, _ = build_bell_basis_sets(
        scenario=scenario,
        start_level=start_level,
        end_level=end_level,
    )
    L = len(adding_set)

    if verbose:
        print(
            f"I4422 campaign: scenario ({scenario.m_A},{scenario.m_B}), "
            f"d=({scenario.d_A},{scenario.d_B}), "
            f"NPA{start_level}→NPA{end_level}"
        )
        print(f"Starting set: {len(starting_set)}, Adding set (N): {L}")
        if args.config:
            print(f"Config file: {args.config}")

    # ------------------------------------------------------------------
    # Build k values  (CLI > config > default)
    # ------------------------------------------------------------------
    if args.ks is not None:
        k_values = sorted(set(args.ks))
    elif args.k_max is not None:
        k_values = list(range(args.k_max + 1))
    elif args.k_max_auto:
        k_values = list(range(L + 1))
    elif "ks" in cfg:
        k_values = sorted(set(cfg["ks"]))
    elif "k_values_to_test" in cfg:          # fra_Opt_Bell compat key
        k_values = sorted(set(cfg["k_values_to_test"]))
    elif "k_max" in cfg:
        k_values = list(range(int(cfg["k_max"]) + 1))
    elif cfg.get("k_max_auto", False):
        k_values = list(range(L + 1))
    else:
        print("ERROR: no k values specified. Use --ks, --k-max, --k-max-auto, "
              "or set them in the config file.", file=__import__("sys").stderr)
        return 1
    k_values = [k for k in k_values if 0 <= k <= L]

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

    # ------------------------------------------------------------------
    # Load expressions
    # ------------------------------------------------------------------
    labels = _local1_labels(scenario)

    if verbose:
        print(f"Loading expressions from: {input_file}")
    operators = load_expressions(input_file, labels)
    if not operators:
        print("No expressions loaded. Check the input file path and format.")
        return 1
    if verbose:
        print(f"Loaded {len(operators)} Bell expressions.")

    # ------------------------------------------------------------------
    # Determine which expression indices to process
    # ------------------------------------------------------------------
    # CLI > config > default (all expressions)
    if args.indices is not None:
        indices = args.indices
    elif cfg.get("indices") is not None:
        indices = cfg["indices"]
    else:
        indices = list(range(len(operators)))

    indices = [i for i in indices if 0 <= i < len(operators)]

    base_name = run_name or _default_run_name(method, feedback)

    # ------------------------------------------------------------------
    # Run sweep for each expression
    # ------------------------------------------------------------------
    import time
    t_total = time.time()

    for expr_idx in indices:
        bell_operator = operators[expr_idx]
        ref_value = REFERENCE_VALUES[expr_idx] if error_to_stop is not None else None

        if verbose:
            ref_str = f", ref={ref_value:.4f}" if ref_value is not None else ""
            print(
                f"\n--- Expression {expr_idx + 1}/{len(operators)} "
                f"({len(k_values)} k-values, {len(seeds)} seeds{ref_str}) ---"
            )

        if error_to_stop is None:
            # Original behaviour: run all k values in one call
            run_dir = run_expression(
                expr_idx=expr_idx,
                bell_operator=bell_operator,
                starting_set=starting_set,
                adding_set=adding_set,
                method=method,
                method_params=method_params,
                k_values=k_values,
                seeds=seeds,
                mosek_tol=mosek_tol,
                out_root=out_root,
                run_name=base_name,
                resume=resume,
                force=args.force,
                verbose=verbose,
                feedback=feedback,
            )
        else:
            # Early-stop mode: iterate k values one at a time
            run_dir = None
            best_across_k: Optional[float] = None  # best value seen so far

            for k in k_values:
                run_dir = run_expression(
                    expr_idx=expr_idx,
                    bell_operator=bell_operator,
                    starting_set=starting_set,
                    adding_set=adding_set,
                    method=method,
                    method_params=method_params,
                    k_values=[k],
                    seeds=seeds,
                    mosek_tol=mosek_tol,
                    out_root=out_root,
                    run_name=base_name,
                    resume=resume,
                    force=args.force,
                    verbose=verbose,
                    feedback=feedback,
                )

                # Read the best value for this k across all seeds from the
                # accumulator (cheapest: re-load from disk)
                try:
                    from artifact_manager import ArtifactManager as _AM
                    _am  = _AM(out_root)
                    _run = _am.create_run(
                        artifact=ARTIFACT,
                        name=f"{base_name}_expr{expr_idx + 1}",
                        config={},   # read-only; config ignored if run exists
                    )
                    _, _arr = _load_existing_runs(_run, len(adding_set))
                    if _arr:
                        _k_mask = np.array(_arr["k"]) == k
                        if _k_mask.any():
                            best_at_k = float(np.min(np.array(_arr["best_value"])[_k_mask]))
                            if best_across_k is None or best_at_k < best_across_k:
                                best_across_k = best_at_k
                except Exception:
                    best_across_k = None

                if best_across_k is not None and ref_value is not None:
                    rel_err = abs(best_across_k - ref_value) / abs(ref_value)
                    if verbose:
                        print(
                            f"  k={k}: best={best_across_k:.6f}, "
                            f"ref={ref_value:.4f}, rel_err={rel_err:.4f}"
                        )
                    if rel_err < error_to_stop:
                        if verbose:
                            print(
                                f"  → rel_err={rel_err:.4f} < error_to_stop={error_to_stop:.4f}; "
                                f"skipping remaining k values for expression {expr_idx + 1}."
                            )
                        break

        if verbose and run_dir:
            print(f"  Results → {run_dir}")

    elapsed = time.time() - t_total
    print(f"\nCampaign complete. Total time: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

