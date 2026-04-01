"""Artifact Manager — clean, organised persistence for experiment results.

Stores results as **plain files** (``meta.json`` + ``data.npz``) grouped
under a user-chosen run directory.

On-disk layout::

    <results_root>/<artifact_name>/v<version>/<run_name>/
        meta.json      ← config, provenance, timestamps
        data.npz       ← result arrays

Example usage (inside a script)::

    from spins_sdp.scripts.artifact_manager import ArtifactManager

    am = ArtifactManager(results_root="spins_sdp/results")

    run = am.create_run(
        artifact="spin_exact_ground_energy",
        name="heisenberg_periodic",
        config={...},
    )

    # Resume: load existing data if present
    existing = run.load_records(key="N", fields=FIELDS)
    # ... compute missing ...
    run.save_records(key="N", fields=FIELDS, records=existing)
    run.update_meta(Ns_present=sorted(existing.keys()))

    # Or for flat-table artifacts (sweeps):
    run.save_table(k=k_arr, seed=seed_arr, best_value=val_arr)
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    """UTC timestamp in compact ISO-8601 format."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def stable_json_dumps(obj: Any) -> str:
    """Deterministic JSON serialisation (sorted keys, no extra whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(config: Mapping[str, Any], *, n_chars: int = 16) -> str:
    """Short deterministic hash of a config dict."""
    payload = stable_json_dumps(config).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:n_chars]


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    _atomic_write_bytes(path, stable_json_dumps(obj).encode("utf-8"))


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Atomically write an ``.npz`` (write temp → rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez(tmp, **arrays)
    # np.savez appends .npz when suffix isn't already .npz
    actual_tmp = Path(str(tmp) + ".npz") if not str(tmp).endswith(".npz") else tmp
    os.replace(actual_tmp, path)


# ---------------------------------------------------------------------------
# RunDir — represents one run on disk
# ---------------------------------------------------------------------------

class RunDir:
    """Handle to a single run directory (``meta.json`` + ``data.npz``).

    You normally obtain a ``RunDir`` via :py:meth:`ArtifactManager.create_run`
    or :py:meth:`ArtifactManager.open_run`, not by constructing it directly.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.meta_path = self.path / "meta.json"
        self.data_path = self.path / "data.npz"

    @property
    def name(self) -> str:
        """The directory name (human-readable run name)."""
        return self.path.name

    @property
    def exists(self) -> bool:
        return self.path.exists()

    # ---- meta.json ----------------------------------------------------------

    def load_meta(self) -> Dict[str, Any]:
        """Load and return the full ``meta.json`` dict."""
        if not self.meta_path.exists():
            return {}
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def save_meta(self, meta: MutableMapping[str, Any]) -> None:
        """Atomically write ``meta.json``, preserving ``created_at``."""
        if self.meta_path.exists():
            try:
                old = json.loads(self.meta_path.read_text(encoding="utf-8"))
                if isinstance(old, dict) and "created_at" in old:
                    meta.setdefault("created_at", old["created_at"])
            except Exception:
                pass
        meta["updated_at"] = utc_now_iso()
        _atomic_write_json(self.meta_path, meta)

    def update_meta(self, **updates: Any) -> None:
        """Merge *updates* into the existing ``meta.json`` and save."""
        meta = self.load_meta()
        meta.update(updates)
        self.save_meta(meta)

    # ---- data.npz (flat table) ----------------------------------------------

    def save_table(self, **arrays: np.ndarray) -> None:
        """Atomically save column arrays as ``data.npz``.

        All 1-D arrays **must** have the same length (validated here).
        Higher-dimensional arrays (e.g. packed masks) are checked on their
        first axis only.

        Raises:
            ValueError: if array lengths are inconsistent.
        """
        if not arrays:
            return
        lengths = {
            name: int(arr.shape[0]) for name, arr in arrays.items() if arr.ndim >= 1
        }
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            detail = ", ".join(f"{n}={l}" for n, l in sorted(lengths.items()))
            raise ValueError(
                f"All arrays must have the same first-axis length.  Got: {detail}"
            )
        _atomic_save_npz(self.data_path, **arrays)

    def load_table(self) -> Dict[str, np.ndarray]:
        """Load ``data.npz`` and return a plain dict of arrays."""
        if not self.data_path.exists():
            return {}
        with np.load(self.data_path, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}

    # ---- data.npz (keyed records) -------------------------------------------

    def save_records(
        self,
        *,
        key: str,
        fields: Mapping[str, np.dtype],
        records: Mapping[int, Mapping[str, Any]],
    ) -> None:
        """Save a ``{key: {field: value}}`` dict as sorted column arrays.

        This is the "one row per unique key" convenience layer, useful when
        each row is identified by a single integer key (e.g. system size N).
        """
        keys_sorted = np.array(sorted(int(k) for k in records), dtype=int)
        arrays: Dict[str, np.ndarray] = {key: keys_sorted}
        for field, dtype in fields.items():
            arrays[field] = np.array(
                [records[int(k)][field] for k in keys_sorted], dtype=dtype
            )
        self.save_table(**arrays)

    def load_records(
        self,
        *,
        key: str,
        fields: Mapping[str, np.dtype],
    ) -> Dict[int, Dict[str, Any]]:
        """Load ``data.npz`` as a ``{key: {field: value}}`` dict."""
        if not self.data_path.exists():
            return {}
        data = np.load(self.data_path, allow_pickle=False)
        if key not in data:
            raise KeyError(f"Missing key array {key!r} in {self.data_path}")

        keys = data[key].astype(int)
        records: Dict[int, Dict[str, Any]] = {}
        for i, k in enumerate(keys):
            rec: Dict[str, Any] = {}
            for field, dtype in fields.items():
                if field not in data:
                    raise KeyError(f"Missing field {field!r} in {self.data_path}")
                value = data[field][i]
                if np.issubdtype(dtype, np.integer):
                    rec[field] = int(value)
                elif np.issubdtype(dtype, np.floating):
                    rec[field] = float(value)
                else:
                    rec[field] = value.item() if hasattr(value, "item") else value
            records[int(k)] = rec
        return records

    def __repr__(self) -> str:
        return f"RunDir({str(self.path)!r})"


# ---------------------------------------------------------------------------
# ArtifactManager — the main entry point
# ---------------------------------------------------------------------------

class ArtifactManager:
    """Manage experiment artifacts under a results root directory.

    Parameters
    ----------
    results_root : Path or str
        Top-level directory.  Each artifact type gets a sub-tree::

            results_root / <artifact> / v<version> / <run_name> /
    """

    def __init__(self, results_root: Path | str | None = None) -> None:
        if results_root is None:
            # Default: spins_sdp/results (relative to this file's package)
            results_root = Path(__file__).resolve().parents[1] / "results"
        self.root = Path(results_root)

    # ---- creating / opening runs -------------------------------------------

    def create_run(
        self,
        *,
        artifact: str,
        name: str,
        config: Mapping[str, Any],
        version: int = 1,
    ) -> RunDir:
        """Create (or reopen) a run directory and initialise its ``meta.json``.

        Parameters
        ----------
        artifact : str
            Artifact type, e.g. ``"spin_exact_ground_energy"``.
        name : str
            Human-readable directory name for this run.
        config : dict
            Full experiment configuration.  Stored in ``meta.json`` and used to
            compute a deterministic ``config_hash``.
        version : int
            Schema version (directory level ``v1``, ``v2``, …).

        Returns
        -------
        RunDir
            Handle to ``results_root / artifact / v<version> / name /``.
        """
        cfg_hash = config_hash(config)

        run = RunDir(self.root / artifact / f"v{version}" / name)
        run.path.mkdir(parents=True, exist_ok=True)

        # Build initial meta
        meta: Dict[str, Any] = dict(config)
        meta["config_hash"] = cfg_hash
        meta.setdefault("created_at", utc_now_iso())
        meta["updated_at"] = utc_now_iso()
        meta["python"] = sys.version
        meta["platform"] = {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        }
        run.save_meta(meta)
        return run

    def open_run(
        self,
        *,
        artifact: str,
        name: str,
        version: int = 1,
    ) -> RunDir:
        """Open an existing run directory (does **not** create it).

        Raises ``FileNotFoundError`` if the directory doesn't exist.
        """
        run = RunDir(self.root / artifact / f"v{version}" / name)
        if not run.exists:
            raise FileNotFoundError(f"Run directory not found: {run.path}")
        return run

    # ---- discovery ----------------------------------------------------------

    def list_runs(
        self,
        artifact: str,
        version: int = 1,
    ) -> List[RunDir]:
        """List all run directories for an artifact type."""
        base = self.root / artifact / f"v{version}"
        if not base.exists():
            return []
        return sorted(
            (RunDir(p) for p in base.iterdir() if p.is_dir()),
            key=lambda r: r.name,
        )

    def find(
        self,
        *,
        artifact: str,
        version: int = 1,
        name: str | None = None,
        config_hash_prefix: str | None = None,
        meta_query: Mapping[str, Any] | None = None,
    ) -> RunDir:
        """Find a run directory by *name*, *config hash prefix*, or *meta query*.

        Resolution order:
          1. Exact ``name`` lookup (fast path).
          2. Match ``config_hash`` field in ``meta.json`` by prefix.
          3. Recursive subset-match on ``meta.json`` fields.

        If multiple candidates match, the most recently updated run wins.

        Raises ``FileNotFoundError`` if nothing matches.
        """
        base = self.root / artifact / f"v{version}"

        # 1) By name
        if name is not None:
            run = RunDir(base / name)
            if not run.exists:
                raise FileNotFoundError(f"No run named {name!r} under {base}")
            return run

        # 2+3) Scan
        candidates: List[RunDir] = []
        for run in self.list_runs(artifact, version):
            try:
                meta = run.load_meta()
            except Exception:
                continue

            if config_hash_prefix is not None:
                stored_hash = meta.get("config_hash", "")
                if not stored_hash.startswith(config_hash_prefix):
                    continue

            if meta_query is not None:
                if not _meta_matches(meta, meta_query):
                    continue

            candidates.append(run)

        if not candidates:
            raise FileNotFoundError(
                f"No matching run under {base}.  "
                f"(config_hash_prefix={config_hash_prefix!r}, meta_query={meta_query!r})"
            )
        if len(candidates) == 1:
            return candidates[0]

        # Pick most recently updated
        return max(candidates, key=lambda r: r.load_meta().get("updated_at", ""))

    # ---- convenience --------------------------------------------------------

    @staticmethod
    def config_hash(config: Mapping[str, Any], *, n_chars: int = 16) -> str:
        """Compute a deterministic hash for a config dict (exposed for callers)."""
        return config_hash(config, n_chars=n_chars)


# ---------------------------------------------------------------------------
# Meta-matching helper (recursive subset match with float tolerance)
# ---------------------------------------------------------------------------

def _values_close(a: Any, b: Any, *, atol: float = 1e-12) -> bool:
    if isinstance(a, (int, str, bool)) or a is None:
        return a == b
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= atol
        except Exception:
            return a == b
    return a == b


def _meta_matches(meta: Any, query: Any) -> bool:
    """Return ``True`` if *query* is a recursive subset of *meta*."""
    if isinstance(query, Mapping):
        if not isinstance(meta, Mapping):
            return False
        return all(
            k in meta and _meta_matches(meta[k], v) for k, v in query.items()
        )
    if isinstance(query, list):
        return meta == query
    return _values_close(meta, query)
