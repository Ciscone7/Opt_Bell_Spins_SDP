"""scripts_fra package — shared utilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a JSON config file and return it as a plain dict.

    Parameters
    ----------
    path :
        Path to a ``.json`` config file.

    Returns
    -------
    dict
        Parsed config.  Keys whose names start with ``"_"`` (comments) are
        stripped automatically.

    Raises
    ------
    SystemExit
        If the file cannot be found or parsed.
    """
    path = Path(path)
    if not path.exists():
        print(f"ERROR: config file not found: {path}", file=sys.stderr)
        raise SystemExit(1)
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON in {path}: {exc}", file=sys.stderr)
        raise SystemExit(1)
    # Strip comment keys (keys starting with "_")
    return {k: v for k, v in data.items() if not k.startswith("_")}
