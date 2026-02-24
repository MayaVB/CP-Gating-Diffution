import json
import math
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np


def load_calibration_json(path: str) -> Dict[str, Any]:
    """
    Load a calibration result previously saved by save_calibration_json().

    Returns the full payload dict; the most important key is 'tau'.
    """
    with open(path, "r") as f:
        return json.load(f)


def calibrate_tau_alpha(traj_logs: List, alpha: float) -> Dict[str, Any]:
    """
    Compute a split-conformal threshold tau from a list of GateTrajectoryLog objects.

    Each log must have had finalize() called so that trajectory_score is available.

    Threshold formula (Angelopoulos & Bates 2022):
        n   = number of calibration examples
        idx = ceil((n+1)*(1-alpha)) - 1    # 0-based, clamped to [0, n-1]
        tau = sorted(G)[idx]

    A sample passes the gate iff its score <= tau.
    Coverage guarantee: P(score <= tau) >= 1 - alpha.

    Args:
        traj_logs : List[GateTrajectoryLog]  (finalize() must have been called on each)
        alpha     : miscoverage level in (0, 1); e.g. 0.1 for 90% coverage

    Returns:
        dict with: tau, alpha, n, idx, G_mean, G_std, G_min, G_max
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be strictly in (0, 1), got {alpha!r}.")

    G = np.array([tl.trajectory_score for tl in traj_logs], dtype=np.float64)
    n = len(G)

    if n == 0:
        raise ValueError("traj_logs is empty — nothing to calibrate.")

    if not np.isfinite(G).all():
        n_bad = int(np.sum(~np.isfinite(G)))
        raise ValueError(
            f"{n_bad}/{n} trajectory scores are non-finite (inf or nan). "
            "Check compute_gate_score / finalize() calls."
        )

    if n < 20:
        warnings.warn(
            f"Calibration set is small (n={n} < 20). "
            "Conformal coverage guarantee may not hold reliably.",
            UserWarning,
            stacklevel=2,
        )

    idx = math.ceil((n + 1) * (1 - alpha)) - 1
    idx = min(max(idx, 0), n - 1)
    tau = float(np.sort(G)[idx])

    return {
        "tau":    tau,
        "alpha":  float(alpha),
        "n":      n,
        "idx":    idx,
        "G_mean": float(G.mean()),
        "G_std":  float(G.std()),
        "G_min":  float(G.min()),
        "G_max":  float(G.max()),
    }


def save_calibration_json(
    path: str,
    result_dict: Dict[str, Any],
    gate_name: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save calibration result to a JSON file.

    Output format:
        {
          "timestamp": "<ISO-8601 UTC>",
          "gate_name": "<name>",
          <result_dict fields>,
          <extra_meta fields>
        }

    Args:
        path        : output file path (e.g. "results/_calib_tau.json")
        result_dict : dict returned by calibrate_tau_alpha()
        gate_name   : string identifier for the gate (e.g. "leakage_gate")
        extra_meta  : optional dict of additional fields (e.g. vars(args)); default None
    """
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gate_name": gate_name,
        **result_dict,
        **(extra_meta or {}),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
