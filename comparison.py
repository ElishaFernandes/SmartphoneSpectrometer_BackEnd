"""
Compare saved sessions and run calibration regressions.

compare_sessions:
    - Load 2+ sessions by id.
    - Return overlay-ready arrays for reflectance, derivatives, and ratios so
      the Flutter frontend can plot them on shared axes.

run_calibration:
    - Filter to sessions that have a concentration value.
    - Require >=2 such sessions.
    - Fit y = m*x + b where x is integrated_reflectance (default) or
      max-peak reflectance, and y is concentration.
    - Return slope, intercept, R^2, and equation string.
"""
from __future__ import annotations

from typing import List

from scipy.stats import linregress

from database import get_session_record


# ---------- comparison ----------

def compare_sessions(db, session_ids: List[str]) -> dict:
    if len(session_ids) < 2:
        raise ValueError("Need at least 2 sessions to compare.")

    sessions = []
    for sid in session_ids:
        rec = get_session_record(db, sid)
        if rec is None:
            raise ValueError(f"Session not found: {sid}")
        sessions.append(rec)

    overlay = []
    for rec in sessions:
        overlay.append({
            "id": rec["id"],
            "preset": rec["preset"],
            "concentration": rec.get("concentration"),
            "wavelengths_nm": rec["wavelength_data"],
            "reflectance": rec["reflectance_data"],
            "first_derivative": rec.get("derivative_data"),
            "second_derivative": rec.get("second_derivative_data"),
            "ratios": rec.get("ratios"),
            "integrated_reflectance": rec.get("integrated_reflectance"),
        })

    # Pull each ratio key across all sessions for histogram-style overlay.
    ratio_keys = ["R_over_G", "R_over_B", "G_over_B", "G_over_R"]
    ratio_overlay = {
        k: [s["ratios"].get(k) if s.get("ratios") else None for s in overlay]
        for k in ratio_keys
    }

    return {
        "n_sessions": len(sessions),
        "overlay": overlay,
        "ratio_overlay": ratio_overlay,
    }


# ---------- calibration ----------

def run_calibration(db, session_ids: List[str],
                    x_metric: str = "integrated_reflectance") -> dict:
    """Linear regression y = m*x + b. y is concentration, x is the chosen metric."""
    if len(session_ids) < 2:
        raise ValueError("Need at least 2 sessions for calibration.")

    sessions = []
    for sid in session_ids:
        rec = get_session_record(db, sid)
        if rec is None:
            raise ValueError(f"Session not found: {sid}")
        sessions.append(rec)

    labelled = [s for s in sessions if s.get("concentration") is not None]
    if len(labelled) < 2:
        raise ValueError(
            "Need at least 2 sessions with concentration values for calibration."
        )

    if x_metric == "integrated_reflectance":
        xs = [float(s["integrated_reflectance"] or 0.0) for s in labelled]
    elif x_metric == "peak_reflectance":
        xs = []
        for s in labelled:
            peaks = s.get("peak_values") or []
            xs.append(max((float(p["reflectance"]) for p in peaks), default=0.0))
    else:
        raise ValueError(f"Unknown x_metric: {x_metric!r}")

    ys = [float(s["concentration"]) for s in labelled]

    res = linregress(xs, ys)
    slope = float(res.slope)
    intercept = float(res.intercept)
    r2 = float(res.rvalue ** 2)

    return {
        "x_metric": x_metric,
        "n_points": len(xs),
        "x": xs,
        "y": ys,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r2,
        "equation": f"y = {slope:.6g} * x + {intercept:.6g}",
        "session_ids_used": [s["id"] for s in labelled],
    }