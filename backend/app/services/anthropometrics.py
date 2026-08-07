"""
India-oriented adult height/weight defaults when the user does not know
their measurements. Midpoints of common published age-band ranges
(illustrative averages — regional variation applies).

Used only as a fallback for Mifflin–St Jeor when height_cm / weight_kg
are missing. Real user values always win.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


# (age_lo, age_hi inclusive) → (height_cm midpoint, weight_kg midpoint)
_MALE_BANDS = [
    (18, 25, 169.5, 65.0),   # 167–172 cm, 60–70 kg
    (26, 35, 170.5, 70.0),   # 168–173, 65–75
    (36, 45, 169.5, 73.0),   # 167–172, 68–78
    (46, 60, 167.5, 75.0),   # 165–170, 70–80
]

_FEMALE_BANDS = [
    (18, 25, 156.0, 55.0),   # 152–160, 50–60
    (26, 35, 158.0, 60.0),   # 154–162, 55–65
    (36, 45, 157.0, 63.0),   # 153–161, 58–68
    (46, 60, 155.0, 65.0),   # 151–159, 60–70
]


def _pick_band(age: int, bands) -> Tuple[float, float]:
    for lo, hi, h, w in bands:
        if lo <= age <= hi:
            return h, w
    if age < bands[0][0]:
        return bands[0][2], bands[0][3]
    return bands[-1][2], bands[-1][3]


def india_default_height_weight(age: Optional[int], sex: Optional[str]) -> Optional[Dict[str, float]]:
    """
    Return {"height_cm", "weight_kg"} midpoints for age+sex, or None if
    age/sex insufficient.
    """
    if age is None:
        return None
    try:
        age_i = int(age)
    except (TypeError, ValueError):
        return None
    if age_i < 10 or age_i > 100:
        return None

    s = (sex or "").strip().lower()
    if s in ("male", "m", "man", "men"):
        h, w = _pick_band(age_i, _MALE_BANDS)
    elif s in ("female", "f", "woman", "women"):
        h, w = _pick_band(age_i, _FEMALE_BANDS)
    else:
        return None
    return {"height_cm": h, "weight_kg": w}


def resolve_height_weight(
    *,
    age: Optional[Any],
    sex: Optional[str],
    height_cm: Optional[Any] = None,
    weight_kg: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Prefer measured values; fill gaps from India age/sex midpoints.

    Returns:
      height_cm, weight_kg,
      height_estimated (bool), weight_estimated (bool),
      anthropometrics_source: "measured" | "india_age_sex_midpoint" | "mixed" | None
    """
    h_est = height_cm is None or height_cm == ""
    w_est = weight_kg is None or weight_kg == ""
    try:
        h_val = float(height_cm) if not h_est else None
    except (TypeError, ValueError):
        h_val, h_est = None, True
    try:
        w_val = float(weight_kg) if not w_est else None
    except (TypeError, ValueError):
        w_val, w_est = None, True

    defaults = india_default_height_weight(age, sex) if (h_est or w_est) else None
    if h_est and defaults:
        h_val = defaults["height_cm"]
    if w_est and defaults:
        w_val = defaults["weight_kg"]

    if h_val is None or w_val is None:
        return {
            "height_cm": h_val,
            "weight_kg": w_val,
            "height_estimated": h_est,
            "weight_estimated": w_est,
            "anthropometrics_source": None,
        }

    if h_est and w_est:
        source = "india_age_sex_midpoint"
    elif h_est or w_est:
        source = "mixed"
    else:
        source = "measured"

    return {
        "height_cm": h_val,
        "weight_kg": w_val,
        "height_estimated": bool(h_est and defaults),
        "weight_estimated": bool(w_est and defaults),
        "anthropometrics_source": source,
    }
