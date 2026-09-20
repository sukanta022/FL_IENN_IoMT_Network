"""fl_ienn/common/schema.py

Column registry and leakage guard for FL-IENN Step 0.

Frozen source of truth: docs/FROZEN_ARCHITECTURE.md, Sections 1.7, 2, 4, 9 (FD-1),
and the Step 0 task brief.

Naming families (Section 2, Frozen Decision 1):
    gt_*          ground truth, evaluation only
    eka_prior_*   Session Layer declaration, EKA only
    pred_*        IENN outputs
    enc_*         encryption routing
No component may read a variable from another family. The leakage guard below
enforces that for any "feature" frame handed to a model.
"""
from __future__ import annotations

import re
from typing import Final

# ---------------------------------------------------------------------------
# Label encodings (Section 1.4 / 1.1)
# ---------------------------------------------------------------------------

HEALTH_TO_INT: Final[dict[str, int]] = {
    "Normal":   0,
    "Moderate": 1,
    "Critical": 2,
}
INT_TO_HEALTH: Final[dict[int, str]] = {v: k for k, v in HEALTH_TO_INT.items()}

SENSITIVITY_TO_INT: Final[dict[str, int]] = {
    "Low":  0,
    "High": 1,
}
INT_TO_SENSITIVITY: Final[dict[int, str]] = {v: k for k, v in SENSITIVITY_TO_INT.items()}

# Severity index used by the EKA formula (CL = tp^alpha * e^n). Section 1.1.
SEVERITY_FROM_HEALTH: Final[dict[str, int]] = {
    "Normal":   1,
    "Moderate": 2,
    "Critical": 3,
}

# ---------------------------------------------------------------------------
# Per-dataset column registry (Section 1.2 dynamic / static lists, 1.7 allow-lists)
# ---------------------------------------------------------------------------

# Raw column names BEFORE any anonymisation. Step 4 introduces `age_anon_mid`
# for the IENN input; that column is added to IENN_FEATURES at Step 4, not here.
RAW_FEATURES: Final[dict[str, list[str]]] = {
    "diabetes": [
        "age",
        "sex",
        "smoking_history",
        "hypertension",
        "heart_disease",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
    ],
    "cvd": [
        "age",
        "sex",
        "height",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
        "weight",
        "ap_hi",
        "ap_lo",
    ],
}

# Static features repeated at every simulated timestep (Section 1.2).
STATIC_FEATURES: Final[dict[str, list[str]]] = {
    "diabetes": ["age", "sex", "smoking_history", "hypertension", "heart_disease"],
    "cvd":      ["age", "sex", "height", "cholesterol", "gluc", "smoke", "alco", "active"],
}

# Dynamic features subject to the backward random walk in STSB (Section 1.2,
# Frozen Decision 7).
DYNAMIC_FEATURES: Final[dict[str, list[str]]] = {
    "diabetes": ["bmi", "HbA1c_level", "blood_glucose_level"],
    "cvd":      ["weight", "ap_hi", "ap_lo"],
}

# Ground-truth disease column per dataset.
DISEASE_COL: Final[dict[str, str]] = {
    "diabetes": "diabetes",
    "cvd":      "cardio",
}

# Raw file names as they appear in data/raw/.
RAW_FILE: Final[dict[str, str]] = {
    "diabetes": "diabetes_iomt_final.csv",
    "cvd":      "cvd_70k_iomt_final.csv",
}

DATASETS: Final[tuple[str, ...]] = ("diabetes", "cvd")

# ---------------------------------------------------------------------------
# Leakage guard (Section 1.7 + Frozen Decision 1 + 15)
# ---------------------------------------------------------------------------

# Columns matching this regex must NEVER appear in a feature frame fed to a
# model. The list mirrors what the spec asks for, with the regex tightened so
# each alternative is anchored/grouped correctly:
#   ^gt_ / ^pred_ / ^eka_  : the family prefixes (must be at column start)
#   ^[kK]_                  : both lowercase `k_*` (EKA / scoring) and
#                             uppercase `K_*` (crypto keys: K_s, K_mac, ...)
#   ^CL$                    : the EKA CL scalar, exactly
#   generalization / record_id / patient_id  : standalone tokens anywhere
IENN_FEATURE_PATTERN_FORBIDDEN: Final[re.Pattern[str]] = re.compile(
    r"(?:^gt_)|(?:^pred_)|(?:^eka_)|(?:^[kK]_)|(?:^CL$)|generalization|record_id|patient_id"
)

LEAKAGE_ASSERT_MSG: Final[str] = (
    "Leakage guard: column matches a forbidden name family "
    "(gt_*, pred_*, eka_*, k_*, CL, generalization, record_id, patient_id) "
    "or is not on the allow-list for this dataset."
)


def allowed_columns(dataset: str) -> list[str]:
    """Return the IENN allow-list for `dataset`.

    Step 0 uses the raw column names. Step 4 will swap `age` -> `age_anon_mid`
    and will be the one to extend this list. Keeping the function here means
    Step 4 only needs to override it, not re-derive it.
    """
    if dataset not in RAW_FEATURES:
        raise KeyError(f"Unknown dataset: {dataset!r}. Known: {DATASETS}")
    return list(RAW_FEATURES[dataset])


def assert_no_leakage(
    columns: list[str] | tuple[str, ...],
    dataset: str,
    *,
    allow_extra: list[str] | tuple[str, ...] = (),
) -> None:
    """Raise ValueError if `columns` contain a forbidden family or anything
    outside the dataset's allow-list (plus `allow_extra`).

    Parameters
    ----------
    columns:
        The column list to validate.
    dataset:
        One of `"diabetes"`, `"cvd"`.
    allow_extra:
        Optional additional columns that are allowed in this context
        (e.g. `record_id` when joining, never as a model feature). Names in
        `allow_extra` are excluded from the forbidden-family check, so a join
        context can carry `record_id` without raising.
    """
    allow_extra_set = set(allow_extra)
    allow = set(allowed_columns(dataset)) | allow_extra_set
    bad_family = [
        c for c in columns
        if c not in allow_extra_set and IENN_FEATURE_PATTERN_FORBIDDEN.search(c)
    ]
    if bad_family:
        raise ValueError(LEAKAGE_ASSERT_MSG + f" Forbidden family: {bad_family}.")
    not_allowed = [c for c in columns if c not in allow]
    if not_allowed:
        raise ValueError(
            LEAKAGE_ASSERT_MSG
            + f" Off-allow-list columns: {not_allowed}."
            + f" Allowed: {sorted(allow)}."
        )


# ---------------------------------------------------------------------------
# Sanity self-checks (run on import; cheap)
# ---------------------------------------------------------------------------

for _ds in DATASETS:
    _dyn = set(DYNAMIC_FEATURES[_ds])
    _stat = set(STATIC_FEATURES[_ds])
    _raw = set(RAW_FEATURES[_ds])
    assert _dyn.isdisjoint(_stat), f"static/dynamic overlap in {_ds}"
    assert _dyn | _stat == _raw, f"STATIC+DYNAMIC must equal RAW_FEATURES in {_ds}"
    assert _ds in RAW_FILE and _ds in DISEASE_COL
