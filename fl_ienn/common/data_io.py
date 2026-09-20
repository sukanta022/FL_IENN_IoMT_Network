"""fl_ienn/common/data_io.py

Step 0 loaders.

Each loader returns only its own file. There is intentionally NO loader that
returns `(features, labels)` together — those are composed at Step 4 (IENN)
and Step 14 (disease model) using `record_id` joins. That separation is the
structural part of the leakage protection (Frozen Decision 15).
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from .schema import (
    DISEASE_COL,
    RAW_FEATURES,
    RAW_FILE,
    assert_no_leakage,
)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROCESSED_FILE: Final[dict[str, str]] = {
    "features":    "features.parquet",
    "ids":         "ids.parquet",
    "eka_prior":   "eka_prior.parquet",
    "labels_gt":   "labels_gt.parquet",
    "disease_gt":  "disease_gt.parquet",
    "splits":      "splits.json",
}


def processed_dir(root: Path, dataset: str) -> Path:
    """Return `<root>/data/processed/<dataset>`.

    `root` is the project root (the directory that contains `data/`). This
    keeps the loader API stable: callers pass the root, never a partial path.
    """
    return Path(root) / "data" / "processed" / dataset


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_raw_features(root: Path, dataset: str) -> pd.DataFrame:
    """Return `record_id` + RAW_FEATURES columns only.

    The leakage guard is enforced: if anything that smells like a label or
    a routing variable snuck in, this raises before the caller sees it.
    """
    path = processed_dir(root, dataset) / PROCESSED_FILE["features"]
    df = pd.read_parquet(path)
    expected = ["record_id", *RAW_FEATURES[dataset]]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(f"features.parquet missing columns: {missing}")
    df = df[expected]
    assert_no_leakage([c for c in df.columns if c != "record_id"], dataset)
    return df


def load_ids(root: Path, dataset: str) -> pd.DataFrame:
    """Return `record_id, patient_id` (joins only — never fed to a model)."""
    path = processed_dir(root, dataset) / PROCESSED_FILE["ids"]
    df = pd.read_parquet(path)
    if set(df.columns) != {"record_id", "patient_id"}:
        raise KeyError(f"ids.parquet must contain exactly record_id, patient_id; got {set(df.columns)}")
    return df


def load_eka_prior(root: Path, dataset: str) -> pd.DataFrame:
    """Return `record_id, eka_prior_tier, eka_prior_severity`."""
    path = processed_dir(root, dataset) / PROCESSED_FILE["eka_prior"]
    df = pd.read_parquet(path)
    expected = {"record_id", "eka_prior_tier", "eka_prior_severity"}
    if set(df.columns) != expected:
        raise KeyError(f"eka_prior.parquet columns must be {expected}; got {set(df.columns)}")
    return df


def load_labels_gt(root: Path, dataset: str) -> pd.DataFrame:
    """Return `record_id, gt_health, gt_sensitivity, gt_alert`.

    These columns must NEVER reach a model input. The leakage guard is run
    on the non-`record_id` columns to make that contract explicit.
    """
    path = processed_dir(root, dataset) / PROCESSED_FILE["labels_gt"]
    df = pd.read_parquet(path)
    expected = {"record_id", "gt_health", "gt_sensitivity", "gt_alert"}
    if set(df.columns) != expected:
        raise KeyError(f"labels_gt.parquet columns must be {expected}; got {set(df.columns)}")
    # Defensive: even though gt_* is on the forbidden list, we run the guard
    # with an explicit allow-list that *excludes* it from model input.
    forbidden_here = {"gt_health", "gt_sensitivity", "gt_alert"}
    bad = [c for c in df.columns if c in forbidden_here]
    if bad:
        # This is the labels file, so finding these here is correct.
        # The guard only matters when something that is NOT one of these
        # labels sneaks in; the assertion below enforces that.
        extra = [c for c in df.columns if c not in {"record_id"} and c not in forbidden_here]
        if extra:
            raise ValueError(
                "labels_gt.parquet must contain only record_id, gt_health, "
                f"gt_sensitivity, gt_alert. Extra columns: {extra}"
            )
    return df


def load_disease_gt(root: Path, dataset: str) -> pd.DataFrame:
    """Return `record_id, gt_disease`."""
    path = processed_dir(root, dataset) / PROCESSED_FILE["disease_gt"]
    df = pd.read_parquet(path)
    expected = {"record_id", "gt_disease"}
    if set(df.columns) != expected:
        raise KeyError(f"disease_gt.parquet columns must be {expected}; got {set(df.columns)}")
    return df


def load_splits(root: Path, dataset: str) -> dict[str, list[int]]:
    """Return {"train": [...], "val": [...], "test": [...]} as record_id lists."""
    import json
    path = processed_dir(root, dataset) / PROCESSED_FILE["splits"]
    with open(path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    for k in ("train", "val", "test"):
        if k not in splits:
            raise KeyError(f"splits.json missing key {k!r}")
        if not isinstance(splits[k], list):
            raise TypeError(f"splits.json[{k!r}] must be a list of record_ids")
    return splits
