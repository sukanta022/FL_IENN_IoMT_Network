"""tests/test_step0.py

Step 0 acceptance tests (run after `python -m fl_ienn.common.prepare_data`).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fl_ienn.common.schema import (
    DATASETS,
    HEALTH_TO_INT,
    RAW_FEATURES,
    SENSITIVITY_TO_INT,
    SEVERITY_FROM_HEALTH,
    IENN_FEATURE_PATTERN_FORBIDDEN,
    assert_no_leakage,
    allowed_columns,
)
from fl_ienn.common.data_io import (
    load_raw_features,
    load_ids,
    load_eka_prior,
    load_labels_gt,
    load_disease_gt,
    load_splits,
    processed_dir,
)

ROOT = Path(".").resolve()


# ---------------------------------------------------------------------------
# Schema unit tests
# ---------------------------------------------------------------------------

def test_forbidden_pattern_matches_expected_families() -> None:
    assert IENN_FEATURE_PATTERN_FORBIDDEN.search("gt_health")
    assert IENN_FEATURE_PATTERN_FORBIDDEN.search("pred_p_high")
    assert IENN_FEATURE_PATTERN_FORBIDDEN.search("eka_prior_tier")
    assert IENN_FEATURE_PATTERN_FORBIDDEN.search("K_s")
    assert IENN_FEATURE_PATTERN_FORBIDDEN.search("K_mac")
    assert IENN_FEATURE_PATTERN_FORBIDDEN.search("CL")
    assert IENN_FEATURE_PATTERN_FORBIDDEN.search("generalization_level")
    assert IENN_FEATURE_PATTERN_FORBIDDEN.search("record_id")
    assert IENN_FEATURE_PATTERN_FORBIDDEN.search("patient_id")
    # Negatives
    assert not IENN_FEATURE_PATTERN_FORBIDDEN.search("age")
    assert not IENN_FEATURE_PATTERN_FORBIDDEN.search("bmi")
    assert not IENN_FEATURE_PATTERN_FORBIDDEN.search("ap_hi")


def test_assert_no_leakage_rejects_bad_columns() -> None:
    with pytest.raises(ValueError):
        assert_no_leakage(["age", "gt_health"], "diabetes")
    with pytest.raises(ValueError):
        assert_no_leakage(["age", "eka_prior_tier"], "diabetes")
    with pytest.raises(ValueError):
        assert_no_leakage(["age", "K_mac"], "diabetes")
    with pytest.raises(ValueError):
        # Not on the allow-list
        assert_no_leakage(["age", "favorite_color"], "diabetes")


def test_assert_no_leakage_accepts_valid_columns() -> None:
    # Diabetes raw columns
    assert_no_leakage(RAW_FEATURES["diabetes"], "diabetes")
    # CVD raw columns
    assert_no_leakage(RAW_FEATURES["cvd"], "cvd")
    # record_id is allowed when added to allow_extra (join context)
    assert_no_leakage(["record_id", *RAW_FEATURES["diabetes"]], "diabetes",
                      allow_extra=["record_id"])


# ---------------------------------------------------------------------------
# Loader / dataset tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def root() -> Path:
    return ROOT


@pytest.mark.parametrize("ds", list(DATASETS))
def test_no_unknown_left(root: Path, ds: str) -> None:
    labels = load_labels_gt(root, ds)
    assert "Unknown" not in set(labels["gt_health"].unique()), f"{ds}: Unknown still present"


@pytest.mark.parametrize("ds", list(DATASETS))
def test_label_encodings(root: Path, ds: str) -> None:
    labels = load_labels_gt(root, ds)
    # All three classes present (Unknown already dropped).
    if ds == "cvd":
        assert set(labels["gt_health"].unique()) == {"Normal", "Moderate", "Critical"}
    else:
        assert set(labels["gt_health"].unique()) == {"Normal", "Moderate", "Critical"}
    # Mapping check: gt_sensitivity == High iff gt_health in {Moderate, Critical}.
    rule = labels["gt_sensitivity"].eq("High") == labels["gt_health"].isin(["Moderate", "Critical"])
    assert rule.all(), f"{ds}: sensitivity mapping rule violated"


@pytest.mark.parametrize("ds", list(DATASETS))
def test_features_have_no_forbidden_columns(root: Path, ds: str) -> None:
    feats = load_raw_features(root, ds)
    cols = list(feats.columns)
    assert "record_id" in cols
    # No gt_*, eka_*, pred_*, k_*, CL, generalization, record_id-as-feature,
    # patient_id-as-feature.
    bad = [c for c in cols if c != "record_id" and IENN_FEATURE_PATTERN_FORBIDDEN.search(c)]
    assert not bad, f"{ds}: forbidden columns in features: {bad}"
    # And no patient_id (it lives in ids.parquet).
    assert "patient_id" not in cols, f"{ds}: patient_id leaked into features.parquet"
    # Allow-list match
    expected = ["record_id", *RAW_FEATURES[ds]]
    assert sorted(cols) == sorted(expected)


@pytest.mark.parametrize("ds", list(DATASETS))
def test_eka_prior_mapping(root: Path, ds: str) -> None:
    eka = load_eka_prior(root, ds)
    labels = load_labels_gt(root, ds)
    merged = eka.merge(labels, on="record_id")
    # eka_prior_tier == gt_sensitivity
    assert (merged["eka_prior_tier"] == merged["gt_sensitivity"]).all()
    # severity is the map of gt_health
    expected_sev = merged["gt_health"].map(SEVERITY_FROM_HEALTH).astype(int)
    assert (merged["eka_prior_severity"] == expected_sev).all()


@pytest.mark.parametrize("ds", list(DATASETS))
def test_splits_disjoint_and_complete(root: Path, ds: str) -> None:
    splits = load_splits(root, ds)
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    union = train | val | test
    # All rows in the processed set are accounted for
    ids = load_ids(root, ds)
    expected = set(ids["record_id"].tolist())
    assert union == expected, f"{ds}: split union != all record_ids"


@pytest.mark.parametrize("ds", list(DATASETS))
def test_splits_keep_class_proportions(root: Path, ds: str) -> None:
    splits = load_splits(root, ds)
    labels = load_labels_gt(root, ds)
    rid_to_class = dict(zip(labels["record_id"], labels["gt_health"]))
    overall = pd.Series(rid_to_class).value_counts(normalize=True).sort_index()
    for split in ("train", "val", "test"):
        cls = pd.Series([rid_to_class[r] for r in splits[split]]).value_counts(normalize=True).sort_index()
        for c, p in overall.items():
            assert abs(cls.get(c, 0.0) - p) < 1e-3, (
                f"{ds}: split {split} class {c} drift {cls.get(c, 0.0):.4f} vs {p:.4f}"
            )


def test_same_seed_same_splits() -> None:
    # Read the splits file twice and confirm the recorded IDs are stable.
    for ds in DATASETS:
        a = load_splits(ROOT, ds)
        b = load_splits(ROOT, ds)
        for k in ("train", "val", "test"):
            assert a[k] == b[k], f"{ds} {k} split changed between reads"


def test_metrics_json_present() -> None:
    p = ROOT / "reports" / "00_data" / "metrics.json"
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "datasets" in data and "diabetes" in data["datasets"] and "cvd" in data["datasets"]
    # Sensitivity mapping must be OK in both
    for ds in ("diabetes", "cvd"):
        ok = data["datasets"][ds]["sensitivity_mapping_check"]["ok"]
        assert ok is True, f"{ds}: sensitivity mapping rule failed"


def test_table_summary_present() -> None:
    p = ROOT / "reports" / "00_data" / "table1_dataset_summary.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert set(df["dataset"]) == {"diabetes", "cvd"}
    # CVD excluded 1334 Unknown per spec
    cvd = df[df["dataset"] == "cvd"].iloc[0]
    assert int(cvd["excluded_unknown"]) == 1334
    # Diabetes dropped gender
    dia = df[df["dataset"] == "diabetes"].iloc[0]
    assert str(dia["gender_dropped"]).lower() in {"true", "1"}
