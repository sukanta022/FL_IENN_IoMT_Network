"""fl_ienn/common/prepare_data.py

CLI: `python -m fl_ienn.common.prepare_data`

Reads the two raw CSVs from config.yaml, normalises column names and labels,
applies the diabetes gender/sex check, drops CVD `Unknown` health_status,
builds the per-record eka_prior, verifies the sensitivity/health mapping,
writes leak-proof parquet files (one concept per file, joined by record_id),
and emits a 70/10/20 stratified split on gt_health.

Outputs:
    data/processed/<dataset>/{features,ids,eka_prior,labels_gt,disease_gt}.parquet
    data/processed/<dataset>/splits.json
    reports/00_data/table1_dataset_summary.csv
    reports/00_data/metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from .data_io import PROCESSED_FILE, processed_dir
from .schema import (
    DATASETS,
    DISEASE_COL,
    HEALTH_TO_INT,
    RAW_FEATURES,
    RAW_FILE,
    SEVERITY_FROM_HEALTH,
    SENSITIVITY_TO_INT,
    allowed_columns,
    assert_no_leakage,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "seed" not in cfg or "splits" not in cfg or "paths" not in cfg:
        raise KeyError("config.yaml must define at least seed, paths, splits")
    s = cfg["splits"]
    if abs(s["train"] + s["val"] + s["test"] - 1.0) > 1e-9:
        raise ValueError(f"splits must sum to 1.0; got {s}")
    return cfg


# ---------------------------------------------------------------------------
# Per-dataset normalisation
# ---------------------------------------------------------------------------

def _normalize_cvd(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """CVD rename + Unknown drop. Returns (clean_df, exclusion_counts)."""
    df = df.rename(columns={"age_years": "age"})
    n_in = len(df)
    counts_before = df["health_status"].value_counts().to_dict()
    df = df[df["health_status"] != "Unknown"].copy()
    n_out = len(df)
    return df, {
        "rows_in":  n_in,
        "rows_out": n_out,
        "excluded_unknown_health": n_in - n_out,
        "health_status_before_drop": {str(k): int(v) for k, v in counts_before.items()},
    }


def _normalize_diabetes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Diabetes: verify gender/sex is one-to-one; drop `gender` if so."""
    info: dict[str, Any] = {
        "rows_in":          int(len(df)),
        "rows_out":         int(len(df)),
        "excluded_unknown_health": 0,
        "gender_dropped":   False,
        "gender_sex_xtab":  {},
    }
    if "gender" in df.columns and "sex" in df.columns:
        xtab = pd.crosstab(df["gender"], df["sex"])
        info["gender_sex_xtab"] = {
            str(g): {str(s): int(c) for s, c in row.items()}
            for g, row in xtab.iterrows()
        }
        # one-to-one iff every (gender, sex) pair with count > 0 is unique
        # across both axes (i.e. no gender maps to two sexes and vice versa).
        nz = xtab.values > 0
        n_nonzero_per_gender = nz.sum(axis=1)
        n_nonzero_per_sex    = nz.sum(axis=0)
        one_to_one = bool(
            (n_nonzero_per_gender <= 1).all()
            and (n_nonzero_per_sex    <= 1).all()
        )
        if one_to_one:
            df = df.drop(columns=["gender"])
            info["gender_dropped"] = True
        else:
            raise RuntimeError(
                "Step 0 STOP: diabetes `gender` vs `sex` is NOT one-to-one; "
                "manual decision required before dropping. "
                f"Crosstab: {info['gender_sex_xtab']}"
            )
    return df, info


def _ensure_record_id(df: pd.DataFrame) -> pd.DataFrame:
    """Stable, unique record_id. If `record_id` is unique, keep it. Otherwise
    build a deterministic id from `patient_id` + row index."""
    if "record_id" in df.columns and df["record_id"].is_unique:
        return df
    if "patient_id" in df.columns and df["patient_id"].is_unique:
        return df.assign(record_id=df["patient_id"])
    df = df.copy()
    df["record_id"] = np.arange(len(df), dtype=np.int64)
    return df


def _build_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build labels_gt, eka_prior, disease_gt from the normalized frame."""
    info: dict[str, Any] = {}
    # gt_health, gt_sensitivity, gt_alert come straight from the dataset.
    for col in ("gt_health", "gt_sensitivity", "gt_alert"):
        if col not in df.columns:
            raise KeyError(f"missing label column {col!r} after rename")
    labels_gt = df[["record_id", "gt_health", "gt_sensitivity", "gt_alert"]].copy()

    # Verify the spec rule: gt_sensitivity == "High" iff gt_health in {Moderate, Critical}.
    rule_high  = labels_gt["gt_sensitivity"].eq("High")
    rule_health_in = labels_gt["gt_health"].isin(["Moderate", "Critical"])
    mismatch = int((rule_high != rule_health_in).sum())
    info["sensitivity_mapping_check"] = {
        "rule":  "gt_sensitivity == 'High'  iff  gt_health in {Moderate, Critical}",
        "mismatches": mismatch,
        "ok":    mismatch == 0,
    }
    if mismatch != 0:
        raise RuntimeError(
            "Step 0 STOP: ground-truth sensitivity mapping does NOT match the spec "
            f"({mismatch} mismatches). See metrics.json."
        )

    # eka_prior: derived from labels (simulator role — see Section 1.1).
    eka_prior = pd.DataFrame({
        "record_id":            labels_gt["record_id"],
        "eka_prior_tier":       labels_gt["gt_sensitivity"],   # {"Low","High"}
        "eka_prior_severity":   labels_gt["gt_health"].map(SEVERITY_FROM_HEALTH).astype(int),
    })

    # disease_gt
    disease_gt = df[["record_id", "gt_disease"]].copy()

    return labels_gt, eka_prior, disease_gt, info


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def _stratified_split(
    record_ids: np.ndarray,
    y_health:   np.ndarray,
    ratios: dict[str, float],
    seed:       int,
) -> dict[str, np.ndarray]:
    """70/10/20 stratified on y_health. Two successive sklearn splits."""
    r_train, r_val = ratios["train"], ratios["val"]
    # train vs (val+test)
    train_ids, vt_ids, _, y_vt = train_test_split(
        record_ids, y_health,
        test_size=(1.0 - r_train),
        random_state=seed,
        stratify=y_health,
    )
    # val vs test within vt
    val_share = r_val / (r_val + ratios["test"])
    val_ids, test_ids = train_test_split(
        vt_ids,
        test_size=(1.0 - val_share),
        random_state=seed,
        stratify=y_vt,
    )
    return {"train": train_ids, "val": val_ids, "test": test_ids}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare_one(
    dataset: str,
    raw_root: Path,
    project_root: Path,
    seed:     int,
    splits:   dict[str, float],
) -> dict[str, Any]:
    raw_path = raw_root / RAW_FILE[dataset]
    df = pd.read_csv(raw_path)

    if dataset == "cvd":
        df, info = _normalize_cvd(df)
    elif dataset == "diabetes":
        df, info = _normalize_diabetes(df)
    else:
        raise KeyError(f"Unknown dataset {dataset!r}")

    df = _ensure_record_id(df)

    # Rename labels to the frozen gt_*/eka_prior_* naming.
    df = df.rename(columns={
        "health_status":    "gt_health",
        "data_sensitivity": "gt_sensitivity",
        "alert_flag":       "gt_alert",
        DISEASE_COL[dataset]: "gt_disease",
    })

    labels_gt, eka_prior, disease_gt, label_info = _build_labels(df)
    info.update(label_info)

    # Features frame: record_id + RAW_FEATURES only.
    feature_cols = ["record_id", *RAW_FEATURES[dataset]]
    features = df[feature_cols].copy()
    # Run the leakage guard before writing.
    assert_no_leakage(RAW_FEATURES[dataset], dataset)

    ids = df[["record_id", "patient_id"]].copy()

    # Splits.
    rec_ids = df["record_id"].to_numpy()
    y_health_int = df["gt_health"].map(HEALTH_TO_INT).to_numpy()
    split_ids = _stratified_split(rec_ids, y_health_int, splits, seed)
    splits_dict = {k: [int(x) for x in v] for k, v in split_ids.items()}

    # Write outputs under `<project_root>/data/processed/<dataset>`.
    out = processed_dir(project_root, dataset)
    out.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out / PROCESSED_FILE["features"], index=False)
    ids.to_parquet(out / PROCESSED_FILE["ids"], index=False)
    eka_prior.to_parquet(out / PROCESSED_FILE["eka_prior"], index=False)
    labels_gt.to_parquet(out / PROCESSED_FILE["labels_gt"], index=False)
    disease_gt.to_parquet(out / PROCESSED_FILE["disease_gt"], index=False)
    with open(out / PROCESSED_FILE["splits"], "w", encoding="utf-8") as f:
        json.dump(splits_dict, f)

    info.update({
        "dataset": dataset,
        "n_rows":  int(len(df)),
        "split_sizes": {k: len(v) for k, v in splits_dict.items()},
        "split_class_distribution": {
            split: labels_gt.iloc[
                labels_gt["record_id"].isin(ids_).to_numpy()
            ]["gt_health"].value_counts().sort_index().to_dict()
            for split, ids_ in splits_dict.items()
        },
        "features_columns": RAW_FEATURES[dataset],
        "allowed_columns": allowed_columns(dataset),
    })
    return info


def _write_reports(all_info: dict[str, dict[str, Any]], reports_root: Path, seed: int) -> None:
    reports_dir = reports_root / "00_data"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "datasets": all_info}, f, indent=2)

    # table1_dataset_summary.csv
    rows: list[dict[str, Any]] = []
    for ds, info in all_info.items():
        row: dict[str, Any] = {
            "dataset":            ds,
            "rows_in":            info["rows_in"],
            "rows_out":           info["rows_out"],
            "excluded_unknown":   info.get("excluded_unknown_health", 0),
            "gender_dropped":     info.get("gender_dropped", "n/a"),
            "n_features":         len(info["features_columns"]),
            "n_rows_final":       info["n_rows"],
            "n_train":            info["split_sizes"]["train"],
            "n_val":              info["split_sizes"]["val"],
            "n_test":             info["split_sizes"]["test"],
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(reports_dir / "table1_dataset_summary.csv", index=False)

    # Per-split class distribution (long format) for inspection.
    long_rows: list[dict[str, Any]] = []
    for ds, info in all_info.items():
        for split, counts in info["split_class_distribution"].items():
            for cls, n in counts.items():
                long_rows.append({"dataset": ds, "split": split, "gt_health": cls, "n": int(n)})
    pd.DataFrame(long_rows).to_csv(reports_dir / "table_split_class_distribution.csv", index=False)


def main() -> None:
    cfg = load_config()
    project_root = Path(".").resolve()
    raw_dir  = Path(cfg["paths"]["raw_dir"])
    rep_dir  = Path(cfg["paths"]["reports_dir"])
    seed     = int(cfg["seed"])
    splits   = cfg["splits"]

    all_info: dict[str, dict[str, Any]] = {}
    for ds in DATASETS:
        all_info[ds] = prepare_one(ds, raw_dir, project_root, seed, splits)

    _write_reports(all_info, rep_dir, seed)
    # Concise console summary so the user can see it ran.
    for ds, info in all_info.items():
        print(
            f"[Step 0] {ds}: in={info['rows_in']} out={info['rows_out']} "
            f"train={info['split_sizes']['train']} val={info['split_sizes']['val']} "
            f"test={info['split_sizes']['test']} sens_rule_ok={info['sensitivity_mapping_check']['ok']}"
        )
    print("[Step 0] Wrote reports/00_data/{metrics.json, table1_dataset_summary.csv, table_split_class_distribution.csv}")


if __name__ == "__main__":
    main()
