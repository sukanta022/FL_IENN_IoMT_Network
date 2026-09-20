"""tests/test_step1.py

Step 1 acceptance tests (run after `python -m fl_ienn.session --build`).
"""
from __future__ import annotations

import json
import re
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fl_ienn.common.data_io import (
    load_ids,
    load_raw_features,
    load_splits,
)
from fl_ienn.common.schema import DATASETS, RAW_FEATURES
from fl_ienn.session.device_sim import (
    Device,
    DeviceRegistry,
    StreamEnvelope,
    assign_device,
    build_registry,
    iter_stream,
    iter_all_datasets,
    REGISTRY_PATH,
)

ROOT = Path(".").resolve()
REPORTS = ROOT / "reports" / "01_session"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def registry(cfg) -> DeviceRegistry:
    return build_registry(cfg)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_device_count(cfg, registry) -> None:
    per = int(cfg["devices"]["per_dataset"])
    assert len(registry.list("diabetes")) == per
    assert len(registry.list("cvd"))      == per
    assert len(registry.devices)         == 2 * per


def test_registry_unique_ids(cfg, registry) -> None:
    ids = [d.device_id for d in registry.devices]
    assert len(set(ids)) == len(ids)
    # Naming convention matches the brief.
    glu = {d.device_id for d in registry.list("diabetes")}
    bpm = {d.device_id for d in registry.list("cvd")}
    assert glu == {f"GLU-{i:03d}" for i in range(1, int(cfg["devices"]["per_dataset"]) + 1)}
    assert bpm == {f"BPM-{i:03d}" for i in range(1, int(cfg["devices"]["per_dataset"]) + 1)}


def test_registry_credential_format(cfg, registry) -> None:
    pwd_len = int(cfg["devices"]["password_len"])
    hex_len = int(cfg["devices"]["secret_k_hex_len"])
    for d in registry.devices:
        assert len(d.password) == pwd_len
        assert len(d.secret_k) == hex_len
        assert re.fullmatch(r"[0-9a-f]+", d.secret_k)


def test_registry_credentials_order(registry) -> None:
    d = registry.devices[0]
    a, b, c = d.credentials()
    assert a == d.device_id
    assert b == d.password
    assert c == d.secret_k


def test_registry_determinism(cfg) -> None:
    r1 = build_registry(cfg)
    r2 = build_registry(cfg)
    same = all(d1.password == d2.password and d1.secret_k == d2.secret_k
               for d1, d2 in zip(r1.devices, r2.devices))
    assert same, "build_registry must be deterministic for a fixed seed"


def test_registry_different_seed_changes_credentials(cfg) -> None:
    r1 = build_registry(cfg)
    cfg2 = dict(cfg); cfg2["seed"] = int(cfg["seed"]) + 1
    r2 = build_registry(cfg2)
    assert any(d1.secret_k != d2.secret_k
               for d1, d2 in zip(r1.devices, r2.devices))


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

def test_assign_device_deterministic(cfg, registry) -> None:
    per = int(cfg["devices"]["per_dataset"])
    a = assign_device(12345, "diabetes", registry, per)
    b = assign_device(12345, "diabetes", registry, per)
    assert a == b
    assert re.fullmatch(r"GLU-\d{3}", a)


def test_assign_device_dataset_scoped(cfg, registry) -> None:
    per = int(cfg["devices"]["per_dataset"])
    pid = 99999
    d_diab = assign_device(pid, "diabetes", registry, per)
    d_cvd  = assign_device(pid, "cvd",      registry, per)
    assert d_diab.startswith("GLU-")
    assert d_cvd.startswith("BPM-")
    assert d_diab != d_cvd


def test_assignment_covers_all_records(cfg, registry) -> None:
    """Every record is assigned exactly once and to a device of its own dataset."""
    per = int(cfg["devices"]["per_dataset"])
    for ds in DATASETS:
        ids = load_ids(ROOT, ds)
        devices = set(d.device_id for d in registry.list(ds))
        counts: dict[str, int] = {}
        for _, row in ids.iterrows():
            dev = assign_device(int(row["patient_id"]), ds, registry, per)
            assert dev in devices, f"{dev} not in registry[{ds}]"
            counts[dev] = counts.get(dev, 0) + 1
        # Every device gets at least one record (modulo very large skew,
        # which 10 devices on tens of thousands of records will not produce).
        missing = devices - counts.keys()
        assert not missing, f"devices with zero records in {ds}: {missing}"


# ---------------------------------------------------------------------------
# Stream envelope shape and ordering
# ---------------------------------------------------------------------------

def _envelope_field_names() -> set[str]:
    return set(StreamEnvelope.__dataclass_fields__.keys())


def test_envelope_fields_are_exactly_allowed() -> None:
    allowed = {"device_id", "dataset", "seq_no", "ts", "record_id",
               "features", "eka_prior_tier", "eka_prior_severity"}
    assert _envelope_field_names() == allowed


def test_envelope_is_frozen() -> None:
    env = StreamEnvelope(
        device_id="GLU-001", dataset="diabetes", seq_no=1, ts=0.0,
        record_id=1, features={"a": 1},
        eka_prior_tier="Low", eka_prior_severity=1,
    )
    with pytest.raises(FrozenInstanceError):
        env.device_id = "GLU-002"  # type: ignore[misc]


def test_stream_seq_no_per_device(cfg, registry) -> None:
    per_device_seqs: dict[str, list[int]] = {}
    for ds in DATASETS:
        for env in iter_stream(ds, split="train", cfg=cfg, registry=registry):
            per_device_seqs.setdefault(env.device_id, []).append(env.seq_no)
    for dev_id, seqs in per_device_seqs.items():
        assert seqs[0] == 1, f"{dev_id} must start at 1, got {seqs[0]}"
        assert seqs == sorted(seqs)
        assert set(seqs) == set(range(1, len(seqs) + 1))


def test_stream_ts_non_decreasing(cfg, registry) -> None:
    """ts must be non-decreasing globally across both datasets."""
    last_ts = -np.inf
    for env in iter_all_datasets(split="train", cfg=cfg, registry=registry):
        assert env.ts >= last_ts, f"ts regressed: {env.ts} < {last_ts}"
        last_ts = env.ts


def test_stream_features_match_raw_columns(cfg, registry) -> None:
    for ds in DATASETS:
        first = next(iter_stream(ds, split="train", cfg=cfg, registry=registry))
        assert set(first.features.keys()) == set(RAW_FEATURES[ds])


def test_stream_no_label_or_disease_loaders_used() -> None:
    """Static guard: device_sim.py must not reference label/disease loaders.

    We import the source fresh from disk so monkey-patching cannot hide a
    deleted import from the test. This is the strongest protection we can
    apply without fudging the regex out.
    """
    src = (ROOT / "fl_ienn/session/device_sim.py").read_text(encoding="utf-8")
    forbidden = ["labels_gt", "disease_gt", "load_labels_gt", "load_disease_gt"]
    # Strip leading '#' comments to allow documentation references.
    non_comment = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    for tok in forbidden:
        assert tok not in non_comment, (
            f"device_sim.py references {tok!r}; Step 1 must not touch labels"
        )


def test_stream_record_ids_match_features(cfg, registry) -> None:
    """Cross-check: every record in a split must appear exactly once in the stream."""
    for ds in DATASETS:
        feats = load_raw_features(ROOT, ds)
        splits = load_splits(ROOT, ds)
        expected = set(int(r) for r in splits["train"])
        seen: set[int] = set()
        for env in iter_stream(ds, split="train", cfg=cfg, registry=registry):
            rid = int(env.record_id)
            assert rid not in seen, f"duplicate record_id {rid} in stream"
            seen.add(rid)
            assert rid in feats["record_id"].to_numpy()
        assert seen == expected


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _find_report(stem: str, suffix: str) -> Path:
    """Return the canonical report file or a timestamped sibling.

    The writer falls back to a `stem.<hex><suffix>` path when an external
    process (e.g. a VS Code preview) holds the canonical file open. The
    test must accept either variant.
    """
    canonical = REPORTS / f"{stem}{suffix}"
    if canonical.exists():
        return canonical
    siblings = sorted(REPORTS.glob(f"{stem}.*{suffix}"))
    if siblings:
        return siblings[-1]
    raise AssertionError(f"missing {canonical}")


def test_reports_exist_and_no_secrets(cfg) -> None:
    table_path = _find_report("table_device_registry", ".csv")
    metrics_path = _find_report("metrics", ".json")
    assert table_path.exists(), f"missing {table_path}"
    assert metrics_path.exists(), f"missing {metrics_path}"

    table = pd.read_csv(table_path)
    # Forbidden in reports: secrets and any obvious credential columns.
    forbidden = {"password", "secret_k", "credentials"}
    assert not (forbidden & set(table.columns))
    # Sanity: the table lists one row per device.
    assert len(table) == 2 * int(cfg["devices"]["per_dataset"])

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "devices_per_dataset" in metrics
    assert "per_dataset_assignment" in metrics
    assert "records_per_device_stats" in metrics
    # Coverage guard.
    assert all(
        not v.get("any_record_twice")
        for k, v in metrics["per_dataset_assignment"].items()
        if k in DATASETS
    )
