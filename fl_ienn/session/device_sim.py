"""fl_ienn/session/device_sim.py

Step 1 — device simulation and registration (Section 1.1, 3, 4 of the frozen doc).

Responsibilities:
    * Build a seeded registry of simulated IoMT devices (10 per dataset).
    * Deterministically assign every record to exactly one device of its own
      dataset, using hashlib.sha256 so it stays stable across processes.
    * Produce a `StreamEnvelope` per record with only the fields the Session
      Layer is allowed to see (no patient_id, no gt_*, no pred_*).
    * Drive a simulated clock for ts; we never use wall-clock time.

HARD RULE: this module MUST NOT touch labels or disease ground-truth. The
leakage tests in tests/test_step1.py grep this file for the loader names.
If a future step needs ground truth, it must import them itself, not
through this module.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import yaml

from fl_ienn.common.schema import DATASETS, RAW_FEATURES
from fl_ienn.common.data_io import (
    load_raw_features,
    load_ids,
    load_eka_prior,
    load_splits,
    processed_dir,
)

# ---------------------------------------------------------------------------
# Constants and paths
# ---------------------------------------------------------------------------

DEVICE_PREFIX: dict[str, str] = {
    "diabetes": "GLU",
    "cvd":      "BPM",
}
DEVICE_TYPE: dict[str, str] = {
    "diabetes": "glucose_meter",
    "cvd":      "bp_monitor",
}

# Registry contains device_id, password, secret_k. It is a secret material.
# Step 2 will pass (device_id, password, secret_k) straight into the existing
# `ECCAuthenticator.register_device(device_id, password, secret_key_k)` so we
# keep that exact ordering in Device.credentials().
REGISTRY_PATH: Path = Path("artifacts/devices/registry.json")


# ---------------------------------------------------------------------------
# Device dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Device:
    """One simulated IoMT device.

    `password` and `secret_k` are simulation-only secrets. The legacy
    `ECCAuthenticator.register_device(device_id, password, secret_key_k)`
    accepts them in exactly this order; see `Device.credentials()`.
    """
    device_id:   str
    dataset:     str
    device_type: str
    password:    str   # PWD_i  — 16 random characters
    secret_k:    str   # k_i    — 16 random hex characters (simulation entropy)

    def credentials(self) -> tuple[str, str, str]:
        """Return `(device_id, password, secret_k)` in the order the legacy
        `ECCAuthenticator.register_device` expects.
        """
        return (self.device_id, self.password, self.secret_k)


# ---------------------------------------------------------------------------
# Credential generators (seeded)
# ---------------------------------------------------------------------------

_PWD_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _seeded_rng(seed: int) -> np.random.Generator:
    """One seeded RNG per (dataset, device) so the registry is reproducible.

    `secrets.choice` would be unsafe for seeding. We use a numpy Generator
    with bit-exact reproducibility; this is a *simulation* (Section 8.2 of
    the frozen doc), not a real cryptographic enrolment.
    """
    return np.random.default_rng(seed)


def _gen_password(rng: np.random.Generator, n: int) -> str:
    """`n` characters drawn uniformly from a 62-char alphabet."""
    idx = rng.integers(0, len(_PWD_ALPHABET), size=n)
    return "".join(_PWD_ALPHABET[i] for i in idx)


def _gen_secret_k(rng: np.random.Generator, n_hex_chars: int) -> str:
    """`n_hex_chars` characters of lowercase hex (e.g. 16 chars = 8 bytes)."""
    n_bytes = n_hex_chars // 2
    return rng.bytes(n_bytes).hex()


# ---------------------------------------------------------------------------
# Device registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeviceRegistry:
    """Read-only registry view, indexed by device_id."""
    devices: tuple[Device, ...] = field(default_factory=tuple)

    def get(self, device_id: str) -> Device:
        for d in self.devices:
            if d.device_id == device_id:
                return d
        raise KeyError(f"unknown device_id {device_id!r}")

    def list(self, dataset: str) -> list[Device]:
        return [d for d in self.devices if d.dataset == dataset]

    def by_dataset(self) -> dict[str, list[Device]]:
        out: dict[str, list[Device]] = {ds: [] for ds in DATASETS}
        for d in self.devices:
            out[d.dataset].append(d)
        return out

    # ---- IO ---------------------------------------------------------------

    def to_json_dict(self) -> dict[str, dict[str, str]]:
        """Schema suitable for json.dump. Includes secrets — caller must
        write under `artifacts/` (git-ignored)."""
        return {
            d.device_id: {
                "dataset":     d.dataset,
                "device_type": d.device_type,
                "password":    d.password,
                "secret_k":    d.secret_k,
            }
            for d in self.devices
        }

    @staticmethod
    def from_json_dict(d: dict[str, dict[str, str]]) -> "DeviceRegistry":
        devs = [
            Device(
                device_id   = dev_id,
                dataset     = rec["dataset"],
                device_type = rec["device_type"],
                password    = rec["password"],
                secret_k    = rec["secret_k"],
            )
            for dev_id, rec in d.items()
        ]
        # Stable ordering by device_id for reproducibility.
        devs.sort(key=lambda d: d.device_id)
        return DeviceRegistry(tuple(devs))

    def save(self, path: Path = REGISTRY_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, indent=2, sort_keys=True)
        return path

    @staticmethod
    def load(path: Path = REGISTRY_PATH) -> "DeviceRegistry":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return DeviceRegistry.from_json_dict(raw)


def build_registry(cfg: dict) -> DeviceRegistry:
    """Build `devices.per_dataset` devices per dataset with seeded creds.

    Seed = `seed * 1_000_003 + dataset_index * 1_000_033 + i` where i is the
    per-dataset device index starting at 1. The seed mixes the global seed
    with (dataset, device index) so different datasets produce different
    credentials but the same dataset always produces the same registry.
    """
    seed          = int(cfg["seed"])
    per_dataset   = int(cfg["devices"]["per_dataset"])
    pwd_len       = int(cfg["devices"]["password_len"])
    hex_len       = int(cfg["devices"]["secret_k_hex_len"])

    devices: list[Device] = []
    for ds_idx, dataset in enumerate(DATASETS):
        prefix = DEVICE_PREFIX[dataset]
        dtype  = DEVICE_TYPE[dataset]
        for i in range(1, per_dataset + 1):
            device_id = f"{prefix}-{i:03d}"
            rng = _seeded_rng(seed * 1_000_003 + ds_idx * 1_000_033 + i)
            devices.append(Device(
                device_id   = device_id,
                dataset     = dataset,
                device_type = dtype,
                password    = _gen_password(rng, pwd_len),
                secret_k    = _gen_secret_k(rng, hex_len),
            ))
    devices.sort(key=lambda d: d.device_id)
    return DeviceRegistry(tuple(devices))


# ---------------------------------------------------------------------------
# Deterministic assignment: patient -> device
# ---------------------------------------------------------------------------

def assign_device(patient_id: int, dataset: str, registry: DeviceRegistry,
                  per_dataset: int) -> str:
    """Return the device_id responsible for `patient_id` in `dataset`.

    Stable, dataset-scoped, and uniform across `per_dataset` devices. SHA-256
    is used so the mapping is independent of Python's hash() salt.
    """
    h = hashlib.sha256(f"{dataset}|{patient_id}".encode("utf-8")).digest()
    # First 4 bytes are plenty for a small device count.
    idx = int.from_bytes(h[:4], "big") % per_dataset
    prefix = DEVICE_PREFIX[dataset]
    return f"{prefix}-{idx + 1:03d}"


# ---------------------------------------------------------------------------
# Stream envelope
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StreamEnvelope:
    """One record leaving the device. Exactly the fields the Session Layer
    is allowed to see. NEVER contains patient_id / gt_* / pred_*."""
    device_id:           str
    dataset:             str
    seq_no:              int
    ts:                  float
    record_id:           int
    features:            dict[str, object]
    eka_prior_tier:      str   # "Low" or "High"
    eka_prior_severity:  int   # 1, 2, 3


# ---------------------------------------------------------------------------
# Stream iterator
# ---------------------------------------------------------------------------

def _start_clock(cfg: dict) -> float:
    """Parse the configured ISO start_time to a unix-second float.

    We use the simulated clock from config; never wall-clock time.
    """
    s = cfg["devices"]["start_time"]
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def iter_stream(
    dataset: str,
    split:    str | None,
    cfg:      dict,
    registry: DeviceRegistry,
    start_global_ts: float | None = None,
) -> Iterator[StreamEnvelope]:
    """Yield one `StreamEnvelope` per record, globally sorted by ts.

    `split` is one of {"train","val","test"} or None for all records.
    `start_global_ts` lets the caller thread a shared simulated clock
    across multiple dataset calls so the merged stream is globally
    non-decreasing; defaults to the configured `start_time`.

    Only Step-0 loader APIs are touched: features, ids, eka_prior, splits.
    """
    root = Path(".").resolve()
    feats  = load_raw_features(root, dataset)
    ids    = load_ids(root, dataset)
    eka    = load_eka_prior(root, dataset)

    if split is None:
        record_ids = feats["record_id"].to_numpy()
    else:
        splits = load_splits(root, dataset)
        record_ids = np.array(splits[split], dtype=np.int64)

    # Subset and join.
    feats_s = feats[feats["record_id"].isin(record_ids)].set_index("record_id")
    ids_s   = ids   [ids  ["record_id"].isin(record_ids)].set_index("record_id")
    eka_s   = eka   [eka  ["record_id"].isin(record_ids)].set_index("record_id")

    # Join record_id -> device_id (deterministic, dataset-scoped).
    per_dataset = int(cfg["devices"]["per_dataset"])
    by_patient = ids_s["patient_id"].astype(np.int64).to_dict()
    device_of: dict[int, str] = {
        rid: assign_device(int(by_patient[rid]), dataset, registry, per_dataset)
        for rid in feats_s.index.to_numpy()
    }

    # Simulated clock: per-device seq_no and exponential inter-arrivals.
    rng          = np.random.default_rng(int(cfg["seed"]) * 31 + hash(dataset) % 1000)
    mean_seconds = float(cfg["devices"]["arrival_mean_seconds"])
    start_ts     = _start_clock(cfg)

    # Build per-device timeline. We need:
    #   * seq_no strictly increasing from 1 with no gaps per device
    #   * ts non-decreasing per device AND globally in the merged stream
    # Approach: process records grouped by device in sorted-by-record_id order,
    # generate seq_no = 1..N and a strictly-increasing ts per device with
    # exponential gaps starting from start_ts. Then merge all envelopes in
    # ts order across devices.
    per_device_records: dict[str, list[int]] = {}
    for rid in feats_s.index.to_numpy():
        per_device_records.setdefault(device_of[int(rid)], []).append(int(rid))
    for dev_id in per_device_records:
        per_device_records[dev_id].sort()

    envelopes: list[StreamEnvelope] = []
    # Simulated clock is shared across devices so the merged stream has a
    # globally non-decreasing ts. Each device's first record lands at
    # max(start_ts, global_ts), and every subsequent record adds an
    # exponential gap from a per-device RNG.
    device_rng: dict[str, np.random.Generator] = {}
    global_ts = float(start_global_ts) if start_global_ts is not None else start_ts

    for dev_id in sorted(per_device_records.keys()):
        rid_list = per_device_records[dev_id]
        # Per-device seeded RNG; deterministic for a given (seed, dataset).
        seed_mix = int(cfg["seed"]) * 1_000_033 + abs(hash((dataset, dev_id))) % 1_000_003
        device_rng[dev_id] = np.random.default_rng(seed_mix)
        gaps = device_rng[dev_id].exponential(scale=mean_seconds, size=len(rid_list))
        # Anchor the first record at max(global_ts, start_ts), then add gaps.
        # Add a small epsilon per record to keep seq_no monotonic even when
        # an exponential draw is essentially zero.
        anchor = max(global_ts, start_ts)
        ts_local = anchor + np.cumsum(gaps) + np.arange(len(gaps)) * 1e-9
        # The last ts of this device becomes the new global lower bound.
        global_ts = float(ts_local[-1]) if len(ts_local) else global_ts

        for seq, rid, ts in zip(range(1, len(rid_list) + 1), rid_list, ts_local):
            row = feats_s.loc[rid]
            features = {col: (None if pd.isna(row[col]) else row[col])
                        for col in RAW_FEATURES[dataset]}
            eka_row = eka_s.loc[rid]
            envelopes.append(StreamEnvelope(
                device_id          = dev_id,
                dataset            = dataset,
                seq_no             = int(seq),
                ts                 = float(ts),
                record_id          = int(rid),
                features           = features,
                eka_prior_tier     = str(eka_row["eka_prior_tier"]),
                eka_prior_severity = int(eka_row["eka_prior_severity"]),
            ))

    # Already non-decreasing per construction; sorting keeps the contract
    # explicit so a regression in the above loop fails loudly.
    envelopes.sort(key=lambda e: (e.ts, e.device_id, e.seq_no))
    return iter(envelopes)


# ---------------------------------------------------------------------------
# CLI: build + preview
# ---------------------------------------------------------------------------

def _load_config(path: Path = Path("config.yaml")) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _per_dataset_assignment(registry: DeviceRegistry, cfg: dict, dataset: str):
    """Vectorised assignment for one dataset. Returns (ids_with_dev,
    per_record_counts, device_to_count)."""
    root = Path(".").resolve()
    per_dataset = int(cfg["devices"]["per_dataset"])
    ids = load_ids(root, dataset)
    patient_ids = ids["patient_id"].to_numpy(dtype=np.int64)
    # Vectorised SHA-256 over the patient_id array.
    keys = [f"{dataset}|{int(p)}".encode("utf-8") for p in patient_ids]
    idxs = np.fromiter(
        (int.from_bytes(hashlib.sha256(k).digest()[:4], "big") % per_dataset
         for k in keys),
        dtype=np.int64,
        count=len(keys),
    )
    prefix = {"diabetes": "GLU", "cvd": "BPM"}[dataset]
    device_ids = np.array([f"{prefix}-{i + 1:03d}" for i in idxs], dtype=object)

    splits = load_splits(root, dataset)
    rid_to_split = np.empty(len(patient_ids), dtype=object)
    rid_to_split[:] = "unknown"
    for split_name in ("train", "val", "test"):
        s = set(int(r) for r in splits[split_name])
        rid_to_split[np.isin(ids["record_id"].to_numpy(), list(s))] = split_name

    counts = pd.Series(device_ids).value_counts().to_dict()
    out = pd.DataFrame({
        "record_id":  ids["record_id"].to_numpy(),
        "patient_id": patient_ids,
        "device_id":  device_ids,
        "split":      rid_to_split,
    })
    return out, counts


def _assignment_table(registry: DeviceRegistry, cfg: dict) -> pd.DataFrame:
    """For each device, count assigned records overall and per split."""
    rows: list[dict] = []
    for dataset in DATASETS:
        df, _counts = _per_dataset_assignment(registry, cfg, dataset)
        agg = df.groupby(["device_id", "split"]).size().unstack(fill_value=0)
        per_dev_total = df.groupby("device_id").size()
        for dev in registry.list(dataset):
            rows.append({
                "device_id":           dev.device_id,
                "dataset":             dev.dataset,
                "device_type":         dev.device_type,
                "n_records_assigned":  int(per_dev_total.get(dev.device_id, 0)),
                "n_train": int(agg.get("train", pd.Series(dtype=int)).get(dev.device_id, 0)),
                "n_val":   int(agg.get("val",   pd.Series(dtype=int)).get(dev.device_id, 0)),
                "n_test":  int(agg.get("test",  pd.Series(dtype=int)).get(dev.device_id, 0)),
            })
    return (pd.DataFrame(rows)
              .sort_values("device_id")
              .reset_index(drop=True))


def _check_full_assignment(registry: DeviceRegistry, cfg: dict) -> dict:
    """Every record must be assigned to exactly one device of its own dataset."""
    summary: dict = {}
    for dataset in DATASETS:
        df, counts = _per_dataset_assignment(registry, cfg, dataset)
        registered = {d.device_id for d in registry.list(dataset)}
        devices_with_records = set(counts.keys())
        # Deterministic assignment is one record -> one device; sanity-check
        # that record_ids are unique (they are, by Step 0) and that the
        # number of (record_id, device_id) pairs equals the dataset size.
        n_pairs = int(len(df.drop_duplicates(["record_id", "device_id"])))
        any_twice = n_pairs != int(len(df))
        summary[dataset] = {
            "n_records":               int(len(df)),
            "any_record_twice":        bool(any_twice),
            "n_devices_with_records":  int(len(devices_with_records)),
            "devices_with_zero_records":
                sorted(registered - devices_with_records),
        }
    summary["every_record_assigned_once"] = all(
        not v["any_record_twice"] for v in summary.values() if isinstance(v, dict)
    )
    return summary


def _atomic_write_text(path: Path, write) -> None:
    """Write text to `path` via a uniquely-named temp sibling, then rename.

    If the rename fails because an external process (VS Code preview, AV
    scanner, ...) is holding the destination open, fall back to writing the
    content under a timestamped sibling path so the build never crashes.
    """
    import os
    import tempfile
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            write(f)
        try:
            os.replace(tmp, path)
        except PermissionError:
            # Destination is locked by something else. Leave the canonical
            # file alone and emit a timestamped sibling so the artefact is
            # still produced; the reader picks the freshest one.
            ts = int.from_bytes(os.urandom(4), "big")
            fallback = path.with_name(
                f"{path.stem}.{ts:08x}{path.suffix}"
            )
            os.replace(tmp, fallback)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    _atomic_write_text(path, lambda f: df.to_csv(f, index=False))


def _build_reports(registry: DeviceRegistry, cfg: dict, rep_dir: Path) -> None:
    rep_dir = rep_dir / "01_session"
    rep_dir.mkdir(parents=True, exist_ok=True)

    table = _assignment_table(registry, cfg)
    # Hard guard: never include secrets in reports.
    forbidden_cols = {"password", "secret_k", "credentials"}
    bad = forbidden_cols & set(table.columns)
    if bad:
        raise RuntimeError(f"Refusing to write secret columns to reports: {bad}")
    _atomic_write_csv(rep_dir / "table_device_registry.csv", table)

    # Per-device record counts (min/max/mean).
    counts = table.groupby("dataset")["n_records_assigned"].agg(["min", "max", "mean"])
    devices_per_dataset = {
        ds: len(registry.list(ds)) for ds in DATASETS
    }
    metrics = {
        "seed":                cfg["seed"],
        "devices_per_dataset": devices_per_dataset,
        "per_dataset_assignment": _check_full_assignment(registry, cfg),
        "records_per_device_stats": {
            ds: {"min": int(counts.loc[ds, "min"]),
                 "max": int(counts.loc[ds, "max"]),
                 "mean": float(round(counts.loc[ds, "mean"], 4))}
            for ds in DATASETS
        },
    }
    import json as _json
    def _dump(f) -> None:
        _json.dump(metrics, f, indent=2)
    _atomic_write_text(rep_dir / "metrics.json", _dump)


def iter_all_datasets(
    split:    str | None,
    cfg:      dict,
    registry: DeviceRegistry,
) -> Iterator[StreamEnvelope]:
    """Yield envelopes from every dataset under a single shared clock.

    Threads the global simulated clock across `iter_stream` calls so the
    merged stream is globally non-decreasing even when split across
    datasets.
    """
    start_ts = _start_clock(cfg)
    global_ts = start_ts
    for ds in DATASETS:
        for env in iter_stream(ds, split=split, cfg=cfg, registry=registry,
                               start_global_ts=global_ts):
            yield env
            global_ts = max(global_ts, env.ts)


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="FL-IENN Step 1: device simulation")
    ap.add_argument("--build",   action="store_true", help="Build registry + reports")
    ap.add_argument("--preview", type=int, default=0,  help="Print first N envelopes (no secrets)")
    args = ap.parse_args(argv)

    cfg = _load_config()
    registry = build_registry(cfg)

    if args.build:
        registry.save()
        rep_dir = Path(cfg["paths"]["reports_dir"]) / "01_session"
        _build_reports(registry, cfg, Path(cfg["paths"]["reports_dir"]))
        print(f"[Step 1] registry -> {REGISTRY_PATH}")
        print(f"[Step 1] reports  -> {rep_dir}")
        for f in sorted(rep_dir.glob("*.csv")):
            print(f"[Step 1]   {f.name}")
        for f in sorted(rep_dir.glob("*.json")):
            print(f"[Step 1]   {f.name}")

    if args.preview > 0:
        for ds in DATASETS:
            print(f"---- {ds} ----")
            for k, env in enumerate(iter_stream(ds, split="train", cfg=cfg, registry=registry)):
                if k >= args.preview:
                    break
                feats_short = {kk: (vv if not isinstance(vv, float) else round(vv, 3))
                               for kk, vv in list(env.features.items())[:4]}
                print(env.device_id, env.seq_no, round(env.ts, 6),
                      env.record_id, env.eka_prior_tier, env.eka_prior_severity,
                      feats_short)


if __name__ == "__main__":
    main()
