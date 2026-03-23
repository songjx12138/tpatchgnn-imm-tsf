import os
import json
import re
import hashlib
import pandas as pd
import glob
from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
from prettytable import PrettyTable
import math
import argparse

import lib.utils as utils
from lib.task_config import (
    TaskConfig,
    get_task_config,
    get_feature_exclude_set,
    ALWAYS_EXCLUDE,
    LABEL_PREFIX,
    TASK_CONFIGS,
)


DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
TIME_COMPONENT_PATTERN = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")


def _parse_datetime(series: pd.Series) -> pd.Series:
    """
    Parse timestamp columns without relying on pandas' slow warning-prone
    inference path.

    MIMIC_sepsis uses full timestamps, while MIMIC_sepsis-full may contain
    date-only strings in some files. We support both.
    """
    try:
        return pd.to_datetime(series, format=DATETIME_FORMAT, errors="raise")
    except (TypeError, ValueError):
        try:
            return pd.to_datetime(series, format="%Y-%m-%d", errors="raise")
        except (TypeError, ValueError):
            return pd.to_datetime(series, format="mixed", errors="raise")


def _has_subday_timestamp_resolution(series: pd.Series) -> bool:
    """Return True only when every timestamp string includes hour/minute info."""
    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return False
    return bool(values.map(lambda x: bool(TIME_COMPONENT_PATTERN.search(x))).all())


def _time_resolution_cache_path(proc_dir: str) -> str:
    return os.path.join(proc_dir, "_time_resolution_cache.json")


def _load_time_resolution_cache(proc_dir: str) -> dict[str, bool]:
    cache_path = _time_resolution_cache_path(proc_dir)
    if not os.path.isfile(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(k): bool(v) for k, v in payload.items()}


def _save_time_resolution_cache(proc_dir: str, cache: dict[str, bool]) -> None:
    cache_path = _time_resolution_cache_path(proc_dir)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)


def filter_records_with_subday_timestamps(
    proc_dir: str,
    rec_ids: list[str],
    verbose: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Keep only patients whose time_series.csv has hour/min/sec resolution.

    Chunk-level tasks require sub-day timestamps for 6h/1h windows. We apply the
    same filter to patient-level tasks for consistency.
    """
    cache = _load_time_resolution_cache(proc_dir)
    cache_changed = False
    valid_recs: list[str] = []
    invalid_recs: list[str] = []

    for rec in rec_ids:
        if rec not in cache:
            ts_path = os.path.join(proc_dir, rec, "time_series.csv")
            if not os.path.isfile(ts_path):
                cache[rec] = False
            else:
                df = pd.read_csv(ts_path, usecols=["date_time"])
                cache[rec] = _has_subday_timestamp_resolution(df["date_time"])
            cache_changed = True

        if cache[rec]:
            valid_recs.append(rec)
        else:
            invalid_recs.append(rec)

    if cache_changed:
        _save_time_resolution_cache(proc_dir, cache)

    if verbose and invalid_recs:
        preview = ", ".join(invalid_recs[:5])
        suffix = " ..." if len(invalid_recs) > 5 else ""
        print(
            f"Skipping {len(invalid_recs)} records without hour/min/sec timestamps "
            f"in time_series.csv: {preview}{suffix}"
        )

    return valid_recs, invalid_recs


def _get_data_device(args) -> torch.device:
    return getattr(args, "data_device", torch.device("cpu"))


# ---------------------------------------------------------------------------
# L1 Cache: raw per-patient tensors (seed-independent, no normalization)
# ---------------------------------------------------------------------------

def _load_raw_patient_data(
    proc_dir: str,
    rec_ids: list[str],
    task_config: TaskConfig | None,
    task: str,
    time_unit: str,
    unit_scale: float | None,
    enable_text: bool,
    use_text_embeddings: bool,
    llm_model_fusion: str | None,
    llm_layers_fusion: int | None,
    max_length: int,
    sec_per_unit: float,
) -> dict:
    """
    Load raw (unnormalized) per-patient tensors from CSV or L1 cache.
    Returns dict with keys: raw_data, feature_names, label_names.
    raw_data is list of (rec_id, tt, vals_raw, mask, texts, labels).
    All tensors are on CPU and NOT normalized.
    """
    # Use exclusion set (not task name) in cache key so tasks with identical
    # feature sets share the same L1 cache (e.g. morta_hosp, morta_90, los)
    if task_config is not None:
        task_exclude = get_feature_exclude_set(task_config.name)
    else:
        task_exclude = set(ALWAYS_EXCLUDE)
    exclude_tag = _cache_key_hash(*sorted(task_exclude)) if task_exclude else "noexcl"

    # Label columns also affect L1 (window_cls tasks store labels in raw_data)
    label_tag = ""
    if task_config is not None and task_config.label_source == "timeseries":
        label_tag = task_config.label_col
    elif task == "classification":
        label_tag = "all_labels"

    text_tag = "notext"
    if enable_text:
        text_tag = (
            f"emb_{llm_model_fusion}_{llm_layers_fusion}_{max_length}"
            if use_text_embeddings else "raw"
        )
    cache_hash = _cache_key_hash(exclude_tag, label_tag, time_unit, unit_scale, text_tag, *rec_ids)
    cache_dir = os.path.join(proc_dir, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"L1_raw_{exclude_tag}_{text_tag}_{cache_hash}.pt")

    if os.path.isfile(cache_path):
        print(f"  [L1 cache] Loading raw patient data from {os.path.basename(cache_path)}")
        return torch.load(cache_path, map_location="cpu")

    print(f"  [L1 cache] Reading {len(rec_ids)} patient CSVs (first run, will cache)...")

    target_label_col = None
    if task_config is not None and task_config.label_source == "timeseries":
        target_label_col = task_config.label_col

    raw_data = []
    feature_names = []
    label_names = []

    for rec in rec_ids:
        ts_path = os.path.join(proc_dir, rec, "time_series.csv")
        if not os.path.isfile(ts_path):
            continue

        df = pd.read_csv(ts_path)
        if not _has_subday_timestamp_resolution(df["date_time"]):
            continue
        df["_ts_raw"] = _parse_datetime(df["date_time"])
        df = df.sort_values("_ts_raw")

        feat_cols = [
            c for c in df.columns
            if c not in ("date_time", "record_id", "_ts_raw")
            and not c.startswith(LABEL_PREFIX)
            and c not in task_exclude
        ]
        if not feature_names:
            feature_names = feat_cols

        # Label columns
        if target_label_col and target_label_col in df.columns:
            label_cols = [target_label_col]
        elif task == "classification":
            label_cols = [c for c in df.columns if c.startswith(LABEL_PREFIX)]
        else:
            label_cols = []
        if task == "classification" and label_cols and not label_names:
            label_names = label_cols

        # Time -> float (unnormalized by time_unit)
        secs = (df["_ts_raw"] - df["_ts_raw"].min()).dt.total_seconds()
        units = secs / sec_per_unit
        tt = torch.tensor(units.values, dtype=torch.float32)

        # Raw values & mask (NO normalization)
        vals_np = df[feat_cols].values.astype("float32")
        mask_np = ~pd.isna(vals_np)
        vals = torch.nan_to_num(torch.tensor(vals_np))
        mask = torch.tensor(mask_np.astype("float32"))

        # Labels
        if task == "classification" and label_cols:
            labels_np = df[label_cols].fillna(0).values.astype("float32")
            labels = torch.tensor(labels_np)
        else:
            labels = None

        if mask.sum() == 0:
            continue

        # Text data
        texts: list[tuple[float, object]] = []
        if use_text_embeddings and llm_model_fusion and enable_text:
            fname = (
                f"text_embeddings_model={llm_model_fusion}"
                f"_layers={llm_layers_fusion or 'full'}"
                f"_maxlen={max_length}.pt"
            )
            path = os.path.join(proc_dir, rec, fname)
            if os.path.isfile(path):
                data = torch.load(path, map_location="cpu")
                emb = data["embeddings"]
                if torch.isnan(emb).any():
                    raise ValueError("text embeddings contains NaN values.")
                rel = data["rel_times"] / sec_per_unit
                for i, t in enumerate(rel):
                    texts.append((t.item(), emb[i]))
            else:
                text_path = os.path.join(proc_dir, rec, "text.csv")
                if os.path.isfile(text_path):
                    raise FileNotFoundError(f"Missing text embeddings file: {path}")
        else:
            text_path = os.path.join(proc_dir, rec, "text.csv")
            if os.path.isfile(text_path):
                tdf = pd.read_csv(text_path)
                tdf["date_time"] = _parse_datetime(tdf["date_time"])
                tdf = tdf.sort_values("date_time")
                cols = [c for c in tdf.columns if c not in ("date_time", "record_id")]
                if len(cols) != 1:
                    raise ValueError(f"{rec}: expected 1 text column, got {cols}")
                text_col = cols[0]
                base = df["_ts_raw"].min()
                for _, row in tdf.iterrows():
                    txt = row[text_col]
                    if pd.isna(txt):
                        continue
                    t_rel = (row["date_time"] - base).total_seconds() / sec_per_unit
                    texts.append((t_rel, txt))

        raw_data.append((rec, tt, vals, mask, texts, labels))

    result = {
        "raw_data": raw_data,
        "feature_names": feature_names,
        "label_names": label_names,
    }

    print(f"  [L1 cache] Saving {len(raw_data)} patients to {os.path.basename(cache_path)}...")
    torch.save(result, cache_path)
    print(f"  [L1 cache] Saved.")
    return result


def _normalize_raw_data(raw_data, feature_names, global_stats):
    """
    Apply global normalization to raw patient tensors in-place (fast, no I/O).
    Returns the same raw_data list with vals tensors normalized.
    """
    if not global_stats:
        return raw_data

    # Build normalization vectors once
    n_feat = len(feature_names)
    mean_vec = torch.zeros(n_feat)
    std_vec = torch.ones(n_feat)
    for i, fname in enumerate(feature_names):
        if fname in global_stats:
            m, s = global_stats[fname]
            mean_vec[i] = m
            std_vec[i] = s if s > 0 else 1.0

    normalized = []
    for rec, tt, vals, mask, texts, labels in raw_data:
        vals_norm = (vals - mean_vec) / std_vec
        # Zero out where mask is 0 (was NaN originally)
        vals_norm = vals_norm * mask
        normalized.append((rec, tt, vals_norm, mask, texts, labels))
    return normalized


def _build_discrete_risk_target(pred_window: int, rel_event_time: float | None) -> torch.Tensor:
    """
    Build a discrete-time onset target over the prediction horizon.

    The target is all zeros for negative windows. For positive onset windows we
    place a 1 in the earliest future hour bin whose upper boundary contains the
    onset time. Example with pred_window=6:
      rel_event_time in [0, 1] -> bin 0
      rel_event_time in (1, 2] -> bin 1
      ...
    """
    horizon = max(int(pred_window), 1)
    target = torch.zeros(horizon, dtype=torch.float32)
    if rel_event_time is None or not math.isfinite(float(rel_event_time)):
        return target

    rel = max(float(rel_event_time), 0.0)
    event_bin = int(math.ceil(rel) - 1)
    event_bin = max(0, min(horizon - 1, event_bin))
    target[event_bin] = 1.0
    return target


class ChunkedTimeSeriesDataset(Dataset):
    """
    Dataset for irregular time series with optional text embeddings.
    Supports task-aware feature exclusion and single-label extraction.
    """

    UNIT_SECONDS = {
        "seconds": 1.0,
        "minutes": 60.0,
        "hours": 3600.0,
        "days": 86400.0,
        "weeks": 604800.0,
    }

    def __init__(
        self,
        root: str,
        history: int,
        pred_window: int,
        stride: int,
        device: torch.device = torch.device("cpu"),
        time_unit: str = "days",
        unit_scale: float | None = None,
        normalize: bool = True,
        enable_text: bool = False,
        use_text_embeddings: bool = False,
        llm_model_fusion: str | None = None,
        llm_layers_fusion: int | None = None,
        max_length: int = 1024,
        args: argparse.Namespace | None = None,
        task: str = "forecasting",
        labels_path: str | None = None,
        prediction_horizon: int | None = None,
        task_config: TaskConfig | None = None,
        global_stats: dict | None = None,
    ):
        super().__init__()
        self.history = history
        self.pred_window = pred_window
        self.stride = stride
        self.device = device
        self.normalize = normalize
        self.enable_text = enable_text
        self.use_text_embeddings = use_text_embeddings
        self.llm_model_fusion = llm_model_fusion
        self.llm_layers_fusion = llm_layers_fusion
        self.task = task
        self.task_config = task_config
        self.label_names: list[str] = []
        self.feature_names: list[str] = []

        # determine time-unit scale
        if time_unit == "custom":
            if unit_scale is None:
                raise ValueError("Must set unit_scale when time_unit='custom'")
            self._sec_per_unit = float(unit_scale)
        else:
            try:
                self._sec_per_unit = self.UNIT_SECONDS[time_unit]
            except KeyError:
                raise ValueError(f"Unknown time_unit '{time_unit}'")

        proc_dir = os.path.join(root, "processed")
        rec_ids = sorted(
            d for d in os.listdir(proc_dir) if os.path.isdir(os.path.join(proc_dir, d))
        )

        if isinstance(args, argparse.Namespace) and getattr(args, "rec_ids", None) is not None:
            rec_ids = args.rec_ids

        # ---- L1: load raw (unnormalized) patient data from cache or CSV ----
        l1 = _load_raw_patient_data(
            proc_dir, rec_ids, task_config, task, time_unit, unit_scale,
            enable_text, use_text_embeddings, llm_model_fusion,
            llm_layers_fusion, max_length, self._sec_per_unit,
        )
        self.feature_names = l1["feature_names"]
        self.label_names = l1["label_names"]
        raw_data = l1["raw_data"]

        # ---- L2: normalize in memory (fast, no I/O) ----
        if normalize and global_stats:
            raw_data = _normalize_raw_data(raw_data, self.feature_names, global_stats)

        # ---- Chunking (pure tensor ops, fast) ----
        onset_only = task_config.onset_only if task_config else False
        total = history + pred_window
        chunks: list[tuple] = []
        for rec, tt, vals, mask, record_texts, rec_labels in raw_data:
            t_max = tt.max().item()
            st = tt.min().item()
            cnt = 0
            skip_onset = 0

            first_onset_time = float("inf")
            if onset_only and rec_labels is not None:
                label_vals = rec_labels[:, 0]
                pos_idx = (label_vals > 0).nonzero(as_tuple=False)
                if pos_idx.numel() > 0:
                    first_onset_time = tt[pos_idx[0].item()].item()

            while st + total <= t_max:
                idx = (
                    ((tt >= st) & (tt < st + total)).nonzero(as_tuple=False).squeeze(1)
                )
                if idx.numel() >= 2:
                    sub_tt = tt[idx] - st
                    sub_vals = vals[idx]
                    sub_mask = mask[idx]

                    hist_mask = sub_mask[sub_tt < history]
                    pred_mask = sub_mask[sub_tt >= history]

                    if hist_mask.sum() == 0 or pred_mask.sum() == 0:
                        st += stride
                        continue

                    if sub_mask.sum() == 0:
                        raise ValueError(f"Sub mask for {rec} is all zeros")

                    if onset_only and rec_labels is not None:
                        hist_end_abs = st + history
                        if hist_end_abs > first_onset_time:
                            skip_onset += 1
                            st += stride
                            continue
                        pred_end_abs = st + total
                        if first_onset_time < pred_end_abs:
                            chunk_labels = torch.tensor([1.0])
                            rel_onset = first_onset_time - hist_end_abs
                            chunk_risk_targets = _build_discrete_risk_target(
                                pred_window, rel_onset
                            )
                        else:
                            chunk_labels = torch.tensor([0.0])
                            chunk_risk_targets = _build_discrete_risk_target(
                                pred_window, None
                            )
                    elif task == "classification" and rec_labels is not None:
                        sub_labels = rec_labels[idx]
                        pred_labels = sub_labels[sub_tt >= history]
                        chunk_labels = (pred_labels.sum(dim=0) > 0).float()
                        chunk_risk_targets = None
                    else:
                        chunk_labels = None
                        chunk_risk_targets = None

                    hist_end = st + history
                    selected = [
                        (t - st, payload)
                        for (t, payload) in record_texts
                        if st <= t < hist_end
                    ]
                    chunk_id = f"{rec}_chunk{cnt}"
                    cnt += 1

                    if enable_text:
                        chunks.append(
                            (
                                chunk_id,
                                sub_tt,
                                sub_vals,
                                sub_mask,
                                selected,
                                chunk_labels,
                                chunk_risk_targets,
                            )
                        )
                    else:
                        chunks.append(
                            (
                                chunk_id,
                                sub_tt,
                                sub_vals,
                                sub_mask,
                                [],
                                chunk_labels,
                                chunk_risk_targets,
                            )
                        )
                st += stride

            if getattr(args, "show_record_chunk_counts", False):
                if cnt == 0 and skip_onset == 0:
                    print(f"Record {rec}: skipped (data too short for history={history}h)")
                else:
                    extra = f", {skip_onset} post-onset skipped" if skip_onset else ""
                    print(f"Record {rec}: {cnt} chunks{extra}")

        if not chunks:
            raise RuntimeError("No chunks created; check history/pred_window/stride")
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


class PatientLevelDataset(Dataset):
    """
    Dataset for patient-level tasks (T5-T7).
    Each sample = one patient's full time series + label from labels.csv.
    No chunking; the entire trajectory is one sample.
    """

    UNIT_SECONDS = ChunkedTimeSeriesDataset.UNIT_SECONDS

    def __init__(
        self,
        root: str,
        task_config: TaskConfig,
        device: torch.device = torch.device("cpu"),
        time_unit: str = "days",
        unit_scale: float | None = None,
        normalize: bool = True,
        enable_text: bool = False,
        use_text_embeddings: bool = False,
        llm_model_fusion: str | None = None,
        llm_layers_fusion: int | None = None,
        max_length: int = 1024,
        args: argparse.Namespace | None = None,
        global_stats: dict | None = None,
    ):
        super().__init__()
        self.task_config = task_config
        self.device = device
        self.enable_text = enable_text
        self.use_text_embeddings = use_text_embeddings
        self.feature_names: list[str] = []
        self.max_seq_time = 0.0

        if time_unit == "custom":
            if unit_scale is None:
                raise ValueError("Must set unit_scale when time_unit='custom'")
            self._sec_per_unit = float(unit_scale)
        else:
            self._sec_per_unit = self.UNIT_SECONDS.get(time_unit)
            if self._sec_per_unit is None:
                raise ValueError(f"Unknown time_unit '{time_unit}'")

        proc_dir = os.path.join(root, "processed")
        rec_ids = sorted(
            d for d in os.listdir(proc_dir) if os.path.isdir(os.path.join(proc_dir, d))
        )
        if isinstance(args, argparse.Namespace) and getattr(args, "rec_ids", None) is not None:
            rec_ids = args.rec_ids

        # ---- L1: load raw (unnormalized) patient data from cache or CSV ----
        # PatientLevelDataset uses task="classification" or "regression" but
        # L1 cache doesn't care about task type, only feature exclusion
        task_str = "classification" if task_config.task_type == "patient_cls" else "regression"
        l1 = _load_raw_patient_data(
            proc_dir, rec_ids, task_config, task_str, time_unit, unit_scale,
            enable_text, use_text_embeddings, llm_model_fusion,
            llm_layers_fusion, max_length, self._sec_per_unit,
        )
        self.feature_names = l1["feature_names"]
        raw_data = l1["raw_data"]

        # ---- L2: normalize in memory (fast, no I/O) ----
        if normalize and global_stats:
            raw_data = _normalize_raw_data(raw_data, self.feature_names, global_stats)

        # ---- Build patient samples with labels from labels.csv ----
        labels_path = os.path.join(root, "labels.csv")
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(f"labels.csv not found at {labels_path}")
        labels_df = pd.read_csv(labels_path)
        labels_df["stay_id"] = labels_df["stay_id"].astype(str)
        label_col = task_config.label_col
        if label_col not in labels_df.columns:
            raise ValueError(f"Label column '{label_col}' not in labels.csv")

        # Build a lookup for fast label access
        label_lookup = dict(zip(labels_df["stay_id"], labels_df[label_col]))

        self.samples = []
        self.rec_ids = []
        self.labels_raw = []

        for rec, tt, vals, mask, texts, _labels in raw_data:
            # Check label exists
            label_value = label_lookup.get(rec)
            if label_value is None or pd.isna(label_value):
                continue
            label_value = float(label_value)

            if tt.numel() > 0:
                self.max_seq_time = max(self.max_seq_time, float(tt.max().item()))

            # For regression (los), apply log(1+x) transform
            if task_config.task_type == "patient_reg":
                label_value = float(np.log1p(label_value))

            self.samples.append((rec, tt, vals, mask, texts, label_value))
            self.rec_ids.append(rec)
            self.labels_raw.append(label_value)

        if not self.samples:
            raise RuntimeError(f"No patient samples for task '{task_config.name}'")
        print(f"PatientLevelDataset: {len(self.samples)} patients for task '{task_config.name}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec, tt, vals, mask, texts, label = self.samples[idx]
        if not self.enable_text:
            texts = []
        return (rec, tt, vals, mask, texts, torch.tensor([label], dtype=torch.float32))


#####################################################################################################
# Global Normalization
#####################################################################################################


def _cache_key_hash(*parts) -> str:
    """Compute a short deterministic hash from arbitrary string parts."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]
    return h


def compute_global_stats(
    root: str,
    rec_ids: list[str],
    train_rec_ids: set[str],
    task_config: TaskConfig | None = None,
) -> dict[str, tuple[float, float]]:
    """
    Compute per-feature mean/std from training set patients only.
    Returns dict: feature_name -> (mean, std).
    Results are cached to a JSON file for fast reuse.
    """
    proc_dir = os.path.join(root, "processed")
    task_name = task_config.name if task_config else "_default"
    sorted_train = sorted(train_rec_ids)
    cache_hash = _cache_key_hash(task_name, *sorted_train)
    cache_dir = os.path.join(root, "processed", "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"global_stats_{task_name}_{cache_hash}.json")

    if os.path.isfile(cache_path):
        print(f"  [cache] Loading global_stats from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: (v[0], v[1]) for k, v in raw.items()}

    print(f"  [cache] Computing global_stats for task={task_name} ({len(sorted_train)} train patients)...")
    task_exclude = get_feature_exclude_set(task_config.name) if task_config else set(ALWAYS_EXCLUDE)

    # First pass: determine feature columns
    feat_cols = None
    for rec in rec_ids:
        ts_path = os.path.join(proc_dir, rec, "time_series.csv")
        if os.path.isfile(ts_path):
            df = pd.read_csv(ts_path, nrows=1)
            feat_cols = [
                c for c in df.columns
                if c not in ("date_time", "record_id")
                and not c.startswith(LABEL_PREFIX)
                and c not in task_exclude
            ]
            break
    if feat_cols is None:
        raise RuntimeError("No valid time_series.csv found")

    # Accumulate running sums from train patients only
    n = {c: 0 for c in feat_cols}
    s = {c: 0.0 for c in feat_cols}
    s2 = {c: 0.0 for c in feat_cols}

    for rec in train_rec_ids:
        ts_path = os.path.join(proc_dir, rec, "time_series.csv")
        if not os.path.isfile(ts_path):
            continue
        df = pd.read_csv(ts_path)
        for c in feat_cols:
            if c not in df.columns:
                continue
            vals = df[c].dropna().values.astype("float64")
            n[c] += len(vals)
            s[c] += vals.sum()
            s2[c] += (vals ** 2).sum()

    stats = {}
    for c in feat_cols:
        if n[c] > 0:
            mean = s[c] / n[c]
            var = s2[c] / n[c] - mean ** 2
            std = float(np.sqrt(max(var, 0.0)))
            stats[c] = (float(mean), std)
        else:
            stats[c] = (0.0, 1.0)

    # Save cache
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False)
    print(f"  [cache] Saved global_stats to {cache_path}")

    return stats


#####################################################################################################
# Patient-level split utilities
#####################################################################################################


def split_patients_stratified(
    rec_ids: list[str],
    labels: list[float],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    random_state: int = 42,
    is_classification: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split patient IDs into train/val/test with stratification for classification.
    For regression, uses simple random split.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0

    # Determine if stratification is feasible
    stratify = None
    if is_classification:
        int_labels = [int(l) for l in labels]
        from collections import Counter
        class_counts = Counter(int_labels)
        # Need at least 2 members per class for stratified split
        if all(c >= 2 for c in class_counts.values()):
            stratify = int_labels
        else:
            print(f"  Warning: class counts {dict(class_counts)} too small for stratification, using random split.")

    train_recs, temp_recs, _, temp_labels = train_test_split(
        rec_ids, labels,
        train_size=train_ratio,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )
    val_frac = val_ratio / (val_ratio + test_ratio)
    temp_stratify = None
    if is_classification and stratify is not None:
        int_temp = [int(l) for l in temp_labels]
        from collections import Counter as C2
        tc = C2(int_temp)
        if all(c >= 2 for c in tc.values()):
            temp_stratify = int_temp
        else:
            print(f"  Warning: temp class counts {dict(tc)} too small, falling back to random split for val/test.")

    val_recs, test_recs = train_test_split(
        temp_recs,
        train_size=val_frac,
        random_state=random_state,
        shuffle=True,
        stratify=temp_stratify,
    )
    return train_recs, val_recs, test_recs


def get_window_task_patient_labels(
    proc_dir: str,
    rec_ids: list[str],
    label_col: str,
) -> tuple[list[str], list[float]]:
    """
    Collapse window-level labels to a patient-level binary target so patient splits
    can stay approximately label-balanced.
    Results are cached to JSON for fast reuse.
    """
    cache_hash = _cache_key_hash(label_col, *rec_ids)
    cache_dir = os.path.join(proc_dir, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"window_patient_labels_{label_col}_{cache_hash}.json")

    if os.path.isfile(cache_path):
        print(f"  [cache] Loading window_patient_labels from {os.path.basename(cache_path)}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["recs"], data["labels"]

    print(f"  [cache] Computing window_patient_labels for {label_col}...")
    valid_recs = []
    patient_labels = []

    for rec in rec_ids:
        ts_path = os.path.join(proc_dir, rec, "time_series.csv")
        if not os.path.isfile(ts_path):
            continue
        df = pd.read_csv(ts_path)
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in {ts_path}")

        y = df[label_col].fillna(0).to_numpy(dtype=float)
        valid_recs.append(rec)
        patient_labels.append(float((y > 0).any()))

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"recs": valid_recs, "labels": patient_labels}, f)
    print(f"  [cache] Saved window_patient_labels to {os.path.basename(cache_path)}")

    return valid_recs, patient_labels


def split_chunk_indices(
    all_chunks: list[tuple],
    task_config: TaskConfig | None,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Fallback split for tiny datasets where patient-level window splitting leaves
    an empty train/val/test split after chunk generation.
    """
    chunk_indices = list(range(len(all_chunks)))
    if len(chunk_indices) < 3:
        raise ValueError("Need at least 3 chunks to build non-empty train/val/test splits.")

    stratify_labels = None
    if task_config is not None and task_config.task_type == "window_cls":
        labels = []
        for chunk in all_chunks:
            chunk_label = chunk[5]
            label_val = 0.0 if chunk_label is None else float(chunk_label.view(-1)[0].item())
            labels.append(int(label_val > 0))
        if len(set(labels)) >= 2 and min(labels.count(0), labels.count(1)) >= 2:
            stratify_labels = labels

    train_idx, temp_idx = train_test_split(
        chunk_indices,
        train_size=0.6,
        random_state=seed,
        shuffle=True,
        stratify=stratify_labels,
    )

    temp_stratify = None
    if stratify_labels is not None:
        temp_labels = [stratify_labels[i] for i in temp_idx]
        if len(set(temp_labels)) >= 2 and min(temp_labels.count(0), temp_labels.count(1)) >= 2:
            temp_stratify = temp_labels

    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=0.5,
        random_state=seed,
        shuffle=True,
        stratify=temp_stratify,
    )
    return list(train_idx), list(val_idx), list(test_idx)


def split_available_chunk_patients(
    proc_dir: str,
    rec_ids: list[str],
    task_config: TaskConfig | None,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """
    Re-split only across patients that produced at least one valid chunk.

    This preserves patient-level separation even when the initial patient split
    becomes empty after chunk filtering.
    """
    if len(rec_ids) < 3:
        raise ValueError(
            "Need at least 3 patients with valid chunks to build train/val/test splits."
        )

    if task_config is not None and task_config.task_type == "window_cls":
        split_recs, split_labels = get_window_task_patient_labels(
            proc_dir, rec_ids, task_config.label_col,
        )
        return split_patients_stratified(
            split_recs,
            split_labels,
            train_ratio=0.6,
            val_ratio=0.2,
            random_state=seed,
            is_classification=True,
        )

    train_recs, temp_recs = train_test_split(
        rec_ids, train_size=0.6, random_state=seed, shuffle=True,
    )
    val_recs, test_recs = train_test_split(
        temp_recs, train_size=0.5, random_state=seed, shuffle=True,
    )
    return train_recs, val_recs, test_recs


#####################################################################################################
# Collate Functions
#####################################################################################################


def variable_time_collate_fn(batch, args, time_max=None):
    data_device = _get_data_device(args)
    observed_tp, observed_data, observed_mask = [], [], []
    predicted_tp, predicted_data, predicted_mask = [], [], []
    chunk_time_max = (
        time_max
        if time_max is not None
        else torch.tensor(args.history + args.pred_window, device=data_device)
    )
    for _, tt, vals, mask in batch:
        hist_idx = torch.where(tt < args.history)[0]
        pred_idx = torch.where(tt >= args.history)[0]
        if mask[pred_idx].sum() == 0:
            raise ValueError(
                f"Mask for batch is all zeros in collate_fn, predicted index: {pred_idx}"
            )
        observed_tp.append(tt[hist_idx])
        observed_data.append(vals[hist_idx])
        observed_mask.append(mask[hist_idx])
        predicted_tp.append(tt[pred_idx])
        predicted_data.append(vals[pred_idx])
        predicted_mask.append(mask[pred_idx])
    observed_tp = pad_sequence(observed_tp, batch_first=True, padding_value=0.0)
    observed_data = pad_sequence(observed_data, batch_first=True, padding_value=0.0)
    observed_mask = pad_sequence(observed_mask, batch_first=True, padding_value=0.0)
    predicted_tp = pad_sequence(predicted_tp, batch_first=True, padding_value=0.0)
    predicted_data = pad_sequence(predicted_data, batch_first=True, padding_value=0.0)
    predicted_mask = pad_sequence(predicted_mask, batch_first=True, padding_value=0.0)

    observed_tp = utils.normalize_masked_tp(
        observed_tp, att_min=0.0, att_max=chunk_time_max
    )
    predicted_tp = utils.normalize_masked_tp(
        predicted_tp, att_min=0.0, att_max=chunk_time_max
    )
    fusion_query_tp = torch.full(
        (len(batch), 1),
        fill_value=float(args.history) / float(chunk_time_max.item()),
        device=data_device,
    )
    return {
        "observed_data": observed_data,
        "observed_tp": observed_tp,
        "observed_mask": observed_mask,
        "data_to_predict": predicted_data,
        "tp_to_predict": predicted_tp,
        "mask_predicted_data": predicted_mask,
        "fusion_query_tp": fusion_query_tp,
    }


def patch_variable_time_collate_fn(batch, args, time_max=None):
    if not batch:
        return None
    data_device = _get_data_device(args)
    D = batch[0][2].shape[1]
    chunk_time_max = (
        time_max
        if time_max is not None
        else torch.tensor(args.history + args.pred_window, device=data_device)
    )
    obs_tps, obs_vals, obs_masks = [], [], []
    pred_tps, pred_vals, pred_masks = [], [], []
    for _, tt, vals, mask in batch:
        hidx = torch.where(tt < args.history)[0]
        pidx = torch.where(tt >= args.history)[0]
        obs_tps.append(tt[hidx])
        obs_vals.append(vals[hidx])
        obs_masks.append(mask[hidx])
        pred_tps.append(tt[pidx])
        pred_vals.append(vals[pidx])
        pred_masks.append(mask[pidx])
    ptp = pad_sequence(pred_tps, batch_first=True, padding_value=0.0)
    pval = pad_sequence(pred_vals, batch_first=True, padding_value=0.0)
    pmask = pad_sequence(pred_masks, batch_first=True, padding_value=0.0)
    non_empty = [t for t in obs_tps if len(t) > 0]
    if non_empty:
        combined_tt, inv = torch.unique(
            torch.cat(non_empty), sorted=True, return_inverse=True
        )
        n_pts = len(combined_tt)
    else:
        combined_tt = torch.tensor([], device=data_device)
        inv = torch.tensor([], dtype=torch.long, device=data_device)
        n_pts = 0
    B = len(batch)
    combined_vals = torch.zeros(B, n_pts, D, device=data_device)
    combined_mask = torch.zeros(B, n_pts, D, device=data_device)
    offset = 0
    for i in range(B):
        tpi = obs_tps[i]
        if len(tpi) > 0:
            idxs = inv[offset : offset + len(tpi)]
            combined_vals[i, idxs] = obs_vals[i]
            combined_mask[i, idxs] = obs_masks[i]
            offset += len(tpi)
    norm_combined_tt = utils.normalize_masked_tp(
        combined_tt, att_min=0.0, att_max=chunk_time_max
    )
    norm_ptp = utils.normalize_masked_tp(ptp, att_min=0.0, att_max=chunk_time_max)
    fusion_query_tp = torch.full(
        (B, 1),
        fill_value=float(args.history) / float(chunk_time_max.item()),
        device=data_device,
    )
    unnorm_tt = combined_tt
    patch_indices = []
    patch_size = args.patch_size
    patch_stride = args.patch_stride
    for i in range(args.npatch):
        st = i * patch_stride
        ed = st + patch_size
        if i == args.npatch - 1:
            mask_idx = (unnorm_tt >= st) & (unnorm_tt < args.history)
        else:
            mask_idx = (unnorm_tt >= st) & (unnorm_tt < ed)
        patch_indices.append(torch.where(mask_idx)[0])
    data_dict = {
        "data": combined_vals,
        "time_steps": norm_combined_tt,
        "mask": combined_mask,
        "data_to_predict": pval,
        "tp_to_predict": norm_ptp,
        "mask_predicted_data": pmask,
        "fusion_query_tp": fusion_query_tp,
    }
    out = utils.split_and_patch_batch(data_dict, args, n_pts, patch_indices)
    out["fusion_query_tp"] = fusion_query_tp
    return out


def patient_level_collate_fn(batch, args, time_max=None):
    """
    Collate for PatientLevelDataset.
    Each sample is a full patient trajectory — no history/pred split.
    We treat the entire sequence as 'observed' and create patches over it.
    """
    if not batch:
        return None
    data_device = _get_data_device(args)

    D = batch[0][2].shape[1]
    B = len(batch)

    # Pad all sequences
    all_tps = [item[1] for item in batch]
    all_vals = [item[2] for item in batch]
    all_masks = [item[3] for item in batch]

    # Use a fixed dataset-level time scale when provided so patching/normalization
    # does not change with batch composition.
    if time_max is None:
        t_maxes = [tt.max().item() for tt in all_tps]
        global_t_max = max(t_maxes) if t_maxes else 1.0
    else:
        global_t_max = float(time_max)
    global_t_max = max(global_t_max, 1.0)
    t_max_tensor = torch.tensor(global_t_max, dtype=torch.float32, device=data_device)

    # Combine all unique time points
    non_empty = [t for t in all_tps if len(t) > 0]
    if non_empty:
        combined_tt, inv = torch.unique(
            torch.cat(non_empty), sorted=True, return_inverse=True
        )
        n_pts = len(combined_tt)
    else:
        combined_tt = torch.tensor([], device=data_device)
        inv = torch.tensor([], dtype=torch.long, device=data_device)
        n_pts = 0

    combined_vals = torch.zeros(B, n_pts, D, device=data_device)
    combined_mask = torch.zeros(B, n_pts, D, device=data_device)
    offset = 0
    for i in range(B):
        tpi = all_tps[i]
        if len(tpi) > 0:
            idxs = inv[offset : offset + len(tpi)]
            combined_vals[i, idxs] = all_vals[i]
            combined_mask[i, idxs] = all_masks[i]
            offset += len(tpi)

    norm_tt = utils.normalize_masked_tp(combined_tt, att_min=0.0, att_max=t_max_tensor)
    fusion_query_tp = torch.tensor(
        [
            float(tt.max().item()) / global_t_max if len(tt) > 0 else 0.0
            for tt in all_tps
        ],
        dtype=torch.float32,
        device=data_device,
    ).unsqueeze(-1)

    # Create patches over the full sequence
    npatch = args.npatch
    patch_size = (
        float(args.patch_size)
        if args.patch_size is not None
        else global_t_max / npatch
    )
    patch_stride = (
        float(args.patch_stride)
        if args.patch_stride is not None
        else patch_size
    )
    patch_indices = []
    for i in range(npatch):
        st = i * patch_stride
        ed = st + patch_size
        if i == npatch - 1:
            mask_idx = (combined_tt >= st) & (combined_tt <= global_t_max)
        else:
            mask_idx = (combined_tt >= st) & (combined_tt < ed)
        patch_indices.append(torch.where(mask_idx)[0])

    # We use observed_data/observed_tp/observed_mask for the encoder
    # data_to_predict is dummy (not used for classification/regression)
    data_dict = {
        "data": combined_vals,
        "time_steps": norm_tt,
        "mask": combined_mask,
        "data_to_predict": torch.zeros(B, 1, D, device=data_device),
        "tp_to_predict": torch.zeros(B, 1, device=data_device),
        "mask_predicted_data": torch.zeros(B, 1, D, device=data_device),
        "fusion_query_tp": fusion_query_tp,
    }
    out = utils.split_and_patch_batch(data_dict, args, n_pts, patch_indices)
    out["fusion_query_tp"] = fusion_query_tp

    # Add labels
    labels = torch.stack([item[5] for item in batch], dim=0)  # (B, 1)
    out["labels"] = labels

    return out


def build_text_guided_graph_batch(
    raws,
    args,
    time_max,
    numeric_tts=None,
    cutoff=None,
    patch_size=None,
    patch_stride=None,
):
    """
    Build text-event tensors aligned to numeric patch windows.
    """
    if not args.use_text_embeddings:
        raise ValueError(
            "--text_guided_graph requires --use_text_embeddings so text events are vectors."
        )

    B = len(raws)
    M = args.npatch
    N_txt = 1
    data_device = _get_data_device(args)
    cutoff = float(args.history if cutoff is None else cutoff)
    patch_size = float(args.patch_size if patch_size is None else patch_size)
    patch_stride = float(args.patch_stride if patch_stride is None else patch_stride)

    d_txt = getattr(args, "d_txt", None)
    if d_txt is None:
        for seq in raws:
            for _, payload in seq:
                if torch.is_tensor(payload):
                    d_txt = int(payload.size(-1))
                    break
            if d_txt is not None:
                break
    if d_txt is None:
        d_txt = 0

    filtered_raws = []
    patch_count_hists = []
    max_events_per_patch = 0
    for seq in raws:
        kept = [(float(t), e) for (t, e) in seq if float(t) <= cutoff]
        filtered_raws.append(kept)

        patch_counts = []
        for i in range(M):
            st = i * patch_stride
            ed = min(st + patch_size, cutoff)
            if i == M - 1:
                cnt = sum(1 for (t, _) in kept if (t >= st and t <= cutoff))
            else:
                cnt = sum(1 for (t, _) in kept if (t >= st and t < ed))
            patch_counts.append(cnt)
            if cnt > max_events_per_patch:
                max_events_per_patch = cnt
        patch_count_hists.append(patch_counts)

    L_txt = max(max_events_per_patch, 1)
    text_data = torch.zeros(B, M, L_txt, N_txt, d_txt, device=data_device)
    text_tp = torch.zeros(B, M, L_txt, N_txt, device=data_device)
    text_mask = torch.zeros(B, M, L_txt, N_txt, device=data_device)

    for b_idx, seq in enumerate(filtered_raws):
        for i in range(M):
            st = i * patch_stride
            ed = min(st + patch_size, cutoff)
            if i == M - 1:
                patch_events = [(t, e) for (t, e) in seq if (t >= st and t <= cutoff)]
            else:
                patch_events = [(t, e) for (t, e) in seq if (t >= st and t < ed)]
            patch_events = patch_events[:L_txt]
            for j, (t, emb) in enumerate(patch_events):
                if not torch.is_tensor(emb):
                    raise ValueError(
                        "Expected precomputed text embedding tensors for --text_guided_graph."
                    )
                text_data[b_idx, i, j, 0] = emb.to(device=data_device, dtype=torch.float32)
                text_tp[b_idx, i, j, 0] = float(t)
                text_mask[b_idx, i, j, 0] = 1.0

    time_max_tensor = torch.as_tensor(
        time_max, dtype=torch.float32, device=data_device
    )
    text_tp = utils.normalize_masked_tp(text_tp, att_min=0.0, att_max=time_max_tensor)

    return {
        "text_observed_data": text_data,
        "text_observed_tp": text_tp,
        "text_observed_mask": text_mask,
        "text_cutoff_time": cutoff,
        "n_text_vars": N_txt,
    }


#####################################################################################################
# Main Data Parsing Function
#####################################################################################################


def get_input_and_pred_len(data_obj):
    """
    Scans one full epoch of train/val/test data to find:
      - max_input_len  = largest observed_data.shape[1]
      - max_pred_len   = largest data_to_predict.shape[1]
    Returns (max_input_len, max_pred_len).
    """
    max_input_len = 0
    max_pred_len = 0

    splits = [
        ("train", data_obj["train_dataloader"]),
        ("val", data_obj["val_dataloader"]),
    ]
    if data_obj.get("test_dataloader") is not None:
        splits.append(("test", data_obj["test_dataloader"]))

    for name, dataloader in splits:
        print(f"Scanning {name} split ({len(dataloader)} batches)...")
        for batch in dataloader:
            T_obs = batch["observed_data"].shape[1]
            T_pred = batch["data_to_predict"].shape[1]
            if T_obs > max_input_len:
                max_input_len = T_obs
            if T_pred > max_pred_len:
                max_pred_len = T_pred

    return max_input_len, max_pred_len


def show_ds_summary(args, rec_ids: list[str] | None = None):
    base_root = (
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.data_root))
        if not os.path.isabs(args.data_root)
        else args.data_root
    )
    proc_dir = os.path.join(base_root, args.dataset, "processed")
    if rec_ids is None:
        data_glob = os.path.join(proc_dir, "*", "time_series.csv")
        paths = glob.glob(data_glob)
    else:
        paths = [
            os.path.join(proc_dir, rec, "time_series.csv")
            for rec in rec_ids
            if os.path.isfile(os.path.join(proc_dir, rec, "time_series.csv"))
        ]
    if not paths:
        print(f"No time_series.csv found under {proc_dir}")
        return
    num_entities = len(paths)
    first_df = pd.read_csv(paths[0])
    first_df["date_time"] = _parse_datetime(first_df["date_time"])
    feature_cols = [c for c in first_df.columns if c not in ["date_time", "record_id"]]
    num_features = len(feature_cols)

    total_obs = 0
    feat_counts = np.zeros(num_features, dtype=float)
    all_times = []
    all_dts = []
    all_text_times = []
    total_text = 0

    for p in paths:
        df = pd.read_csv(p)
        df["date_time"] = _parse_datetime(df["date_time"])
        mask = df[feature_cols].notna().to_numpy(dtype=int)
        total_obs += mask.sum()
        feat_counts += mask.sum(axis=0)
        times = df["date_time"].sort_values().to_numpy()
        all_times.append(times)
        dts = np.diff(times).astype("timedelta64[s]").astype(float)
        all_dts.append(dts)

        text_path = p.replace("time_series.csv", "text.csv")
        if os.path.isfile(text_path):
            tdf = pd.read_csv(text_path)
            tdf["date_time"] = _parse_datetime(tdf["date_time"])
            text_cols = [c for c in tdf.columns if c not in ("date_time", "record_id")]
            if len(text_cols) == 1:
                total_text += tdf[text_cols[0]].notna().sum()
                all_text_times.append(tdf["date_time"].dropna().to_numpy())

    all_times = np.concatenate(all_times)
    all_dts = np.concatenate(all_dts)
    num_unique_timestamps = len(np.unique(all_times))

    p_feat = feat_counts / total_obs
    feat_obs_entropy = -(p_feat * np.log(p_feat + 1e-12)).sum()
    H_feat_max = math.log(num_features)
    feat_obs_entropy_norm = feat_obs_entropy / H_feat_max

    K = 10
    t_min = all_times.min().astype("datetime64[s]").astype(float)
    t_max = all_times.max().astype("datetime64[s]").astype(float)
    bins = np.linspace(t_min, t_max, K + 1)
    counts, _ = np.histogram(all_times.astype("datetime64[s]").astype(float), bins=bins)
    p_time = counts / counts.sum()
    temp_obs_entropy = -(p_time * np.log(p_time + 1e-12)).sum()
    H_temp_max = math.log(K)
    temp_obs_entropy_norm = temp_obs_entropy / H_temp_max

    if total_text > 0 and len(all_text_times) > 0:
        all_text_times = np.concatenate(all_text_times)
        t_text_min = all_text_times.min().astype("datetime64[s]").astype(float)
        t_text_max = all_text_times.max().astype("datetime64[s]").astype(float)
        bins_text = np.linspace(t_text_min, t_text_max, K + 1)
        counts_text, _ = np.histogram(
            all_text_times.astype("datetime64[s]").astype(float), bins=bins_text
        )
        p_text_time = counts_text / counts_text.sum()
        temp_text_entropy = -(p_text_time * np.log(p_text_time + 1e-12)).sum()
        temp_text_entropy_norm = temp_text_entropy / H_temp_max
    else:
        temp_text_entropy_norm = None

    SEC_PER_UNIT = {
        "seconds": 1, "minutes": 60, "hours": 3600, "days": 86400, "weeks": 604800,
    }
    if args.time_unit == "custom":
        if getattr(args, "unit_scale", None) is None:
            raise ValueError("unit_scale must be set when time_unit='custom'")
        sec_per_unit = float(args.unit_scale)
    else:
        sec_per_unit = SEC_PER_UNIT[args.time_unit]
    mean_ioi = (all_dts / sec_per_unit).mean()

    start_str = pd.to_datetime(t_min, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    end_str = pd.to_datetime(t_max, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    timespan = f"{start_str}~{end_str}"

    summary = {
        "num_entities": num_entities,
        "num_features": num_features,
        "num_unique_timestamps": num_unique_timestamps,
        "num_observations": int(total_obs),
        "Feat observability entropy (norm)": round(feat_obs_entropy_norm, 4),
        "Temporal observation entropy (norm)": round(temp_obs_entropy_norm, 4),
        "Mean IOI": f"{round(mean_ioi, 4)} {args.time_unit}",
        "timespan": timespan,
        "num_text": int(total_text),
        "Text temporal entropy (norm)": (
            round(temp_text_entropy_norm, 4)
            if temp_text_entropy_norm is not None
            else "N/A"
        ),
    }

    table = PrettyTable(["Metric", "Value"])
    for metric, value in summary.items():
        table.add_row([metric, value])
    print(table)


def _make_multimodal_collate_fn(base_collate, args, time_max):
    """
    Wraps an existing collate function to add text modality data.
    """
    from torch.nn.utils.rnn import pad_sequence
    data_device = _get_data_device(args)
    time_max_tensor = torch.as_tensor(time_max, dtype=torch.float32, device=data_device)

    def multimodal_collate(batch):
        numeric_batch = [item[:4] for item in batch]
        out = base_collate(numeric_batch, args, time_max)
        out["n_numeric_vars"] = int(out["observed_data"].shape[-1])

        raws = [item[4] for item in batch]

        time_seqs = [
            torch.tensor(
                [t for (t, _) in seq], dtype=torch.float32, device=data_device
            )
            for seq in raws
        ]
        tau = pad_sequence(time_seqs, batch_first=True, padding_value=0.0)
        tau = utils.normalize_masked_tp(tau, att_min=0.0, att_max=time_max_tensor)
        out["tau"] = tau

        if args.enable_text and not args.use_text_embeddings:
            out["notes_text"] = [[txt for (_, txt) in seq] for seq in raws]

        if args.enable_text and args.use_text_embeddings:
            d_txt = None
            for seq in raws:
                if seq:
                    d_txt = seq[0][1].size(-1)
                    break
            if d_txt is None:
                emb_padded = torch.zeros((len(batch), 0, 0), device=data_device)
            else:
                emb_seqs = []
                for seq in raws:
                    if seq:
                        emb_seqs.append(torch.stack([e for (_, e) in seq], dim=0))
                    else:
                        emb_seqs.append(torch.zeros((0, d_txt), device=data_device))
                emb_padded = pad_sequence(
                    emb_seqs, batch_first=True, padding_value=0.0
                )
            out["notes_embeddings"] = emb_padded

        if args.enable_text and getattr(args, "text_guided_graph", False):
            numeric_tts = [item[1] for item in batch]
            text_batch = build_text_guided_graph_batch(
                raws, args, time_max, numeric_tts=numeric_tts,
            )
            out.update(text_batch)
            out["n_total_vars"] = int(out["n_numeric_vars"] + text_batch["n_text_vars"])
        else:
            out["n_text_vars"] = 0
            out["n_total_vars"] = int(out["n_numeric_vars"])

        # Classification labels (6th element)
        if len(batch[0]) > 5 and batch[0][5] is not None:
            labels = torch.stack([item[5] for item in batch], dim=0)
            out["labels"] = labels

        if len(batch[0]) > 6 and batch[0][6] is not None:
            risk_targets = torch.stack([item[6] for item in batch], dim=0)
            out["risk_targets"] = risk_targets

        # Per-sample text availability mask (for fallback logic)
        out["has_text"] = torch.tensor(
            [len(item[4]) > 0 for item in batch],
            dtype=torch.bool, device=data_device,
        )

        return out

    return multimodal_collate


def _make_patient_multimodal_collate_fn(args, time_max):
    """
    Wraps patient_level_collate_fn to add text modality data.
    """
    from torch.nn.utils.rnn import pad_sequence
    data_device = _get_data_device(args)
    time_max_tensor = torch.as_tensor(time_max, dtype=torch.float32, device=data_device)

    def collate(batch):
        out = patient_level_collate_fn(batch, args, time_max=time_max)
        out["n_numeric_vars"] = int(out["observed_data"].shape[-1])

        raws = [item[4] for item in batch]

        time_seqs = [
            torch.tensor(
                [t for (t, _) in seq], dtype=torch.float32, device=data_device
            )
            for seq in raws
        ]
        tau = pad_sequence(time_seqs, batch_first=True, padding_value=0.0)
        tau = utils.normalize_masked_tp(tau, att_min=0.0, att_max=time_max_tensor)
        out["tau"] = tau

        if args.enable_text and not args.use_text_embeddings:
            out["notes_text"] = [[txt for (_, txt) in seq] for seq in raws]

        if args.enable_text and args.use_text_embeddings:
            d_txt = None
            for seq in raws:
                if seq:
                    d_txt = seq[0][1].size(-1)
                    break
            if d_txt is None:
                emb_padded = torch.zeros((len(batch), 0, 0), device=data_device)
            else:
                emb_seqs = []
                for seq in raws:
                    if seq:
                        emb_seqs.append(torch.stack([e for (_, e) in seq], dim=0))
                    else:
                        emb_seqs.append(torch.zeros((0, d_txt), device=data_device))
                emb_padded = pad_sequence(
                    emb_seqs, batch_first=True, padding_value=0.0
                )
            out["notes_embeddings"] = emb_padded

        if args.enable_text and getattr(args, "text_guided_graph", False):
            text_batch = build_text_guided_graph_batch(
                raws,
                args,
                time_max=time_max,
                cutoff=time_max,
                patch_size=args.patch_size,
                patch_stride=args.patch_stride,
            )
            out.update(text_batch)
            out["n_text_vars"] = int(text_batch["n_text_vars"])
            out["n_total_vars"] = int(out["n_numeric_vars"] + out["n_text_vars"])
        else:
            out["n_text_vars"] = 0
            out["n_total_vars"] = int(out["n_numeric_vars"])

        # Per-sample text availability mask
        out["has_text"] = torch.tensor(
            [len(item[4]) > 0 for item in batch],
            dtype=torch.bool, device=data_device,
        )

        return out

    return collate


def parse_datasets(args, show_summary=True):
    """
    Load and split time-series data.
    Supports both window-level (chunked) and patient-level tasks.
    """
    base = (
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.data_root))
        if not os.path.isabs(args.data_root)
        else args.data_root
    )
    dataset_path = os.path.join(base, args.dataset)
    print(f"Using dataset path: {dataset_path}")

    task = getattr(args, "task", "forecasting")
    task_name = getattr(args, "task_name", None)
    task_config = None
    if task_name:
        task_config = get_task_config(task_name)
        # Override task type based on config
        if task_config.task_type in ("window_cls", "patient_cls"):
            task = "classification"
        elif task_config.task_type == "patient_reg":
            task = "regression"
        args.task = task

    # ---- Step 1: Determine patient IDs and split ----
    proc_dir = os.path.join(dataset_path, "processed")
    data_device = _get_data_device(args)
    all_rec_ids = sorted(
        d for d in os.listdir(proc_dir) if os.path.isdir(os.path.join(proc_dir, d))
    )
    if getattr(args, "rec_ids", None) is not None:
        all_rec_ids = args.rec_ids
    all_rec_ids, invalid_time_recs = filter_records_with_subday_timestamps(
        proc_dir, all_rec_ids, verbose=show_summary,
    )
    if getattr(args, "n", None) is not None:
        all_rec_ids = all_rec_ids[: min(len(all_rec_ids), int(args.n))]

    if not all_rec_ids:
        raise RuntimeError(
            "No valid records remain after filtering patients without hour/min/sec "
            "timestamps in time_series.csv."
        )

    # For patient-level tasks, get labels for stratified split
    if task_config and task_config.task_type in ("patient_cls", "patient_reg"):
        labels_path = os.path.join(dataset_path, "labels.csv")
        labels_df = pd.read_csv(labels_path)
        labels_df["stay_id"] = labels_df["stay_id"].astype(str)
        label_col = task_config.label_col

        # Filter to patients with valid labels
        valid_recs = []
        valid_labels = []
        for rec in all_rec_ids:
            row = labels_df[labels_df["stay_id"] == rec]
            if not row.empty and not pd.isna(row[label_col].values[0]):
                valid_recs.append(rec)
                valid_labels.append(float(row[label_col].values[0]))

        is_cls = task_config.task_type == "patient_cls"
        train_recs, val_recs, test_recs = split_patients_stratified(
            valid_recs, valid_labels,
            train_ratio=0.6, val_ratio=0.2,
            random_state=args.seed, is_classification=is_cls,
        )
    elif task_config and task_config.task_type == "window_cls":
        split_recs, split_labels = get_window_task_patient_labels(
            proc_dir, all_rec_ids, task_config.label_col,
        )
        train_recs, val_recs, test_recs = split_patients_stratified(
            split_recs,
            split_labels,
            train_ratio=0.6,
            val_ratio=0.2,
            random_state=args.seed,
            is_classification=True,
        )
    else:
        # Window-level or forecasting: split by patient ID
        # For classification, get chunk-level labels for stratification later
        train_recs, temp_recs = train_test_split(
            all_rec_ids, train_size=0.6, random_state=args.seed, shuffle=True,
        )
        val_recs, test_recs = train_test_split(
            temp_recs, train_size=0.5, random_state=args.seed, shuffle=True,
        )

    print(f"Patient split: train={len(train_recs)}, val={len(val_recs)}, test={len(test_recs)}")

    # ---- Step 2: Compute global normalization stats from train set ----
    global_stats = compute_global_stats(
        dataset_path, all_rec_ids, set(train_recs), task_config=task_config,
    )

    # ---- Step 3: Build dataset ----
    if task_config and task_config.task_type in ("patient_cls", "patient_reg"):
        # Patient-level dataset
        ds = PatientLevelDataset(
            root=dataset_path,
            task_config=task_config,
            device=data_device,
            time_unit=args.time_unit,
            unit_scale=getattr(args, "unit_scale", None),
            normalize=True,
            enable_text=args.enable_text,
            use_text_embeddings=args.use_text_embeddings,
            llm_model_fusion=args.llm_model_fusion,
            llm_layers_fusion=args.llm_layers_fusion,
            max_length=args.max_length,
            args=args,
            global_stats=global_stats,
        )

        # Map rec_ids to indices
        rec_to_idx = {rec: i for i, rec in enumerate(ds.rec_ids)}
        train_idx = [rec_to_idx[r] for r in train_recs if r in rec_to_idx]
        val_idx = [rec_to_idx[r] for r in val_recs if r in rec_to_idx]
        test_idx = [rec_to_idx[r] for r in test_recs if r in rec_to_idx]

        input_dim = len(ds.feature_names)

        # Patch config for patient-level
        args.npatch = args.npatch or 10
        patient_time_max = max(float(getattr(ds, "max_seq_time", 0.0)), 1.0)
        if (
            args.enable_text
            and args.use_text_embeddings
            and getattr(args, "text_guided_graph", False)
        ):
            for _, _, _, _, texts, _ in ds.samples:
                if texts and torch.is_tensor(texts[0][1]):
                    args.d_txt = int(texts[0][1].size(-1))
                    break
        if args.patch_size is None:
            args.patch_size = patient_time_max / args.npatch
        if args.patch_stride is None:
            args.patch_stride = args.patch_size

        collate_fn = _make_patient_multimodal_collate_fn(args, patient_time_max)

    else:
        # Window-level (chunked) dataset
        ds = ChunkedTimeSeriesDataset(
            root=dataset_path,
            history=args.history,
            pred_window=args.pred_window,
            stride=args.stride,
            device=data_device,
            time_unit=args.time_unit,
            unit_scale=getattr(args, "unit_scale", None),
            normalize=True,
            enable_text=args.enable_text,
            use_text_embeddings=args.use_text_embeddings,
            llm_model_fusion=args.llm_model_fusion,
            llm_layers_fusion=args.llm_layers_fusion,
            max_length=args.max_length,
            args=args,
            task=task,
            task_config=task_config,
            global_stats=global_stats,
        )

        if show_summary:
            show_ds_summary(args, rec_ids=all_rec_ids)

        all_chunks = ds.chunks
        if not all_chunks:
            raise ValueError("No chunks available! Check history/pred_window/stride.")

        if (
            args.enable_text
            and args.use_text_embeddings
            and getattr(args, "text_guided_graph", False)
        ):
            for chunk in all_chunks:
                texts = chunk[4]
                if texts and torch.is_tensor(texts[0][1]):
                    args.d_txt = int(texts[0][1].size(-1))
                    break

        input_dim = all_chunks[0][2].size(-1)

        # Map chunks to train/val/test by patient ID
        train_set = set(train_recs)
        val_set = set(val_recs)
        test_set = set(test_recs)

        train_idx, val_idx, test_idx = [], [], []
        for i, (cid, *_) in enumerate(all_chunks):
            rec = cid.rsplit("_chunk", 1)[0]
            if rec in train_set:
                train_idx.append(i)
            elif rec in val_set:
                val_idx.append(i)
            elif rec in test_set:
                test_idx.append(i)

        if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
            available_recs = sorted({cid.rsplit("_chunk", 1)[0] for cid, *_ in all_chunks})
            print(
                "Warning: patient-level split produced an empty chunk split after window "
                "filtering; re-splitting only across patients that produced valid chunks."
            )
            train_recs, val_recs, test_recs = split_available_chunk_patients(
                proc_dir, available_recs, task_config=task_config, seed=args.seed,
            )
            train_set = set(train_recs)
            val_set = set(val_recs)
            test_set = set(test_recs)
            train_idx, val_idx, test_idx = [], [], []
            for i, (cid, *_) in enumerate(all_chunks):
                rec = cid.rsplit("_chunk", 1)[0]
                if rec in train_set:
                    train_idx.append(i)
                elif rec in val_set:
                    val_idx.append(i)
                elif rec in test_set:
                    test_idx.append(i)
            if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
                raise ValueError(
                    "Unable to build non-empty patient-level chunk splits after "
                    "re-splitting patients with valid chunks."
                )

        # Patch config
        if args.patch_size is None:
            args.patch_size = max(1, args.history // 5)
        args.npatch = args.npatch or 5
        args.patch_stride = args.patch_stride or args.patch_size
        if args.enable_text and getattr(args, "text_guided_graph", False):
            args.max_text_vars = int(getattr(args, "max_text_vars", 1))

        print(
            f"Using Patch Collate Fn: patch_size={args.patch_size}, "
            f"npatch={args.npatch}, patch_stride={args.patch_stride}"
        )

        tm = torch.tensor(
            args.history + args.pred_window, dtype=torch.float32, device=data_device
        )
        collate_fn = _make_multimodal_collate_fn(patch_variable_time_collate_fn, args, tm)

    print(
        f"After splitting: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )

    result = {"input_dim": input_dim, "ds": ds}

    # Classification metadata
    train_sampler = None
    if task == "classification":
        if task_config and task_config.task_type in ("patient_cls",):
            result["label_names"] = [task_config.label_col]
            result["n_labels"] = 1
        elif hasattr(ds, "label_names") and ds.label_names:
            result["label_names"] = ds.label_names
            result["n_labels"] = len(ds.label_names)
        else:
            result["n_labels"] = 1
            result["label_names"] = [task_config.label_col] if task_config else []
        print(f"Classification mode: {result['n_labels']} labels = {result.get('label_names', [])}")

    # Regression metadata
    if task == "regression":
        result["n_labels"] = 1
        result["label_names"] = [task_config.label_col]
        print(f"Regression mode: target = {task_config.label_col}")

    # Compute pos_weight for classification tasks
    if task == "classification":
        all_labels = []
        for idx in train_idx:
            sample = ds[idx]
            if sample[5] is not None:
                all_labels.append(sample[5])
        if all_labels:
            all_labels_t = torch.stack(all_labels, dim=0)
            n_pos = all_labels_t.sum(dim=0)
            n_neg = all_labels_t.shape[0] - n_pos
            pos_weight = torch.ones_like(n_pos, dtype=torch.float32)
            valid = (n_pos > 0) & (n_neg > 0)
            pos_weight[valid] = (n_neg[valid] / n_pos[valid]).to(torch.float32)
            result["pos_weight"] = pos_weight
            if (~valid).any():
                print(
                    "Warning: some training labels contain only one class; "
                    "using neutral pos_weight=1.0 for those labels."
                )
            print(f"pos_weight: {pos_weight.tolist()}")

            if (
                task_config is not None
                and task_config.use_weighted_sampler
                and all_labels_t.ndim == 2
                and all_labels_t.shape[1] == 1
                and bool(valid.all())
            ):
                sample_weights = torch.where(
                    all_labels_t.squeeze(1) > 0.5,
                    torch.full(
                        (all_labels_t.shape[0],),
                        float(pos_weight[0].item()),
                        dtype=torch.float32,
                    ),
                    torch.ones(all_labels_t.shape[0], dtype=torch.float32),
                )
                sampler_generator = torch.Generator()
                sampler_generator.manual_seed(int(getattr(args, "seed", 1)))
                train_sampler = WeightedRandomSampler(
                    weights=sample_weights.to(torch.double),
                    num_samples=len(sample_weights),
                    replacement=True,
                    generator=sampler_generator,
                )
                positive_rate = float((all_labels_t.squeeze(1) > 0.5).float().mean().item())
                print(
                    "Using WeightedRandomSampler for imbalanced classification "
                    f"(train positive rate={positive_rate:.4f})."
                )

    # Build DataLoaders
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    test_ds = Subset(ds, test_idx) if test_idx else None

    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate_fn,
        "num_workers": getattr(args, "num_workers", 0),
        "pin_memory": bool(getattr(args, "pin_memory", False)),
    }
    if dataloader_kwargs["num_workers"] > 0:
        dataloader_kwargs["persistent_workers"] = bool(
            getattr(args, "persistent_workers", False)
        )

    train_loader = DataLoader(
        train_ds,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **dataloader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, shuffle=False, **dataloader_kwargs
    )
    test_loader = (
        DataLoader(
            test_ds, shuffle=False, **dataloader_kwargs
        )
        if test_ds
        else None
    )

    result["train_dataloader"] = train_loader
    result["val_dataloader"] = val_loader
    result["test_dataloader"] = test_loader
    return result
