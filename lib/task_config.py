"""
Task configuration for multi-task MIMIC-Sepsis experiments.

Defines 7 downstream tasks (T1-T7) with per-task feature exclusion,
label source, loss type, and evaluation metrics.

Window-level tasks (T1-T4) use **onset prediction**: only windows where
the event has NOT yet occurred are kept, and the label indicates whether
the event will occur for the first time within the prediction window.
"""

from dataclasses import dataclass, field


@dataclass
class TaskConfig:
    name: str
    task_type: str  # "window_cls" | "patient_cls" | "patient_reg"
    label_col: str  # column name in time_series.csv or labels.csv
    label_source: str  # "timeseries" | "labels_csv"
    exclude_features: list[str] = field(default_factory=list)
    loss: str = "bce"  # "bce" | "huber"
    metrics: list[str] = field(default_factory=list)
    onset_only: bool = False  # True = first-occurrence prediction
    early_stop_metric: str | None = None
    tune_decision_threshold: bool = False
    use_weighted_sampler: bool = False
    # window params (only for window_cls); None means use args defaults
    default_history: int | None = None
    default_pred_window: int | None = None
    default_stride: int | None = None


# All tasks auto-exclude these columns from features
ALWAYS_EXCLUDE = {
    "date_time", "record_id", "_ts_raw",
    "mechvent",  # identical to __label__mechvent
}

# Label columns are always excluded from features
LABEL_PREFIX = "__label__"


TASK_CONFIGS: dict[str, TaskConfig] = {
    # ---- T1-T4: Window-level classification (onset prediction) ----
    "sepsis": TaskConfig(
        name="sepsis",
        task_type="window_cls",
        label_col="__label__sepsis",
        label_source="timeseries",
        exclude_features=["sofa_score", "sirs_score"],
        loss="bce",
        metrics=["auroc", "auprc", "f1"],
        onset_only=True,
        default_history=6,
        default_pred_window=24,
        default_stride=6,
    ),
    "septic_shock": TaskConfig(
        name="septic_shock",
        task_type="window_cls",
        label_col="__label__septic_shock",
        label_source="timeseries",
        exclude_features=["shock_index"],
        loss="bce",
        metrics=["auroc", "auprc", "f1"],
        onset_only=True,
        early_stop_metric="auprc",
        tune_decision_threshold=True,
        use_weighted_sampler=True,
        default_history=6,
        default_pred_window=24,
        default_stride=6,
    ),
    "mechvent": TaskConfig(
        name="mechvent",
        task_type="window_cls",
        label_col="__label__mechvent",
        label_source="timeseries",
        exclude_features=[],  # mechvent already in ALWAYS_EXCLUDE
        loss="bce",
        metrics=["auroc", "auprc", "f1"],
        onset_only=True,
        early_stop_metric="auprc",
        tune_decision_threshold=True,
        use_weighted_sampler=True,
        default_history=6,
        default_pred_window=24,
        default_stride=6,
    ),
    "vasopressor": TaskConfig(
        name="vasopressor",
        task_type="window_cls",
        label_col="__label__vasopressor",
        label_source="timeseries",
        exclude_features=["sofa_score"],
        loss="bce",
        metrics=["auroc", "auprc", "f1"],
        onset_only=True,
        early_stop_metric="auprc",
        tune_decision_threshold=True,
        use_weighted_sampler=True,
        default_history=6,
        default_pred_window=24,
        default_stride=6,
    ),
    # ---- T5-T6: Patient-level classification ----
    "morta_hosp": TaskConfig(
        name="morta_hosp",
        task_type="patient_cls",
        label_col="morta_hosp",
        label_source="labels_csv",
        exclude_features=[],
        loss="bce",
        metrics=["auroc", "auprc", "f1"],
    ),
    "morta_90": TaskConfig(
        name="morta_90",
        task_type="patient_cls",
        label_col="morta_90",
        label_source="labels_csv",
        exclude_features=[],
        loss="bce",
        metrics=["auroc", "auprc", "f1"],
    ),
    # ---- T7: Patient-level regression ----
    "los": TaskConfig(
        name="los",
        task_type="patient_reg",
        label_col="los",
        label_source="labels_csv",
        exclude_features=[],
        loss="huber",
        metrics=["mae", "rmse", "r2"],
    ),
}


def get_task_config(task_name: str) -> TaskConfig:
    if task_name not in TASK_CONFIGS:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {list(TASK_CONFIGS.keys())}"
        )
    return TASK_CONFIGS[task_name]


def get_feature_exclude_set(task_name: str) -> set[str]:
    """Return the full set of columns to exclude from features for a given task."""
    cfg = get_task_config(task_name)
    exclude = set(ALWAYS_EXCLUDE)
    exclude.update(cfg.exclude_features)
    return exclude
