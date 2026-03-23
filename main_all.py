"""
Paper-oriented batch runner for all 7 MIMIC-Sepsis tasks.

Default behavior:
    - runs 7 tasks x 3 modalities = 21 experiments
    - schedules experiments sequentially on one GPU, or in parallel across multiple GPUs
    - uses a sensible full-run profile for MIMIC_sepsis-full
    - writes per-experiment logs and continuously updated progress files

Examples:
    python main_all.py
    python main_all.py --gpu 0
    python main_all.py --gpu 0,1
    python main_all.py --tasks sepsis los
    python main_all.py --modalities num text
    python main_all.py --run_profile smoke -n 128 --tasks los --modalities text
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from prettytable import PrettyTable

from lib.task_config import TASK_CONFIGS
from main import finalize_args, get_args_from_parser


TASK_LIST = list(TASK_CONFIGS.keys())
MODEL_LIST = ["tPatchGNN", "Linear", "LSTM", "Transformer"]


def _extend_if_set(argv, flag, value):
    if value is None:
        return
    argv.extend([flag, str(value)])


def _flag_present(argv_tokens, *flags):
    return any(flag in argv_tokens for flag in flags)


def _resolve_data_root(data_root: str) -> str:
    return (
        os.path.abspath(os.path.join(os.path.dirname(__file__), data_root))
        if not os.path.isabs(data_root)
        else data_root
    )


def _resolve_dataset_path(data_root: str, dataset: str) -> str:
    return os.path.join(_resolve_data_root(data_root), dataset)


def _count_available_records(data_root: str, dataset: str) -> int:
    dataset_path = _resolve_dataset_path(data_root, dataset)
    proc_dir = os.path.join(dataset_path, "processed")
    if not os.path.isdir(proc_dir):
        raise FileNotFoundError(f"Processed dataset directory not found: {proc_dir}")
    return sum(
        1 for name in os.listdir(proc_dir)
        if os.path.isdir(os.path.join(proc_dir, name))
    )


def _default_state_for_profile(profile: str) -> str:
    if profile == "smoke":
        return "smoke_main_all"
    return "paper_full_h6_p6_s1"


def _default_save_for_profile(profile: str) -> str:
    if profile == "smoke":
        return os.path.join("experiments", "main_all_smoke")
    return os.path.join("experiments", "paper_full_h6p6s1")


def _default_log_root_for_profile(profile: str) -> str:
    if profile == "smoke":
        return os.path.join("logs", "main_all_smoke")
    return os.path.join("logs", "main_all_runs")


def _apply_runner_profile(base_args, raw_argv):
    profile = base_args.run_profile or "paper_full"
    base_args.run_profile = profile

    if profile == "paper_full":
        batch_size_explicit = _flag_present(raw_argv, "-b", "--batch_size")
        if not _flag_present(raw_argv, "--dataset") and base_args.dataset == "MIMIC":
            base_args.dataset = "MIMIC_sepsis-full"
        if not _flag_present(raw_argv, "--save") and base_args.save == "experiments/":
            base_args.save = _default_save_for_profile(profile)
        if not _flag_present(raw_argv, "--state") and base_args.state == "def":
            base_args.state = _default_state_for_profile(profile)
        if not _flag_present(raw_argv, "--log_root") and base_args.log_root == "logs/main_all":
            base_args.log_root = _default_log_root_for_profile(profile)
        if not _flag_present(raw_argv, "--epoch") and base_args.epoch == 1000:
            base_args.epoch = 50
        if not _flag_present(raw_argv, "--patience") and base_args.patience == 3:
            base_args.patience = 8
        if not _flag_present(raw_argv, "--num_workers") and base_args.num_workers == 0:
            base_args.num_workers = 0 if os.name == "nt" else 4
        if not _flag_present(raw_argv, "--history") and base_args.history is None:
            base_args.history = 6
        if not _flag_present(raw_argv, "--pred_window") and base_args.pred_window is None:
            base_args.pred_window = 6
        if not _flag_present(raw_argv, "--stride") and base_args.stride is None:
            base_args.stride = 1
        if (
            not batch_size_explicit
            and not _flag_present(raw_argv, "--batch_size_num")
            and base_args.batch_size_num is None
        ):
            base_args.batch_size_num = 32
        if (
            not batch_size_explicit
            and not _flag_present(raw_argv, "--batch_size_text")
            and base_args.batch_size_text is None
        ):
            base_args.batch_size_text = 32
        if (
            not batch_size_explicit
            and
            not _flag_present(raw_argv, "--batch_size_text_guided_graph")
            and base_args.batch_size_text_guided_graph is None
        ):
            base_args.batch_size_text_guided_graph = 32
        if (
            not _flag_present(raw_argv, "--pin_memory")
            and base_args.device.type == "cuda"
        ):
            base_args.pin_memory = True
        if (
            not _flag_present(raw_argv, "--persistent_workers")
            and base_args.num_workers > 0
        ):
            base_args.persistent_workers = True
        if not _flag_present(raw_argv, "--use_amp") and base_args.device.type == "cuda":
            base_args.use_amp = True

    elif profile == "smoke":
        batch_size_explicit = _flag_present(raw_argv, "-b", "--batch_size")
        if not _flag_present(raw_argv, "--save") and base_args.save == "experiments/":
            base_args.save = _default_save_for_profile(profile)
        if not _flag_present(raw_argv, "--state") and base_args.state == "def":
            base_args.state = _default_state_for_profile(profile)
        if not _flag_present(raw_argv, "--log_root") and base_args.log_root == "logs/main_all":
            base_args.log_root = _default_log_root_for_profile(profile)
        if not _flag_present(raw_argv, "--epoch") and base_args.epoch == 1000:
            base_args.epoch = 1
        if not _flag_present(raw_argv, "--patience") and base_args.patience == 3:
            base_args.patience = 1
        if not _flag_present(raw_argv, "--num_workers") and base_args.num_workers == 0:
            base_args.num_workers = 0
        if (
            not batch_size_explicit
            and not _flag_present(raw_argv, "--batch_size_num")
            and base_args.batch_size_num is None
        ):
            base_args.batch_size_num = 8
        if (
            not batch_size_explicit
            and not _flag_present(raw_argv, "--batch_size_text")
            and base_args.batch_size_text is None
        ):
            base_args.batch_size_text = 8
        if (
            not batch_size_explicit
            and
            not _flag_present(raw_argv, "--batch_size_text_guided_graph")
            and base_args.batch_size_text_guided_graph is None
        ):
            base_args.batch_size_text_guided_graph = 8

    return finalize_args(base_args)


def _resolve_seeds(base_args):
    if base_args.seeds:
        return list(base_args.seeds)
    return [base_args.seed]


def _resolve_batch_size(base_args, mode_name: str) -> int:
    if mode_name == "num" and base_args.batch_size_num is not None:
        return int(base_args.batch_size_num)
    if mode_name == "text" and base_args.batch_size_text is not None:
        return int(base_args.batch_size_text)
    if (
        mode_name == "text_guided_graph"
        and base_args.batch_size_text_guided_graph is not None
    ):
        return int(base_args.batch_size_text_guided_graph)
    return int(base_args.batch_size)


def _parse_gpu_slots(gpu_arg: str):
    tokens = []
    raw_value = "" if gpu_arg is None else str(gpu_arg)
    for chunk in raw_value.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        tokens.extend(part for part in chunk.split() if part)

    unique_tokens = []
    for token in tokens:
        if token not in unique_tokens:
            unique_tokens.append(token)
    return unique_tokens


def _resolve_worker_slots(base_args):
    if base_args.device.type != "cuda":
        return [{"slot_id": "cpu", "gpu_arg": None, "launch_gpu_arg": str(base_args.gpu), "label": "CPU"}]

    gpu_slots = _parse_gpu_slots(base_args.gpu) or [str(base_args.gpu)]
    return [
        {"slot_id": gpu_id, "gpu_arg": gpu_id, "launch_gpu_arg": gpu_id, "label": f"GPU {gpu_id}"}
        for gpu_id in gpu_slots
    ]


def _build_experiment_argv(
    base_args,
    task_name: str,
    model_name: str,
    mode,
    seed: int,
    batch_size: int,
    assigned_gpu: str,
):
    argv_override = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("main.py")),
        "--dataset", base_args.dataset,
        "--data_root", base_args.data_root,
        "--gpu", assigned_gpu,
        "--seed", str(seed),
        "--model", model_name,
        "--task_name", task_name,
        "--batch_size", str(batch_size),
        "--epoch", str(base_args.epoch),
        "--patience", str(base_args.patience),
        "--lr", str(base_args.lr),
        "--hid_dim", str(base_args.hid_dim),
        "--save", str(base_args.save),
        "--state", str(base_args.state),
        "--time_unit", base_args.time_unit,
        "--outlayer", base_args.outlayer,
        "--te_dim", str(base_args.te_dim),
        "--node_dim", str(base_args.node_dim),
        "--hop", str(base_args.hop),
        "--tf_layer", str(base_args.tf_layer),
        "--nlayer", str(base_args.nlayer),
        "--n_heads", str(base_args.n_heads),
        "--early_stop_delta", str(base_args.early_stop_delta),
        "--w_decay", str(base_args.w_decay),
        "--dropout", str(base_args.dropout),
        "--hazard_loss_weight", str(base_args.hazard_loss_weight),
        "--aux_bce_weight", str(base_args.aux_bce_weight),
        "--num_workers", str(base_args.num_workers),
        "--logmode", str(base_args.logmode),
    ]

    for flag, value in [
        ("--unit_scale", base_args.unit_scale),
        ("--history", base_args.history),
        ("--pred_window", base_args.pred_window),
        ("--stride", base_args.stride),
        ("--patch_size", base_args.patch_size),
        ("--npatch", base_args.npatch),
        ("--patch_stride", base_args.patch_stride),
        ("--load", base_args.load),
        ("-n", base_args.n),
    ]:
        _extend_if_set(argv_override, flag, value)

    if base_args.rec_ids:
        argv_override.extend(["--rec_ids", *base_args.rec_ids])
    if base_args.use_amp:
        argv_override.append("--use_amp")
    if base_args.pin_memory:
        argv_override.append("--pin_memory")
    if base_args.persistent_workers:
        argv_override.append("--persistent_workers")
    if base_args.show_dataset_summary:
        argv_override.append("--show_dataset_summary")
    if base_args.show_record_chunk_counts:
        argv_override.append("--show_record_chunk_counts")

    if mode["enable_text"]:
        argv_override.append("--enable_text")
        argv_override.append("--use_text_embeddings")
        if mode.get("text_guided_graph", False):
            argv_override.append("--text_guided_graph")
        argv_override.extend(["--TTF_module", base_args.TTF_module])
        argv_override.extend(["--MMF_module", base_args.MMF_module])
        argv_override.extend(["--llm_model_fusion", base_args.llm_model_fusion])
        if base_args.llm_layers_fusion is not None:
            argv_override.extend(["--llm_layers_fusion", str(base_args.llm_layers_fusion)])
        if base_args.max_length is not None:
            argv_override.extend(["--max_length", str(base_args.max_length)])
        if base_args.d_txt is not None:
            argv_override.extend(["--d_txt", str(base_args.d_txt)])
        if base_args.max_text_vars is not None:
            argv_override.extend(["--max_text_vars", str(base_args.max_text_vars)])
        if base_args.recency_sigma is not None:
            argv_override.extend(["--recency_sigma", str(base_args.recency_sigma)])
        if base_args.n_heads_fusion is not None:
            argv_override.extend(["--n_heads_fusion", str(base_args.n_heads_fusion)])
        if base_args.kappa is not None:
            argv_override.extend(["--kappa", str(base_args.kappa)])
        if base_args.dbg_text_graph:
            argv_override.append("--dbg_text_graph")

    return argv_override


def _make_run_paths(base_args):
    run_tag = f"{base_args.dataset}_{base_args.state}"
    logs_dir = Path(base_args.log_root) / run_tag
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("experiment_results") / run_tag
    results_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir, results_dir


def _make_exp_dir(base_args, task_name: str, mode_tag: str, seed: int) -> str:
    exp_name = f"{base_args.dataset}_{task_name}_{mode_tag}_s{seed}_{base_args.state}"
    return os.path.join(base_args.save, exp_name)


def _build_run_signature(base_args, task_name, mode_name, seed, batch_size, available_records):
    return {
        "run_profile": base_args.run_profile,
        "dataset": base_args.dataset,
        "data_root": base_args.data_root,
        "available_records": available_records,
        "task_name": task_name,
        "mode": mode_name,
        "seed": seed,
        "state": base_args.state,
        "save": str(base_args.save),
        "n": base_args.n,
        "rec_ids": list(base_args.rec_ids) if base_args.rec_ids else None,
        "history": base_args.history,
        "pred_window": base_args.pred_window,
        "stride": base_args.stride,
        "time_unit": base_args.time_unit,
        "unit_scale": base_args.unit_scale,
        "batch_size": batch_size,
        "epoch": base_args.epoch,
        "patience": base_args.patience,
        "early_stop_delta": base_args.early_stop_delta,
        "lr": base_args.lr,
        "w_decay": base_args.w_decay,
        "dropout": base_args.dropout,
        "hazard_loss_weight": base_args.hazard_loss_weight,
        "aux_bce_weight": base_args.aux_bce_weight,
        "num_workers": base_args.num_workers,
        "pin_memory": bool(base_args.pin_memory),
        "persistent_workers": bool(base_args.persistent_workers),
        "show_dataset_summary": bool(base_args.show_dataset_summary),
        "show_record_chunk_counts": bool(base_args.show_record_chunk_counts),
        "outlayer": base_args.outlayer,
        "patch_size": base_args.patch_size,
        "npatch": base_args.npatch,
        "patch_stride": base_args.patch_stride,
        "hid_dim": base_args.hid_dim,
        "te_dim": base_args.te_dim,
        "node_dim": base_args.node_dim,
        "hop": base_args.hop,
        "tf_layer": base_args.tf_layer,
        "nlayer": base_args.nlayer,
        "n_heads": base_args.n_heads,
        "use_amp": bool(base_args.use_amp),
        "ttf_module": base_args.TTF_module,
        "mmf_module": base_args.MMF_module,
        "llm_model_fusion": base_args.llm_model_fusion,
        "llm_layers_fusion": base_args.llm_layers_fusion,
        "max_length": base_args.max_length,
        "d_txt": base_args.d_txt,
        "max_text_vars": base_args.max_text_vars,
        "recency_sigma": base_args.recency_sigma,
        "n_heads_fusion": base_args.n_heads_fusion,
        "kappa": base_args.kappa,
        "dbg_text_graph": bool(base_args.dbg_text_graph),
        "load": base_args.load,
    }


def _load_existing_signature(result_file: Path):
    if not result_file.exists():
        return None
    try:
        with result_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return payload.get("signature")


def _write_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _render_summary_table(run_records):
    table = PrettyTable()
    table.field_names = ["Experiment", "Model", "Batch", "Time (s)", "Status"]
    for record in run_records:
        table.add_row([
            record["desc"],
            record.get("model", "tPatchGNN"),
            record["batch_size"],
            f"{record['elapsed']:.1f}",
            record["status"],
        ])
    return table


def _estimate_remaining(run_records, remaining_count):
    finished = [r["elapsed"] for r in run_records if r["status"] not in {"SKIP"}]
    if not finished or remaining_count <= 0:
        return None
    return sum(finished) / len(finished) * remaining_count


def _sorted_run_records(run_records):
    return sorted(run_records, key=lambda record: record.get("order", 0))


def _write_progress_artifacts(progress_path: Path, summary_path: Path, run_records, total_count: int):
    ordered_records = _sorted_run_records(run_records)
    remaining_eta = _estimate_remaining(ordered_records, total_count - len(ordered_records))
    payload = {
        "completed": len(ordered_records),
        "total": total_count,
        "eta_seconds": remaining_eta,
        "runs": ordered_records,
    }
    _write_json(progress_path, payload)
    _write_json(summary_path, payload)
    return remaining_eta, ordered_records


def run_all():
    raw_argv = list(sys.argv[1:])
    base_args = get_args_from_parser()
    base_args = _apply_runner_profile(base_args, raw_argv)

    tasks_to_run = getattr(base_args, "tasks", None) or TASK_LIST
    invalid_tasks = [task for task in tasks_to_run if task not in TASK_LIST]
    if invalid_tasks:
        raise ValueError(f"Unknown tasks requested: {invalid_tasks}. Available: {TASK_LIST}")

    models_to_run = getattr(base_args, "models", None) or [
        "tPatchGNN", "Linear", "LSTM", "Transformer"
    ]
    invalid_models = [m for m in models_to_run if m not in MODEL_LIST]
    if invalid_models:
        raise ValueError(f"Unknown models requested: {invalid_models}. Available: {MODEL_LIST}")

    seeds = _resolve_seeds(base_args)

    modality_map = {
        "num": {"enable_text": False, "text_guided_graph": False, "tag": "num"},
        "text": {"enable_text": True, "text_guided_graph": False, "tag": "text"},
        "text_guided_graph": {
            "enable_text": True,
            "text_guided_graph": True,
            "tag": "text_guided_graph",
        },
    }
    requested_modalities = getattr(base_args, "modalities", None) or [
        "num", "text", "text_guided_graph"
    ]
    text_modes = [modality_map[name] for name in requested_modalities]

    # Build combos: task x model x modality x seed
    # Baseline models don't support text_guided_graph
    combos = []
    for task in tasks_to_run:
        for model_name in models_to_run:
            for mode in text_modes:
                if mode["tag"] == "text_guided_graph" and model_name != "tPatchGNN":
                    continue  # skip unsupported combo
                for seed in seeds:
                    combos.append((task, model_name, mode, seed))

    available_records = _count_available_records(base_args.data_root, base_args.dataset)
    effective_n = available_records if base_args.n is None else min(base_args.n, available_records)

    if base_args.device.type != "cuda":
        print("Warning: running the batch runner on CPU will be very slow.")
    worker_slots = _resolve_worker_slots(base_args)

    logs_dir, results_dir = _make_run_paths(base_args)
    manifest_path = logs_dir / "run_manifest.json"
    progress_path = logs_dir / "run_progress.json"
    summary_path = logs_dir / "summary.json"
    summary_table_path = logs_dir / "summary.txt"

    manifest = {
        "profile": base_args.run_profile,
        "dataset": base_args.dataset,
        "dataset_path": _resolve_dataset_path(base_args.data_root, base_args.dataset),
        "available_records": available_records,
        "effective_n": effective_n,
        "device": str(base_args.device),
        "gpu": base_args.gpu,
        "gpu_pool": [slot["gpu_arg"] for slot in worker_slots if slot["gpu_arg"]],
        "max_parallel_runs": len(worker_slots),
        "tasks": tasks_to_run,
        "models": models_to_run,
        "modalities": requested_modalities,
        "seeds": seeds,
        "save_root": base_args.save,
        "logs_dir": str(logs_dir),
        "results_dir": str(results_dir),
        "state": base_args.state,
        "epoch": base_args.epoch,
        "patience": base_args.patience,
        "use_amp": bool(base_args.use_amp),
        "num_workers": base_args.num_workers,
        "pin_memory": bool(base_args.pin_memory),
        "persistent_workers": bool(base_args.persistent_workers),
        "batch_size_num": _resolve_batch_size(base_args, "num"),
        "batch_size_text": _resolve_batch_size(base_args, "text"),
        "batch_size_text_guided_graph": _resolve_batch_size(base_args, "text_guided_graph"),
        "window_task_defaults": {
            task_name: {
                "history": (
                    base_args.history
                    if base_args.history is not None
                    else TASK_CONFIGS[task_name].default_history
                ),
                "pred_window": (
                    base_args.pred_window
                    if base_args.pred_window is not None
                    else TASK_CONFIGS[task_name].default_pred_window
                ),
                "stride": (
                    base_args.stride
                    if base_args.stride is not None
                    else TASK_CONFIGS[task_name].default_stride
                ),
            }
            for task_name in tasks_to_run
            if task_name in TASK_CONFIGS and TASK_CONFIGS[task_name].task_type == "window_cls"
        },
        "combos": [
            {
                "task": task_name,
                "model": model_name,
                "mode": mode["tag"],
                "seed": seed,
                "batch_size": _resolve_batch_size(base_args, mode["tag"]),
                "exp_dir": _make_exp_dir(base_args, task_name, mode["tag"], seed),
                "log_file": str(logs_dir / f"{task_name}_{model_name}_{mode['tag']}_s{seed}.log"),
            }
            for task_name, model_name, mode, seed in combos
        ],
    }
    _write_json(manifest_path, manifest)

    print("===== main_all configuration =====")
    print(f"profile: {base_args.run_profile}")
    print(f"dataset: {base_args.dataset} ({effective_n}/{available_records} records)")
    print(f"workers: {len(worker_slots)} ({', '.join(slot['label'] for slot in worker_slots)})")
    print(f"tasks: {tasks_to_run}")
    print(f"models: {models_to_run}")
    print(f"modalities: {requested_modalities}")
    print(f"seeds: {seeds}")
    print(
        "batch sizes: "
        f"num={_resolve_batch_size(base_args, 'num')}, "
        f"text={_resolve_batch_size(base_args, 'text')}, "
        f"text_guided_graph={_resolve_batch_size(base_args, 'text_guided_graph')}"
    )
    print(
        "chunk window config: "
        f"history={base_args.history}, pred_window={base_args.pred_window}, stride={base_args.stride}"
    )
    print(f"epoch={base_args.epoch}, patience={base_args.patience}, use_amp={base_args.use_amp}")
    print(
        "dataloader: "
        f"num_workers={base_args.num_workers}, "
        f"pin_memory={base_args.pin_memory}, "
        f"persistent_workers={base_args.persistent_workers}"
    )
    print(f"save_root: {base_args.save}")
    print(f"logs_dir: {logs_dir}")
    print(f"results_dir: {results_dir}")
    print(f"Total experiments: {len(combos)}")
    print()

    run_records = []
    pending_jobs = []
    total_start = time.time()
    total_runs = len(combos)

    for idx, (task_name, model_name, mode, seed) in enumerate(combos):
        mode_tag = mode["tag"]
        batch_size = _resolve_batch_size(base_args, mode_tag)
        desc = f"{task_name} | {model_name} | {mode_tag} | seed={seed}"
        result_file = results_dir / f"{task_name}_{model_name}_{mode_tag}_s{seed}.json"
        log_file = logs_dir / f"{task_name}_{model_name}_{mode_tag}_s{seed}.log"
        exp_dir = _make_exp_dir(base_args, task_name, mode_tag, seed)
        signature = _build_run_signature(
            base_args, task_name, mode_tag, seed, batch_size, available_records
        )
        signature["model"] = model_name

        if _load_existing_signature(result_file) == signature:
            print(f"[{idx+1}/{len(combos)}] SKIP (complete): {desc}")
            run_records.append(
                {
                    "task": task_name,
                    "model": model_name,
                    "mode": mode_tag,
                    "seed": seed,
                    "order": idx,
                    "desc": desc,
                    "batch_size": batch_size,
                    "elapsed": 0.0,
                    "status": "SKIP",
                    "gpu": None,
                    "log_file": str(log_file),
                    "exp_dir": exp_dir,
                }
            )
            _write_progress_artifacts(progress_path, summary_path, run_records, total_runs)
            continue

        pending_jobs.append(
            {
                "task": task_name,
                "model": model_name,
                "mode": mode,
                "mode_tag": mode_tag,
                "seed": seed,
                "order": idx,
                "desc": desc,
                "batch_size": batch_size,
                "result_file": result_file,
                "log_file": log_file,
                "exp_dir": exp_dir,
                "signature": signature,
            }
        )

    active_jobs = {}
    launched_count = 0
    project_root = str(Path(__file__).resolve().parent)

    try:
        while pending_jobs or active_jobs:
            free_slots = [slot for slot in worker_slots if slot["slot_id"] not in active_jobs]

            while free_slots and pending_jobs:
                slot = free_slots.pop(0)
                job = pending_jobs.pop(0)
                launched_count += 1
                argv_override = _build_experiment_argv(
                    base_args,
                    job["task"],
                    job["model"],
                    job["mode"],
                    job["seed"],
                    job["batch_size"],
                    slot["launch_gpu_arg"],
                )
                print(f"\n[start {launched_count}/{total_runs}] {job['desc']} on {slot['label']}")
                print(f"  log: {job['log_file']}")
                print(f"  exp: {job['exp_dir']}")

                log_handle = None
                try:
                    log_handle = job["log_file"].open("w", encoding="utf-8")
                    print("===== Experiment configuration =====", file=log_handle)
                    print(json.dumps(job["signature"], ensure_ascii=False, indent=2), file=log_handle)
                    print(f"assigned_worker: {slot['label']}", file=log_handle)
                    print("===== Begin main.py output =====", file=log_handle)
                    log_handle.flush()
                    process = subprocess.Popen(
                        argv_override,
                        cwd=project_root,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
                except Exception as exc:
                    if log_handle is not None and not log_handle.closed:
                        print(f"Launch failed: {exc}", file=log_handle)
                        log_handle.close()
                    print(f"  ERROR: failed to launch process on {slot['label']}: {exc}")
                    run_records.append(
                        {
                            "task": job["task"],
                            "model": job["model"],
                            "mode": job["mode_tag"],
                            "seed": job["seed"],
                            "order": job["order"],
                            "desc": job["desc"],
                            "batch_size": job["batch_size"],
                            "elapsed": 0.0,
                            "status": f"FAIL: {exc}",
                            "error": str(exc),
                            "gpu": slot["gpu_arg"] if base_args.device.type == "cuda" else None,
                            "log_file": str(job["log_file"]),
                            "exp_dir": job["exp_dir"],
                        }
                    )
                    _write_progress_artifacts(progress_path, summary_path, run_records, total_runs)
                    continue

                active_jobs[slot["slot_id"]] = {
                    **job,
                    "worker": slot,
                    "process": process,
                    "log_handle": log_handle,
                    "start_time": time.time(),
                }

            if not active_jobs:
                continue

            time.sleep(1.0)

            for slot_id, job in list(active_jobs.items()):
                returncode = job["process"].poll()
                if returncode is None:
                    continue

                log_handle = job["log_handle"]
                print("===== End main.py output =====", file=log_handle)
                print(f"process_exit_code: {returncode}", file=log_handle)
                log_handle.close()

                status = "OK" if returncode == 0 else f"FAIL: exit code {returncode}"
                error_message = None if returncode == 0 else f"Process exited with code {returncode}"
                elapsed = time.time() - job["start_time"]

                if returncode == 0:
                    _write_json(
                        job["result_file"],
                        {
                            "task": job["task"],
                            "model": job["model"],
                            "mode": job["mode_tag"],
                            "seed": job["seed"],
                            "status": "done",
                            "gpu": job["worker"]["gpu_arg"],
                            "signature": job["signature"],
                            "log_file": str(job["log_file"]),
                            "exp_dir": job["exp_dir"],
                        },
                    )
                else:
                    print(f"  ERROR: {job['desc']} failed on {job['worker']['label']} with exit code {returncode}")

                run_records.append(
                    {
                        "task": job["task"],
                        "model": job["model"],
                        "mode": job["mode_tag"],
                        "seed": job["seed"],
                        "order": job["order"],
                        "desc": job["desc"],
                        "batch_size": job["batch_size"],
                        "elapsed": elapsed,
                        "status": status,
                        "error": error_message,
                        "gpu": job["worker"]["gpu_arg"] if base_args.device.type == "cuda" else None,
                        "log_file": str(job["log_file"]),
                        "exp_dir": job["exp_dir"],
                    }
                )

                del active_jobs[slot_id]
                remaining_eta, _ = _write_progress_artifacts(
                    progress_path, summary_path, run_records, total_runs
                )

                if remaining_eta is not None:
                    print(
                        f"  -> finished on {job['worker']['label']} in {elapsed:.1f}s, "
                        f"estimated remaining {remaining_eta/60:.1f} min"
                    )
                else:
                    print(f"  -> finished on {job['worker']['label']} in {elapsed:.1f}s")

    except KeyboardInterrupt:
        print("\nInterrupted. Terminating active experiments...")
        for job in active_jobs.values():
            if job["process"].poll() is None:
                job["process"].terminate()
        for job in active_jobs.values():
            try:
                job["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                job["process"].kill()
            finally:
                if not job["log_handle"].closed:
                    print("===== End main.py output =====", file=job["log_handle"])
                    print("process_exit_code: interrupted", file=job["log_handle"])
                    job["log_handle"].close()
        raise

    total_elapsed = time.time() - total_start
    ordered_run_records = _sorted_run_records(run_records)
    table = _render_summary_table(ordered_run_records)
    print("\n===== Execution Summary =====")
    print(table)
    print(f"\nTotal: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    with summary_table_path.open("w", encoding="utf-8") as f:
        f.write(str(table))
        f.write(f"\n\nTotal: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)\n")

    final_summary = {
        "profile": base_args.run_profile,
        "dataset": base_args.dataset,
        "state": base_args.state,
        "total_seconds": total_elapsed,
        "runs": ordered_run_records,
    }
    _write_json(summary_path, final_summary)


if __name__ == "__main__":
    run_all()
