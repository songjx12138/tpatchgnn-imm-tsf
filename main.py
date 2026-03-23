import argparse
import math
import os
import time

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from fusions.FusionModel import FusionModel
from fusions.load_llm import get_default_max_length
from lib.evaluation import (
    compute_all_losses,
    compute_classification_losses,
    compute_regression_losses,
    evaluation,
    evaluation_classification,
    evaluation_regression,
)
from lib.parse_datasets import parse_datasets
from lib.task_config import get_task_config, TASK_CONFIGS
from models.tPatchGNN import tPatchGNN
from models.baseline_models import BASELINE_MODELS, create_baseline_model
from utils.tools import print_formatted_dict, set_seed


def get_args_from_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TIME-IMM (tPatchGNN only)")

    # Execution
    parser.add_argument("--overwrite_args", action="store_true", default=False)
    parser.add_argument("--state", type=str, default="def")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=str, default="0")

    # Data
    parser.add_argument("--dataset", type=str, default="MIMIC")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="Maximum number of patient/record folders to use. Default: use all available.",
    )
    parser.add_argument("--enable_text", action="store_true")
    parser.add_argument("--use_text_embeddings", action="store_true")
    parser.add_argument("--text_guided_graph", action="store_true")
    parser.add_argument("--dbg_text_graph", action="store_true")
    parser.add_argument("--time_unit", type=str, default="hours",
                        choices=["seconds", "minutes", "hours", "days", "weeks", "custom"])
    parser.add_argument("--unit_scale", type=float, default=None)
    parser.add_argument("--history", type=int, default=None)
    parser.add_argument("--pred_window", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--rec_ids", nargs="+", default=None)
    parser.add_argument("--task", type=str, default="forecasting",
                        choices=["forecasting", "classification", "regression"])
    parser.add_argument("--task_name", type=str, default=None,
                        choices=list(TASK_CONFIGS.keys()),
                        help="Named task (overrides --task, --history, --pred_window, --stride)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="(Used by main_all.py) List of tasks to run")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=["tPatchGNN", "Linear", "LSTM", "Transformer"],
                        help="(Used by main_all.py) List of models to run")
    parser.add_argument("--modalities", nargs="+", default=None,
                        choices=["num", "text", "text_guided_graph"],
                        help="(Used by main_all.py) Modalities to run")
    parser.add_argument("--run_profile", type=str, default=None,
                        choices=["paper_full", "smoke"],
                        help="(Used by main_all.py) Runner preset to apply")
    parser.add_argument("--batch_size_num", type=int, default=None,
                        help="(Used by main_all.py) Batch size for numeric-only runs")
    parser.add_argument("--batch_size_text", type=int, default=None,
                        help="(Used by main_all.py) Batch size for late-fusion text runs")
    parser.add_argument("--batch_size_text_guided_graph", type=int, default=None,
                        help="(Used by main_all.py) Batch size for text-guided-graph runs")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="(Used by main_all.py) Seeds to run")
    parser.add_argument("--log_root", type=str, default="logs/main_all",
                        help="(Used by main_all.py) Directory for batch-run logs")
    parser.add_argument("--show_dataset_summary", action="store_true", default=False,
                        help="Print expensive full-dataset descriptive summary before training")
    parser.add_argument("--show_record_chunk_counts", action="store_true", default=False,
                        help="Print per-record chunk counts during window generation")

    # tPatchGNN
    parser.add_argument("--model", type=str, default="tPatchGNN",
                        choices=["tPatchGNN", "Linear", "LSTM", "Transformer"])
    parser.add_argument("--outlayer", type=str, default="Linear", choices=["Linear", "CNN"])
    parser.add_argument("-ps", "--patch_size", type=int, default=None)
    parser.add_argument("--npatch", type=int, default=None)
    parser.add_argument("--patch_stride", type=int, default=None)
    parser.add_argument("-hd", "--hid_dim", type=int, default=32)
    parser.add_argument("-td", "--te_dim", type=int, default=10)
    parser.add_argument("-nd", "--node_dim", type=int, default=10)
    parser.add_argument("--hop", type=int, default=1)
    parser.add_argument("--tf_layer", type=int, default=1)
    parser.add_argument("--nlayer", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=1)

    # Fusion (TTF/MMF)
    parser.add_argument("--TTF_module", type=str, default="TTF_RecAvg",
                        choices=["TTF_RecAvg", "TTF_T2V_XAttn"])
    parser.add_argument("--MMF_module", type=str, default="MMF_GR_Add",
                        choices=["MMF_GR_Add", "MMF_XAttn_Add"])
    parser.add_argument("--llm_model_fusion", type=str, default="ClinicalBERT")
    parser.add_argument("--llm_layers_fusion", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--d_txt", type=int, default=768)
    parser.add_argument("--max_text_vars", type=int, default=1)
    parser.add_argument("--recency_sigma", type=float, default=1.0)
    parser.add_argument("--n_heads_fusion", type=int, default=1)
    parser.add_argument("--kappa", type=float, default=0.5)

    # Training
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--early_stop_delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w_decay", type=float, default=0.01)
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hazard_loss_weight", type=float, default=0.7,
                        help="Auxiliary discrete-time risk loss weight for window classification.")
    parser.add_argument("--aux_bce_weight", type=float, default=0.2,
                        help="Auxiliary BCE weight on the global pooled classification head.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--persistent_workers", action="store_true", default=False)
    parser.add_argument("--use_amp", action="store_true", default=False)

    # Logging
    parser.add_argument("--logmode", type=str, default="a")
    parser.add_argument("--save", type=str, default="experiments/")
    parser.add_argument("--load", type=str, default=None)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.data_device = torch.device("cpu")
    args.PID = os.getpid()
    print("PID, device:", args.PID, args.device)

    return args


def apply_task_config(args: argparse.Namespace) -> argparse.Namespace:
    """If --task_name is given, override task type and window params from registry."""
    if args.task_name is None:
        return args
    cfg = get_task_config(args.task_name)
    if cfg.task_type in ("window_cls", "patient_cls"):
        args.task = "classification"
    elif cfg.task_type == "patient_reg":
        args.task = "regression"
    args.exclude_features = cfg.exclude_features
    args.task_level = cfg.task_type  # "window_cls" / "patient_cls" / "patient_reg"
    # Only apply task defaults when user didn't explicitly set them
    if args.history is None and cfg.default_history is not None:
        args.history = cfg.default_history
    if args.pred_window is None and cfg.default_pred_window is not None:
        args.pred_window = cfg.default_pred_window
    if args.stride is None and cfg.default_stride is not None:
        args.stride = cfg.default_stride
    # Flag for model constructor
    args.is_regression = cfg.task_type == "patient_reg"
    return args


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Normalize coupled flags and validate high-risk argument combinations."""
    if args.text_guided_graph:
        # Text-guided graph always consumes precomputed text embeddings.
        args.enable_text = True
        args.use_text_embeddings = True

    # Baseline models only support late fusion, not text_guided_graph
    if args.model in ("Linear", "LSTM", "Transformer") and args.text_guided_graph:
        raise ValueError(
            f"--text_guided_graph is not supported for baseline model {args.model}. "
            "Use --enable_text for late fusion instead."
        )

    if args.task == "regression" and args.task_name is None:
        raise ValueError(
            "Standalone --task regression is not supported. "
            "Use --task_name los for the patient-level regression task."
        )

    if args.n is not None and args.n <= 0:
        raise ValueError("-n must be a positive integer.")
    if args.batch_size_num is not None and args.batch_size_num <= 0:
        raise ValueError("--batch_size_num must be a positive integer.")
    if args.batch_size_text is not None and args.batch_size_text <= 0:
        raise ValueError("--batch_size_text must be a positive integer.")
    if args.batch_size_text_guided_graph is not None and args.batch_size_text_guided_graph <= 0:
        raise ValueError("--batch_size_text_guided_graph must be a positive integer.")
    if args.npatch is not None and args.npatch <= 0:
        raise ValueError("--npatch must be a positive integer.")
    if args.patch_size is not None and args.patch_size <= 0:
        raise ValueError("--patch_size must be a positive integer.")
    if args.patch_stride is not None and args.patch_stride <= 0:
        raise ValueError("--patch_stride must be a positive integer.")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be a non-negative integer.")
    if os.name == "nt" and args.num_workers > 0:
        print("Windows detected: forcing --num_workers=0 for collate compatibility.")
        args.num_workers = 0
    if args.num_workers == 0:
        args.persistent_workers = False

    # The dataset loader needs the file naming convention before it starts reading.
    if args.enable_text and args.max_length is None:
        args.max_length = get_default_max_length(args.llm_model_fusion)

    return args


# ===================== Training helpers =====================


def train_one_epoch_cls(
    model, fusion, optimizer, train_loader, args, pos_weight=None, scaler=None
):
    """Train one epoch for classification tasks."""
    model.train()
    if fusion is not None:
        fusion.train()
    total_loss = 0.0
    n_batches = 0

    for batch_dict in tqdm(train_loader, desc="Train(cls)"):
        optimizer.zero_grad()

        if args.use_amp:
            with autocast():
                results = compute_classification_losses(
                    model, batch_dict,
                    enable_text=args.enable_text,
                    text_guided_graph=args.text_guided_graph,
                    pos_weight=pos_weight,
                    fusion=fusion,
                    use_text_embeddings=args.use_text_embeddings,
                )
            scaler.scale(results["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            results = compute_classification_losses(
                model, batch_dict,
                enable_text=args.enable_text,
                text_guided_graph=args.text_guided_graph,
                pos_weight=pos_weight,
                fusion=fusion,
                use_text_embeddings=args.use_text_embeddings,
            )
            results["loss"].backward()
            optimizer.step()

        total_loss += float(results["loss"].item())
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


def train_one_epoch_reg(
    model, fusion, optimizer, train_loader, args, scaler=None
):
    """Train one epoch for regression tasks."""
    model.train()
    if fusion is not None:
        fusion.train()
    total_loss = 0.0
    n_batches = 0

    for batch_dict in tqdm(train_loader, desc="Train(reg)"):
        optimizer.zero_grad()

        if args.use_amp:
            with autocast():
                results = compute_regression_losses(
                    model, batch_dict,
                    enable_text=args.enable_text,
                    text_guided_graph=args.text_guided_graph,
                    fusion=fusion,
                    use_text_embeddings=args.use_text_embeddings,
                )
            scaler.scale(results["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            results = compute_regression_losses(
                model, batch_dict,
                enable_text=args.enable_text,
                text_guided_graph=args.text_guided_graph,
                fusion=fusion,
                use_text_embeddings=args.use_text_embeddings,
            )
            results["loss"].backward()
            optimizer.step()

        total_loss += results["huber"]
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


def train_one_epoch_forecast(
    model, fusion, optimizer, train_loader, args, scaler=None
):
    """Train one epoch for forecasting tasks."""
    model.train()
    if fusion is not None:
        fusion.train()
    total_loss = 0.0
    n_batches = 0

    for batch_dict in tqdm(train_loader, desc="Train(fcst)"):
        optimizer.zero_grad()

        if args.use_amp:
            with autocast():
                results = compute_all_losses(
                    model, fusion, batch_dict,
                    enable_text=args.enable_text,
                    use_text_embeddings=args.use_text_embeddings,
                    text_guided_graph=args.text_guided_graph,
                )
            scaler.scale(results["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            results = compute_all_losses(
                model, fusion, batch_dict,
                enable_text=args.enable_text,
                use_text_embeddings=args.use_text_embeddings,
                text_guided_graph=args.text_guided_graph,
            )
            results["loss"].backward()
            optimizer.step()

        total_loss += results["mse"]
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


def get_early_stop_metric(args):
    """Return (metric_name, higher_is_better) for early stopping."""
    if args.task_name is not None:
        cfg = get_task_config(args.task_name)
        if cfg.early_stop_metric is not None:
            metric_name = cfg.early_stop_metric
            higher_is_better = metric_name in {"auroc", "auprc", "f1", "r2"}
            return metric_name, higher_is_better
        if cfg.metrics:
            metric_name = cfg.metrics[0]
            higher_is_better = metric_name in {"auroc", "auprc", "f1", "r2"}
            return metric_name, higher_is_better
    if args.task == "classification":
        return "auroc", True
    if args.task == "regression":
        return "mae", False
    return "mse", False


# ===================== Main =====================


def main():
    args = get_args_from_parser()
    args = apply_task_config(args)
    args = finalize_args(args)
    # Fallback defaults for window params
    if args.history is None:
        args.history = 24
    if args.pred_window is None:
        args.pred_window = 24
    if args.stride is None:
        args.stride = args.history  # default stride = history
    set_seed(args.seed)

    print(f"Task: {args.task}" + (f" ({args.task_name})" if args.task_name else ""))

    # ---- Data ----
    data_obj = parse_datasets(
        args,
        show_summary=bool(getattr(args, "show_dataset_summary", False)),
    )
    train_loader = data_obj["train_dataloader"]
    val_loader = data_obj["val_dataloader"]
    test_loader = data_obj["test_dataloader"]
    input_dim = data_obj["input_dim"]

    args.C = input_dim  # number of variables (channels)

    # ---- Model ----
    if args.task == "classification":
        args.n_labels = data_obj.get("n_labels", 1)
    elif args.task == "regression":
        args.n_labels = 0
        args.is_regression = True
    else:
        args.n_labels = 0
        args.is_regression = False

    # tPatchGNN does not consume max_input_len/max_pred_len, so avoid an extra
    # full pass over every dataloader before training starts.
    args.max_input_len = None
    args.max_pred_len = None

    if args.model == "tPatchGNN":
        model = tPatchGNN(args, dropout=args.dropout).to(args.device)
    else:
        model = create_baseline_model(args, dropout=args.dropout).to(args.device)
    print(f"Model: {args.model}, params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Fusion (optional) ----
    fusion = None
    if args.enable_text and not args.text_guided_graph:
        # For cls/reg tasks, fuse at representation level (C = hid_dim)
        # For forecasting, fuse at output level (C = input_dim)
        if args.task in ("classification", "regression"):
            fusion_args = argparse.Namespace(**vars(args))
            fusion_args.C = args.hid_dim
            fusion = FusionModel(fusion_args).to(args.device)
        else:
            fusion = FusionModel(args).to(args.device)
        print(f"Fusion params: {sum(p.numel() for p in fusion.parameters()):,}")

    if args.load is not None:
        if not os.path.isfile(args.load):
            raise FileNotFoundError(f"Checkpoint not found: {args.load}")
        ckpt = torch.load(args.load, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        if fusion is not None and "fusion" in ckpt:
            fusion.load_state_dict(ckpt["fusion"])
        print(f"Loaded checkpoint from {args.load}")

    # ---- Optimizer ----
    params = list(model.parameters())
    if fusion is not None:
        params += list(fusion.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.w_decay)
    scaler = GradScaler() if args.use_amp else None

    # ---- pos_weight for classification ----
    pos_weight = data_obj.get("pos_weight", None)

    # ---- Experiment dir ----
    task_tag = args.task_name or args.task
    if args.text_guided_graph:
        modality_tag = "text_guided_graph"
    elif args.enable_text:
        modality_tag = "text"
    else:
        modality_tag = "num"
    exp_name = (
        f"{args.dataset}_{task_tag}_"
        f"{args.model}_"
        f"{modality_tag}_"
        f"s{args.seed}_{args.state}"
    )
    exp_dir = os.path.join(args.save, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    ckpt_path = os.path.join(exp_dir, "best_model.pt")
    print(f"Experiment dir: {exp_dir}")

    # ---- Early stopping setup ----
    es_metric, es_higher_better = get_early_stop_metric(args)
    best_val = -float("inf") if es_higher_better else float("inf")
    patience_counter = 0

    label_names = data_obj.get("label_names", None)
    task_cfg = get_task_config(args.task_name) if args.task_name else None
    tune_decision_thresholds = bool(
        args.task == "classification"
        and task_cfg is not None
        and task_cfg.tune_decision_threshold
    )

    # ---- Training loop ----
    for epoch in range(1, args.epoch + 1):
        t_start = time.time()

        # --- Train ---
        if args.task == "classification":
            train_res = train_one_epoch_cls(
                model, fusion, optimizer, train_loader, args,
                pos_weight=pos_weight, scaler=scaler,
            )
        elif args.task == "regression":
            train_res = train_one_epoch_reg(
                model, fusion, optimizer, train_loader, args, scaler=scaler,
            )
        else:
            train_res = train_one_epoch_forecast(
                model, fusion, optimizer, train_loader, args, scaler=scaler,
            )

        # --- Validate ---
        model.eval()
        if fusion is not None:
            fusion.eval()
        with torch.no_grad():
            if args.task == "classification":
                val_res = evaluation_classification(
                    model, val_loader,
                    enable_text=args.enable_text,
                    text_guided_graph=args.text_guided_graph,
                    label_names=label_names,
                    fusion=fusion,
                    use_text_embeddings=args.use_text_embeddings,
                    tune_decision_thresholds=tune_decision_thresholds,
                )
            elif args.task == "regression":
                val_res = evaluation_regression(
                    model, val_loader,
                    enable_text=args.enable_text,
                    text_guided_graph=args.text_guided_graph,
                    fusion=fusion,
                    use_text_embeddings=args.use_text_embeddings,
                )
            else:
                val_res = evaluation(
                    model, fusion, val_loader,
                    enable_text=args.enable_text,
                    use_text_embeddings=args.use_text_embeddings,
                    text_guided_graph=args.text_guided_graph,
                )

        elapsed = time.time() - t_start
        val_metric = val_res.get(es_metric, val_res["loss"])

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_res['loss']:.4f} | "
            f"val_{es_metric}={val_metric:.4f} | "
            f"time={elapsed:.1f}s"
        )
        if tune_decision_thresholds and "decision_thresholds" in val_res:
            tuned_thresholds = ", ".join(
                f"{thr:.3f}" for thr in val_res["decision_thresholds"]
            )
            print(f"  -> val_decision_thresholds=[{tuned_thresholds}]")

        # --- Early stopping ---
        if math.isnan(val_metric):
            # Can't compare nan; treat as no improvement but save if no checkpoint yet
            if not os.path.exists(ckpt_path):
                ckpt = {"model": model.state_dict(), "args": vars(args)}
                if fusion is not None:
                    ckpt["fusion"] = fusion.state_dict()
                if tune_decision_thresholds and "decision_thresholds" in val_res:
                    ckpt["decision_thresholds"] = val_res["decision_thresholds"]
                torch.save(ckpt, ckpt_path)
                print(f"  -> val_{es_metric}=nan, saved initial checkpoint.")
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
                break
        else:
            improved = (
                (val_metric > best_val + args.early_stop_delta)
                if es_higher_better
                else (val_metric < best_val - args.early_stop_delta)
            )
            if improved:
                best_val = val_metric
                patience_counter = 0
                ckpt = {"model": model.state_dict(), "args": vars(args)}
                if fusion is not None:
                    ckpt["fusion"] = fusion.state_dict()
                if tune_decision_thresholds and "decision_thresholds" in val_res:
                    ckpt["decision_thresholds"] = val_res["decision_thresholds"]
                torch.save(ckpt, ckpt_path)
                print(f"  -> New best val_{es_metric}={best_val:.4f}, saved.")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
                    break

    # ---- Test ----
    if test_loader is not None:
        print("\n--- Test evaluation ---")
        if not os.path.exists(ckpt_path):
            print("No checkpoint saved during training, using final model weights.")
        else:
            ckpt = torch.load(ckpt_path, map_location=args.device)
            model.load_state_dict(ckpt["model"])
            if fusion is not None and "fusion" in ckpt:
                fusion.load_state_dict(ckpt["fusion"])
        decision_thresholds = None
        if args.task == "classification" and 'ckpt' in locals():
            decision_thresholds = ckpt.get("decision_thresholds")
        model.eval()
        if fusion is not None:
            fusion.eval()

        with torch.no_grad():
            if args.task == "classification":
                test_res = evaluation_classification(
                    model, test_loader,
                    enable_text=args.enable_text,
                    text_guided_graph=args.text_guided_graph,
                    label_names=label_names,
                    fusion=fusion,
                    use_text_embeddings=args.use_text_embeddings,
                    decision_thresholds=decision_thresholds,
                )
            elif args.task == "regression":
                test_res = evaluation_regression(
                    model, test_loader,
                    enable_text=args.enable_text,
                    text_guided_graph=args.text_guided_graph,
                    fusion=fusion,
                    use_text_embeddings=args.use_text_embeddings,
                )
            else:
                test_res = evaluation(
                    model, fusion, test_loader,
                    enable_text=args.enable_text,
                    use_text_embeddings=args.use_text_embeddings,
                    text_guided_graph=args.text_guided_graph,
                )

        print("Test results:")
        print_formatted_dict(test_res)

        # Save test results
        results_path = os.path.join(exp_dir, "test_results.txt")
        with open(results_path, "w") as f:
            for k, v in test_res.items():
                f.write(f"{k}: {v}\n")
        print(f"Results saved to {results_path}")
    else:
        print("No test set available.")

    print("Done.")


if __name__ == "__main__":
    main()
