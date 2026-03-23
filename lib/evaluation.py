import gc
import numpy as np
import sklearn as sk
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import lib.utils as utils
from lib.utils import get_device

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent


def compute_error(truth, pred_y, mask, func, reduce, norm_dict=None):
    # pred_y shape [n_traj_samples, n_batch, n_tp, n_dim]
    # truth shape  [n_bacth, n_tp, n_dim] or [B, L, n_dim]

    if len(pred_y.shape) == 3:
        pred_y = pred_y.unsqueeze(dim=0)
    n_traj_samples, n_batch, n_tp, n_dim = pred_y.size()
    truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
    mask = mask.repeat(pred_y.size(0), 1, 1, 1)

    if func == "MSE":
        error = (
            (truth_repeated - pred_y) ** 2
        ) * mask  # (n_traj_samples, n_batch, n_tp, n_dim)
    elif func == "MAE":
        error = (
            torch.abs(truth_repeated - pred_y) * mask
        )  # (n_traj_samples, n_batch, n_tp, n_dim)
    elif func == "MAPE":
        if norm_dict == None:
            mask = (truth_repeated != 0) * mask
            truth_div = torch.abs(truth_repeated) + (truth_repeated == 0) * 1e-8
            error = torch.abs(truth_repeated - pred_y) / truth_div * mask
        else:
            data_max = norm_dict["data_max"]
            data_min = norm_dict["data_min"]
            truth_rescale = truth_repeated * (data_max - data_min) + data_min
            pred_y_rescale = pred_y * (data_max - data_min) + data_min
            mask = (truth_rescale != 0) * mask
            truth_rescale_div = torch.abs(truth_rescale) + (truth_rescale == 0) * 1e-8
            error = torch.abs(truth_rescale - pred_y_rescale) / truth_rescale_div * mask
    else:
        raise Exception("Error function not specified")

    error_var_sum = error.reshape(-1, n_dim).sum(dim=0)  # (n_dim, )
    mask_count = mask.reshape(-1, n_dim).sum(dim=0)  # (n_dim, )

    if reduce == "mean":
        ### 1. Compute avg error of each variable first
        ### 2. Compute avg error along the variables
        error_var_avg = error_var_sum / (mask_count + 1e-8)  # (n_dim, )
        # print("error_var_avg", error_var_avg.max().item(), error_var_avg.min().item(), (1.0*error_var_avg).mean().item())
        n_avai_var = torch.count_nonzero(mask_count)
        error_avg = error_var_avg.sum() / n_avai_var  # (1, )

        return error_avg  # a scalar (1, )

    elif reduce == "sum":
        # (n_dim, ) , (n_dim, )
        return error_var_sum, mask_count

    else:
        raise Exception("Reduce argument not specified!")


def _extract_text_node_tensors(batch_dict):
    needed = ("text_observed_data", "text_observed_tp", "text_observed_mask")
    missing = [k for k in needed if k not in batch_dict]
    if missing:
        raise KeyError(
            f"Missing text-guided batch fields: {missing}. "
            "Expected text_observed_data/text_observed_tp/text_observed_mask."
        )
    return (
        batch_dict["text_observed_data"],
        batch_dict["text_observed_tp"],
        batch_dict["text_observed_mask"],
    )


def compute_all_losses(
    model,
    fusion,
    batch_dict,
    enable_text=True,
    use_text_embeddings=True,
    text_guided_graph=False,
):
    # Condition on subsampled points
    # Make predictions for all the points
    # shape of pred --- [n_traj_samples=1, n_batch, n_tp, n_dim]
    model_device = next(model.parameters()).device
    batch_dict = utils.move_batch_to_device(batch_dict, model_device)

    text_guided_mode = bool(enable_text and text_guided_graph)
    dbg_text_graph = bool(getattr(model, "dbg_text_graph", False) and text_guided_mode)
    text_var_data = None
    text_var_tp = None
    text_var_mask = None
    if text_guided_mode:
        text_var_data, text_var_tp, text_var_mask = _extract_text_node_tensors(batch_dict)

    pred_y = model.forecasting(
        batch_dict["tp_to_predict"],
        batch_dict["observed_data"],
        batch_dict["observed_tp"],
        batch_dict["observed_mask"],
        text_var_data=text_var_data,
        text_var_tp=text_var_tp,
        text_var_mask=text_var_mask,
    )
    if torch.isnan(pred_y).any():
        print(f"pred_y: {pred_y}")
        raise ValueError("pred_y contains NaN values.")

    if dbg_text_graph:
        n_pred_var = int(pred_y.size(-1))
        n_tgt_var = int(batch_dict["data_to_predict"].size(-1))
        n_mask_var = int(batch_dict["mask_predicted_data"].size(-1))
        assert n_pred_var == n_tgt_var == n_mask_var, (
            f"Loss variable mismatch: pred={n_pred_var}, target={n_tgt_var}, mask={n_mask_var}"
        )
        if not hasattr(model, "_dbg_loss_shape_logged"):
            print(
                "[DBG text_graph][loss] pred/target/mask vars:",
                n_pred_var,
                n_tgt_var,
                n_mask_var,
            )
            model._dbg_loss_shape_logged = True

    if enable_text and fusion is not None and (not text_guided_mode):
        has_text = batch_dict.get("has_text", None)
        if has_text is None or has_text.any():
            notes_input = (
                batch_dict["notes_embeddings"]
                if use_text_embeddings
                else batch_dict["notes_text"]
            )
            fused_y = fusion(
                notes_input,
                batch_dict["tau"],
                batch_dict["tp_to_predict"],
                pred_y,
            )
            # Only apply fusion to samples that actually have text
            if has_text is not None and not has_text.all():
                mask_txt = has_text.view(-1, 1, 1, 1) if fused_y.dim() == 4 else has_text.view(-1, 1, 1)
                pred_y = torch.where(mask_txt, fused_y, pred_y)
            else:
                pred_y = fused_y

    # Compute avg error of each variable first, then compute avg error of all variables
    if torch.isnan(pred_y).any():
        raise ValueError("pred_y contains NaN values.")
    if torch.isnan(batch_dict["data_to_predict"]).any():
        raise ValueError("data_to_predict contains NaN values.")
    mse = compute_error(
        batch_dict["data_to_predict"],
        pred_y,
        mask=batch_dict["mask_predicted_data"],
        func="MSE",
        reduce="mean",
    )  # a scalar
    # raise Exception(
    #     pred_y.shape,
    #     batch_dict["data_to_predict"].shape,
    #     batch_dict["mask_predicted_data"].shape,
    # )
    # print(batch_dict["mask_predicted_data"])
    # Assert that every sample in batch_dict["mask_predicted_data"] is not all 0
    # for i in range(batch_dict["mask_predicted_data"].shape[0]):
    #     if batch_dict["mask_predicted_data"][i].sum() == 0:
    #         raise ValueError(
    #             f"mask_predicted_data for sample {i} is all zeros: {batch_dict['mask_predicted_data'][i]}"
    #         )
    
    # Check mask is not all zero for every sample
    for i in range(batch_dict["mask_predicted_data"].shape[0]):
        if batch_dict["mask_predicted_data"][i].sum() == 0:
            raise ValueError(
                f"mask_predicted_data for sample {i} is all zeros: {batch_dict['mask_predicted_data'][i]}"
            )

    # mse = masked_mse_nn(
    #     pred_y,
    #     batch_dict["data_to_predict"],
    #     mask=batch_dict["mask_predicted_data"],
    # )  # a scalar
    # print("mse", mse.item())
    # Check if mse is nan
    if torch.isnan(mse).any():
        print(f"pred_y: {pred_y}")
        raise ValueError("MSE is NaN")
    # rmse = torch.sqrt(mse)
    # print(mse, rmse)
    # mae = compute_error(
    #     batch_dict["data_to_predict"],
    #     pred_y,
    #     mask=batch_dict["mask_predicted_data"],
    #     func="MAE",
    #     reduce="mean",
    # )  # a scalar

    ################################
    # mse loss
    loss = mse

    results = {}
    results["loss"] = loss
    results["mse"] = mse.item()
    # results["rmse"] = rmse.item()
    # results["mae"] = mae.item()

    return results


def masked_mse_nn(pred_y: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked MSE using nn.MSELoss, with elementwise masking.

    Args:
        pred_y: [B, T, D] — predictions
        target: [B, T, D] — ground truth
        mask:   [B, T, D] — binary mask

    Returns:
        Scalar masked MSE
    """
    mse_loss = nn.MSELoss(reduction='mean')

    # Flatten all to 1D
    pred_flat = pred_y.reshape(-1)
    target_flat = target.reshape(-1)
    mask_flat = mask.reshape(-1).bool()

    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=pred_y.device)

    return mse_loss(pred_flat[mask_flat], target_flat[mask_flat])


def evaluation(
    model,
    fusion,
    dataloader,
    enable_text=True,
    use_text_embeddings=True,
    text_guided_graph=False,
):

    n_eval_samples = 0
    n_eval_samples_mape = 0
    total_results = {}
    total_results["loss"] = 0
    total_results["mse"] = 0
    total_results["mae"] = 0
    total_results["rmse"] = 0
    total_results["mape"] = 0

    # for _ in range(n_batches):
    for step, batch_dict in enumerate(tqdm(dataloader)):
        model_device = next(model.parameters()).device
        batch_dict = utils.move_batch_to_device(batch_dict, model_device)
        text_guided_mode = bool(enable_text and text_guided_graph)
        dbg_text_graph = bool(getattr(model, "dbg_text_graph", False) and text_guided_mode)
        text_var_data = None
        text_var_tp = None
        text_var_mask = None
        if text_guided_mode:
            text_var_data, text_var_tp, text_var_mask = _extract_text_node_tensors(batch_dict)

        pred_y = model.forecasting(
            batch_dict["tp_to_predict"],
            batch_dict["observed_data"],
            batch_dict["observed_tp"],
            batch_dict["observed_mask"],
            text_var_data=text_var_data,
            text_var_tp=text_var_tp,
            text_var_mask=text_var_mask,
        )

        if dbg_text_graph:
            n_pred_var = int(pred_y.size(-1))
            n_tgt_var = int(batch_dict["data_to_predict"].size(-1))
            n_mask_var = int(batch_dict["mask_predicted_data"].size(-1))
            assert n_pred_var == n_tgt_var == n_mask_var, (
                f"[eval] Loss variable mismatch: pred={n_pred_var}, target={n_tgt_var}, mask={n_mask_var}"
            )

        if enable_text and fusion is not None and (not text_guided_mode):
            has_text = batch_dict.get("has_text", None)
            if has_text is None or has_text.any():
                notes_input = (
                    batch_dict["notes_embeddings"]
                    if use_text_embeddings
                    else batch_dict["notes_text"]
                )
                fused_y = fusion(
                    notes_input,
                    batch_dict["tau"],
                    batch_dict["tp_to_predict"],
                    pred_y,
                )
                if has_text is not None and not has_text.all():
                    mask_txt = has_text.view(-1, 1, 1, 1) if fused_y.dim() == 4 else has_text.view(-1, 1, 1)
                    pred_y = torch.where(mask_txt, fused_y, pred_y)
                else:
                    pred_y = fused_y

        # (n_dim, ) , (n_dim, )
        se_var_sum, mask_count = compute_error(
            batch_dict["data_to_predict"],
            pred_y,
            mask=batch_dict["mask_predicted_data"],
            func="MSE",
            reduce="sum",
        )  # a vector

        ae_var_sum, _ = compute_error(
            batch_dict["data_to_predict"],
            pred_y,
            mask=batch_dict["mask_predicted_data"],
            func="MAE",
            reduce="sum",
        )  # a vector

        # norm_dict = {"data_max": batch_dict["data_max"], "data_min": batch_dict["data_min"]}
        ape_var_sum, mask_count_mape = compute_error(
            batch_dict["data_to_predict"],
            pred_y,
            mask=batch_dict["mask_predicted_data"],
            func="MAPE",
            reduce="sum",
        )  # a vector

        # add a tensor (n_dim, )
        total_results["loss"] += se_var_sum
        total_results["mse"] += se_var_sum
        total_results["mae"] += ae_var_sum
        total_results["mape"] += ape_var_sum
        n_eval_samples += mask_count
        n_eval_samples_mape += mask_count_mape

    n_avai_var = torch.count_nonzero(n_eval_samples)
    n_avai_var_mape = torch.count_nonzero(n_eval_samples_mape)

    ### 1. Compute avg error of each variable first
    ### 2. Compute avg error along the variables
    total_results["loss"] = (
        total_results["loss"] / (n_eval_samples + 1e-8)
    ).sum() / n_avai_var
    total_results["mse"] = (
        total_results["mse"] / (n_eval_samples + 1e-8)
    ).sum() / n_avai_var
    total_results["mae"] = (
        total_results["mae"] / (n_eval_samples + 1e-8)
    ).sum() / n_avai_var
    total_results["rmse"] = torch.sqrt(total_results["mse"])
    total_results["mape"] = (
        total_results["mape"] / (n_eval_samples_mape + 1e-8)
    ).sum() / n_avai_var_mape

    for key, var in total_results.items():
        if isinstance(var, torch.Tensor):
            var = var.item()
        total_results[key] = var

    return total_results


# ===================== Classification =====================


def _get_text_kwargs(batch_dict, enable_text, text_guided_graph):
    """Extract text-guided-graph kwargs if needed, else return empty dict."""
    text_guided_mode = bool(enable_text and text_guided_graph)
    if not text_guided_mode:
        return {}
    text_var_data, text_var_tp, text_var_mask = _extract_text_node_tensors(batch_dict)
    return dict(
        text_var_data=text_var_data,
        text_var_tp=text_var_tp,
        text_var_mask=text_var_mask,
    )


def _get_late_fusion_query_tp(batch_dict):
    """
    Resolve the time query used by late fusion for cls/reg tasks.

    For chunked classification we explicitly pass the history-window cutoff in
    `fusion_query_tp`. For patient-level tasks we pass each patient's observed
    trajectory end time. If older callers do not provide this field, fall back
    to the latest observed numeric timestamp in the batch.
    """
    if "fusion_query_tp" in batch_dict:
        t_hat = batch_dict["fusion_query_tp"]
        if t_hat.dim() == 1:
            t_hat = t_hat.unsqueeze(-1)
        return t_hat

    observed_tp = batch_dict["observed_tp"]
    observed_mask = batch_dict["observed_mask"]
    if observed_tp.shape != observed_mask.shape:
        raise ValueError(
            "observed_tp and observed_mask must have matching shapes to derive "
            "a late-fusion query time."
        )

    valid_tp = torch.where(
        observed_mask > 0,
        observed_tp,
        torch.full_like(observed_tp, float("-inf")),
    )
    t_hat = valid_tp.reshape(valid_tp.size(0), -1).amax(dim=1, keepdim=True)
    t_hat = torch.where(torch.isfinite(t_hat), t_hat, torch.zeros_like(t_hat))
    return t_hat


def _apply_late_fusion_on_h(
    model, fusion, batch_dict, h_pooled, head_fn,
    use_text_embeddings=True,
):
    """
    Apply late text fusion at the representation level for cls/reg tasks.

    Args:
        model: the tPatchGNN model (used for cls_head/reg_head)
        fusion: FusionModel instance (TTF + MMF, with C = hid_dim)
        batch_dict: batch dictionary containing tau, notes, has_text
        h_pooled: (B, hid_dim) encoder pooled representation
        head_fn: callable, model.cls_head or model.reg_head
        use_text_embeddings: whether to use precomputed embeddings

    Returns:
        fused output from head_fn applied on fused representation
    """
    has_text = batch_dict.get("has_text", None)

    # If no samples in this batch have text, skip fusion entirely
    if has_text is not None and not has_text.any():
        return head_fn(h_pooled)

    # Extract text inputs
    notes_input = (
        batch_dict["notes_embeddings"]
        if use_text_embeddings
        else batch_dict["notes_text"]
    )
    tau = batch_dict["tau"]  # (B, N_max) normalized note timestamps
    t_hat = _get_late_fusion_query_tp(batch_dict)  # (B, 1)

    # Reshape h_pooled to (B, 1, hid_dim) to match MMF's (B, T, C) format
    h_3d = h_pooled.unsqueeze(1)  # (B, 1, hid_dim)

    # Apply fusion: TTF + MMF
    fused_h_3d = fusion(notes_input, tau, t_hat, h_3d)  # (B, 1, hid_dim)
    fused_h = fused_h_3d.squeeze(1)  # (B, hid_dim)

    # Only apply fusion to samples that actually have text
    if has_text is not None and not has_text.all():
        mask_txt = has_text.view(-1, 1)  # (B, 1)
        fused_h = torch.where(mask_txt, fused_h, h_pooled)

    # Apply the task head on fused representation
    return head_fn(fused_h)


def _discrete_hazard_loss(hazard_logits, risk_targets, eps=1e-6):
    """
    Discrete-time survival loss for onset prediction.

    risk_targets is either all zeros (no event in horizon) or a one-hot vector
    marking the future hour bin containing the onset.
    """
    hazard_probs = torch.sigmoid(hazard_logits).clamp(min=eps, max=1.0 - eps)
    log_h = torch.log(hazard_probs)
    log_1mh = torch.log1p(-hazard_probs)

    event_mask = risk_targets.sum(dim=-1) > 0.5
    event_bins = risk_targets.argmax(dim=-1)

    padded_survival = F.pad(torch.cumsum(log_1mh, dim=-1), (1, 0), value=0.0)
    surv_before_event = padded_survival.gather(1, event_bins.unsqueeze(1)).squeeze(1)
    event_logprob = log_h.gather(1, event_bins.unsqueeze(1)).squeeze(1)
    pos_loss = -(surv_before_event + event_logprob)
    neg_loss = -log_1mh.sum(dim=-1)

    loss = torch.where(event_mask, pos_loss, neg_loss)
    return loss.mean()


def compute_classification_losses(
    model,
    batch_dict,
    enable_text=True,
    text_guided_graph=False,
    pos_weight=None,
    fusion=None,
    use_text_embeddings=True,
):
    """
    Compute BCE loss for classification task.
    When fusion is provided (and not text_guided_graph), applies late text fusion
    at the representation level before the classification head.
    Returns dict with 'loss' (Tensor) and 'bce' (float).
    """
    model_device = next(model.parameters()).device
    batch_dict = utils.move_batch_to_device(batch_dict, model_device)
    text_guided_mode = bool(enable_text and text_guided_graph)
    text_kwargs = _get_text_kwargs(batch_dict, enable_text, text_guided_graph)

    # Determine if we need late fusion
    use_late_fusion = (
        enable_text
        and fusion is not None
        and not text_guided_mode
    )

    if use_late_fusion:
        logits, h_pooled = model.classify(
            batch_dict["observed_data"],
            batch_dict["observed_tp"],
            batch_dict["observed_mask"],
            return_h=True,
            **text_kwargs,
        )
        logits = _apply_late_fusion_on_h(
            model, fusion, batch_dict, h_pooled, model.cls_head,
            use_text_embeddings=use_text_embeddings,
        )
    else:
        logits = model.classify(
            batch_dict["observed_data"],
            batch_dict["observed_tp"],
            batch_dict["observed_mask"],
            return_aux=True,
            **text_kwargs,
        )  # (B, n_labels)
        if isinstance(logits, tuple):
            logits, aux_outputs = logits
        else:
            aux_outputs = {}

    labels = batch_dict["labels"].to(logits.device, dtype=logits.dtype)  # (B, n_labels)

    if pos_weight is not None:
        pw = pos_weight.to(logits.device, dtype=logits.dtype)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    primary_bce = loss_fn(logits, labels)
    loss = primary_bce
    hazard_loss_value = 0.0
    aux_bce_value = 0.0

    if not use_late_fusion:
        hazard_logits = aux_outputs.get("hazard_logits")
        risk_targets = batch_dict.get("risk_targets")
        hazard_weight = float(getattr(model, "hazard_loss_weight", 0.7))
        aux_bce_weight = float(getattr(model, "aux_bce_weight", 0.2))
        if hazard_logits is not None and risk_targets is not None:
            hazard_loss = _discrete_hazard_loss(
                hazard_logits,
                risk_targets.to(hazard_logits.device, dtype=hazard_logits.dtype),
            )
            loss = loss + hazard_weight * hazard_loss
            hazard_loss_value = float(hazard_loss.item())
        aux_global_logit = aux_outputs.get("aux_global_logit")
        if aux_global_logit is not None:
            aux_bce = loss_fn(aux_global_logit, labels)
            loss = loss + aux_bce_weight * aux_bce
            aux_bce_value = float(aux_bce.item())

    return {
        "loss": loss,
        "bce": primary_bce.item(),
        "hazard": hazard_loss_value,
        "aux_bce": aux_bce_value,
    }


def _collect_classification_outputs(
    model,
    dataloader,
    enable_text=True,
    text_guided_graph=False,
    fusion=None,
    use_text_embeddings=True,
):
    """Run a classification dataloader once and collect probabilities/labels."""
    all_probs = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    text_guided_mode = bool(enable_text and text_guided_graph)
    use_late_fusion = (
        enable_text
        and fusion is not None
        and not text_guided_mode
    )

    for batch_dict in tqdm(dataloader, desc="Eval(cls)"):
        model_device = next(model.parameters()).device
        batch_dict = utils.move_batch_to_device(batch_dict, model_device)
        text_kwargs = _get_text_kwargs(batch_dict, enable_text, text_guided_graph)

        if use_late_fusion:
            logits, h_pooled = model.classify(
                batch_dict["observed_data"],
                batch_dict["observed_tp"],
                batch_dict["observed_mask"],
                return_h=True,
                **text_kwargs,
            )
            logits = _apply_late_fusion_on_h(
                model, fusion, batch_dict, h_pooled, model.cls_head,
                use_text_embeddings=use_text_embeddings,
            )
        else:
            logits = model.classify(
                batch_dict["observed_data"],
                batch_dict["observed_tp"],
                batch_dict["observed_mask"],
                **text_kwargs,
            )

        labels = batch_dict["labels"].to(logits.device, dtype=logits.dtype)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        total_loss += loss.item()
        n_batches += 1

        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    if not all_probs:
        return {
            "loss": float("nan"),
            "probs": np.empty((0, 0), dtype=np.float32),
            "labels": np.empty((0, 0), dtype=np.float32),
        }

    return {
        "loss": total_loss / max(n_batches, 1),
        "probs": np.concatenate(all_probs, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
    }


def _find_best_f1_thresholds(all_probs, all_labels):
    """Choose per-label thresholds on validation data by maximizing F1."""
    n_labels = all_probs.shape[1]
    thresholds = np.full(n_labels, 0.5, dtype=np.float32)
    for i in range(n_labels):
        y_true = all_labels[:, i]
        y_score = all_probs[:, i]
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        if pr_thresholds.size == 0:
            continue
        f1_scores = (2.0 * precision[:-1] * recall[:-1]) / (
            precision[:-1] + recall[:-1] + 1e-8
        )
        if not np.isfinite(f1_scores).any():
            continue
        best_idx = int(np.nanargmax(f1_scores))
        thresholds[i] = float(pr_thresholds[best_idx])
    return thresholds


def _classification_metrics_from_outputs(
    all_probs,
    all_labels,
    label_names=None,
    decision_thresholds=None,
):
    """Compute scalar and per-label classification metrics from cached outputs."""
    if all_probs.size == 0 or all_labels.size == 0:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "f1": float("nan"),
        }

    n_labels = all_probs.shape[1]
    if decision_thresholds is None:
        thresholds = np.full(n_labels, 0.5, dtype=np.float32)
    else:
        thresholds = np.asarray(decision_thresholds, dtype=np.float32).reshape(-1)
        if thresholds.shape[0] != n_labels:
            raise ValueError(
                f"Expected {n_labels} decision thresholds, got {thresholds.shape[0]}."
            )

    aurocs = []
    auprcs = []
    f1s = []
    per_label = {}

    for i in range(n_labels):
        y_true = all_labels[:, i]
        y_score = all_probs[:, i]
        thr = float(thresholds[i])
        y_pred = (y_score >= thr).astype(np.float32)
        name = label_names[i] if label_names and i < len(label_names) else f"label_{i}"

        f1 = f1_score(y_true, y_pred, zero_division=0)
        f1s.append(f1)
        per_label[f"f1_{name}"] = f1
        per_label[f"decision_threshold_{name}"] = thr

        if len(np.unique(y_true)) < 2:
            per_label[f"auroc_{name}"] = float("nan")
            per_label[f"auprc_{name}"] = float("nan")
            continue

        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        aurocs.append(auc)
        auprcs.append(ap)
        per_label[f"auroc_{name}"] = auc
        per_label[f"auprc_{name}"] = ap

    results = {
        "auroc": float(np.nanmean(aurocs)) if aurocs else float("nan"),
        "auprc": float(np.nanmean(auprcs)) if auprcs else float("nan"),
        "f1": float(np.mean(f1s)) if f1s else float("nan"),
        "decision_thresholds": thresholds.tolist(),
    }
    if n_labels == 1:
        results["decision_threshold"] = float(thresholds[0])
    results.update(per_label)
    return results


def evaluation_classification(
    model,
    dataloader,
    enable_text=True,
    text_guided_graph=False,
    label_names=None,
    fusion=None,
    use_text_embeddings=True,
    decision_thresholds=None,
    tune_decision_thresholds=False,
):
    """
    Evaluate classification: collect all predictions, compute AUROC & AUPRC per label.
    Returns dict with 'loss', 'auroc', 'auprc', 'f1', and per-label metrics.
    """
    outputs = _collect_classification_outputs(
        model,
        dataloader,
        enable_text=enable_text,
        text_guided_graph=text_guided_graph,
        fusion=fusion,
        use_text_embeddings=use_text_embeddings,
    )
    thresholds = decision_thresholds
    if tune_decision_thresholds:
        thresholds = _find_best_f1_thresholds(outputs["probs"], outputs["labels"])

    results = _classification_metrics_from_outputs(
        outputs["probs"],
        outputs["labels"],
        label_names=label_names,
        decision_thresholds=thresholds,
    )
    results["loss"] = outputs["loss"]
    results["threshold_tuned_on_eval"] = bool(tune_decision_thresholds)
    return results


# ===================== Regression =====================


def compute_regression_losses(
    model,
    batch_dict,
    enable_text=True,
    text_guided_graph=False,
    fusion=None,
    use_text_embeddings=True,
):
    """
    Compute Huber loss for regression task.
    When fusion is provided (and not text_guided_graph), applies late text fusion
    at the representation level before the regression head.
    Returns dict with 'loss' (Tensor) and 'huber' (float).
    """
    model_device = next(model.parameters()).device
    batch_dict = utils.move_batch_to_device(batch_dict, model_device)
    text_guided_mode = bool(enable_text and text_guided_graph)
    text_kwargs = _get_text_kwargs(batch_dict, enable_text, text_guided_graph)

    use_late_fusion = (
        enable_text
        and fusion is not None
        and not text_guided_mode
    )

    if use_late_fusion:
        preds, h_pooled = model.regress(
            batch_dict["observed_data"],
            batch_dict["observed_tp"],
            batch_dict["observed_mask"],
            return_h=True,
            **text_kwargs,
        )
        preds = _apply_late_fusion_on_h(
            model, fusion, batch_dict, h_pooled, model.reg_head,
            use_text_embeddings=use_text_embeddings,
        )
    else:
        preds = model.regress(
            batch_dict["observed_data"],
            batch_dict["observed_tp"],
            batch_dict["observed_mask"],
            **text_kwargs,
        )  # (B, 1)

    labels = batch_dict["labels"].to(preds.device, dtype=preds.dtype)  # (B, 1)

    loss_fn = nn.HuberLoss(delta=1.0)
    loss = loss_fn(preds, labels)

    return {"loss": loss, "huber": loss.item()}


def evaluation_regression(
    model,
    dataloader,
    enable_text=True,
    text_guided_graph=False,
    fusion=None,
    use_text_embeddings=True,
):
    """
    Evaluate regression: collect all predictions, compute MAE, RMSE, R^2.
    Labels are in log(1+x) space; MAE is reported in original scale.
    """
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    text_guided_mode = bool(enable_text and text_guided_graph)
    use_late_fusion = (
        enable_text
        and fusion is not None
        and not text_guided_mode
    )

    for step, batch_dict in enumerate(tqdm(dataloader, desc="Eval(reg)")):
        model_device = next(model.parameters()).device
        batch_dict = utils.move_batch_to_device(batch_dict, model_device)
        text_kwargs = _get_text_kwargs(batch_dict, enable_text, text_guided_graph)

        if use_late_fusion:
            preds, h_pooled = model.regress(
                batch_dict["observed_data"],
                batch_dict["observed_tp"],
                batch_dict["observed_mask"],
                return_h=True,
                **text_kwargs,
            )
            preds = _apply_late_fusion_on_h(
                model, fusion, batch_dict, h_pooled, model.reg_head,
                use_text_embeddings=use_text_embeddings,
            )
        else:
            preds = model.regress(
                batch_dict["observed_data"],
                batch_dict["observed_tp"],
                batch_dict["observed_mask"],
                **text_kwargs,
            )

        labels = batch_dict["labels"].to(preds.device, dtype=preds.dtype)
        loss = nn.functional.huber_loss(preds, labels)
        total_loss += loss.item()
        n_batches += 1

        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()

    # Convert from log(1+x) back to original scale
    preds_orig = np.expm1(all_preds)
    labels_orig = np.expm1(all_labels)

    mae = float(np.mean(np.abs(preds_orig - labels_orig)))
    rmse = float(np.sqrt(np.mean((preds_orig - labels_orig) ** 2)))
    median_ae = float(np.median(np.abs(preds_orig - labels_orig)))

    ss_res = np.sum((labels_orig - preds_orig) ** 2)
    ss_tot = np.sum((labels_orig - np.mean(labels_orig)) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

    return {
        "loss": total_loss / max(n_batches, 1),
        "mae": mae,
        "rmse": rmse,
        "median_ae": median_ae,
        "r2": r2,
    }
