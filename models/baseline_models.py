"""
Baseline models (Linear, LSTM, Transformer) adapted to the tPatchGNN interface.

These models receive the same batch_dict from the dataloader as tPatchGNN.
The irregular patched input (B, M, L, N) is converted to a regular (B, T, N)
representation by taking the per-patch mean of observed values, then fed into
standard sequence models.

All three models expose classify / regress / forecasting with the same
signature as tPatchGNN so that evaluation.py works without modification.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared: convert patched irregular data to regular sequence
# ---------------------------------------------------------------------------

def _patches_to_regular(X, mask):
    """
    Convert patched irregular time series to a regular (B, M, N) representation.

    Args:
        X:    (B, M, L, N) — patched observed values
        mask: (B, M, L, N) — binary observation mask

    Returns:
        X_reg: (B, M, N) — per-patch mean (masked), zero-filled where no obs
    """
    # Sum observed values per patch, divide by count
    mask_sum = mask.sum(dim=2).clamp(min=1e-8)  # (B, M, N)
    X_sum = (X * mask).sum(dim=2)                # (B, M, N)
    X_reg = X_sum / mask_sum                     # (B, M, N)
    return X_reg


# ---------------------------------------------------------------------------
# Positional Encoding (shared by Transformer)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


# ---------------------------------------------------------------------------
# Base class with shared interface
# ---------------------------------------------------------------------------

class _BaselineModel(nn.Module):
    """
    Abstract base providing classify / regress / forecasting with the same
    call signature as tPatchGNN.  Subclasses only need to implement
    ``_encode(X_reg) -> (B, hid_dim)``.
    """

    def __init__(self, args, dropout=0):
        super().__init__()
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.n_labels = getattr(args, "n_labels", 1)
        self.is_regression = getattr(args, "is_regression", False)

        # Will be built in subclass after encoder is defined
        self.cls_head = None
        self.reg_head = None

    def _build_heads(self):
        """Build classification / regression heads (call after encoder is ready)."""
        self.cls_head = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hid_dim, max(self.n_labels, 1)),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hid_dim, 1),
        )

    # --- subclass hook ---
    def _encode(self, X_reg):
        """(B, M, N) -> (B, hid_dim)"""
        raise NotImplementedError

    # --- tPatchGNN-compatible interface ---

    def classify(self, X, truth_time_steps, mask=None, return_h=False,
                 text_var_data=None, text_var_tp=None, text_var_mask=None,
                 return_aux=False):
        X_reg = _patches_to_regular(X, mask)  # (B, M, N)
        h = self._encode(X_reg)               # (B, hid_dim)
        logits = self.cls_head(h)             # (B, n_labels)
        if return_h:
            if return_aux:
                return logits, h, {}
            return logits, h
        if return_aux:
            return logits, {}
        return logits

    def regress(self, X, truth_time_steps, mask=None, return_h=False,
                text_var_data=None, text_var_tp=None, text_var_mask=None):
        X_reg = _patches_to_regular(X, mask)
        h = self._encode(X_reg)
        pred = self.reg_head(h)  # (B, 1)
        if return_h:
            return pred, h
        return pred

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None,
                    text_var_data=None, text_var_tp=None, text_var_mask=None):
        """
        Simple forecasting: encode history, then project to (B, L_pred, N).
        """
        B, M, L_in, N = X.shape
        L_pred = time_steps_to_predict.shape[-1]
        X_reg = _patches_to_regular(X, mask)  # (B, M, N)
        h = self._encode(X_reg)               # (B, hid_dim)
        # Decode: project to (B, L_pred * N) then reshape
        out = self.forecast_head(h)           # (B, L_pred * N)
        return out.view(B, L_pred, N)


# ---------------------------------------------------------------------------
# Linear baseline
# ---------------------------------------------------------------------------

class BaselineLinear(_BaselineModel):
    """Flatten + Linear projection baseline."""

    def __init__(self, args, dropout=0):
        super().__init__(args, dropout)
        C = args.C       # number of variables
        M = args.npatch or 1  # number of patches
        flat_dim = M * C
        self.flatten_proj = nn.Sequential(
            nn.Linear(flat_dim, self.hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self._build_heads()
        # Forecasting head (only used if task == forecasting)
        self._C = C
        self._M = M

    def _build_forecast_head(self, L_pred):
        if not hasattr(self, "forecast_head"):
            self.forecast_head = nn.Linear(
                self.hid_dim, L_pred * self._C
            ).to(self.device)

    def _encode(self, X_reg):
        # X_reg: (B, M, N)
        B = X_reg.size(0)
        x = X_reg.reshape(B, -1)          # (B, M*N)
        return self.flatten_proj(x)        # (B, hid_dim)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None,
                    text_var_data=None, text_var_tp=None, text_var_mask=None):
        B, M, L_in, N = X.shape
        L_pred = time_steps_to_predict.shape[-1]
        self._build_forecast_head(L_pred)
        return super().forecasting(
            time_steps_to_predict, X, truth_time_steps, mask,
            text_var_data, text_var_tp, text_var_mask,
        )


# ---------------------------------------------------------------------------
# LSTM baseline
# ---------------------------------------------------------------------------

class BaselineLSTM(_BaselineModel):
    """2-layer LSTM over the patch sequence."""

    def __init__(self, args, dropout=0):
        super().__init__(args, dropout)
        C = args.C
        M = args.npatch or 1
        self.lstm = nn.LSTM(
            input_size=C,
            hidden_size=self.hid_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
        )
        self._build_heads()
        self._C = C
        self._M = M

    def _build_forecast_head(self, L_pred):
        if not hasattr(self, "forecast_head"):
            self.forecast_head = nn.Linear(
                self.hid_dim, L_pred * self._C
            ).to(self.device)

    def _encode(self, X_reg):
        # X_reg: (B, M, N) — treat M as sequence length
        out, _ = self.lstm(X_reg)       # (B, M, hid_dim)
        return out[:, -1, :]            # (B, hid_dim)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None,
                    text_var_data=None, text_var_tp=None, text_var_mask=None):
        B, M, L_in, N = X.shape
        L_pred = time_steps_to_predict.shape[-1]
        self._build_forecast_head(L_pred)
        return super().forecasting(
            time_steps_to_predict, X, truth_time_steps, mask,
            text_var_data, text_var_tp, text_var_mask,
        )


# ---------------------------------------------------------------------------
# Transformer baseline
# ---------------------------------------------------------------------------

class BaselineTransformer(_BaselineModel):
    """Transformer encoder over the patch sequence."""

    def __init__(self, args, dropout=0):
        super().__init__(args, dropout)
        C = args.C
        M = args.npatch or 1
        n_heads = getattr(args, "n_heads", 1)
        # Ensure hid_dim is divisible by n_heads
        if self.hid_dim % n_heads != 0:
            n_heads = 1

        self.input_proj = nn.Linear(C, self.hid_dim)
        self.pos_enc = PositionalEncoding(self.hid_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hid_dim,
            nhead=n_heads,
            dim_feedforward=self.hid_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self._build_heads()
        self._C = C
        self._M = M

    def _build_forecast_head(self, L_pred):
        if not hasattr(self, "forecast_head"):
            self.forecast_head = nn.Linear(
                self.hid_dim, L_pred * self._C
            ).to(self.device)

    def _encode(self, X_reg):
        # X_reg: (B, M, N)
        x = self.input_proj(X_reg)      # (B, M, hid_dim)
        x = self.pos_enc(x)
        x = self.transformer(x)         # (B, M, hid_dim)
        return x[:, -1, :]              # (B, hid_dim)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None,
                    text_var_data=None, text_var_tp=None, text_var_mask=None):
        B, M, L_in, N = X.shape
        L_pred = time_steps_to_predict.shape[-1]
        self._build_forecast_head(L_pred)
        return super().forecasting(
            time_steps_to_predict, X, truth_time_steps, mask,
            text_var_data, text_var_tp, text_var_mask,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

BASELINE_MODELS = {
    "Linear": BaselineLinear,
    "LSTM": BaselineLSTM,
    "Transformer": BaselineTransformer,
}


def create_baseline_model(args, dropout=0):
    """Instantiate a baseline model by name (args.model)."""
    cls = BASELINE_MODELS.get(args.model)
    if cls is None:
        raise ValueError(
            f"Unknown baseline model: {args.model}. "
            f"Available: {list(BASELINE_MODELS.keys())}"
        )
    return cls(args, dropout=dropout)
