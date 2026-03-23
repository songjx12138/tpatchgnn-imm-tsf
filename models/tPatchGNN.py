import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

class nconv(nn.Module):
	def __init__(self):
		super(nconv,self).__init__()

	def forward(self, x, A):
		# x (B, F, N, M)
		# A (B, M, N, N)
		x = torch.einsum('bfnm,bmnv->bfvm',(x,A)) # used
		# print(x.shape)
		return x.contiguous() # (B, F, N, M)

class linear(nn.Module):
	def __init__(self, c_in, c_out):
		super(linear,self).__init__()
		# self.mlp = nn.Linear(c_in, c_out)
		self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1,1), padding=(0,0), stride=(1,1), bias=True)

	def forward(self, x):
		# x (B, F, N, M)

		# return self.mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
		return self.mlp(x)
		
class gcn(nn.Module):
	def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
		super(gcn,self).__init__()
		self.nconv = nconv()
		c_in = (order*support_len+1)*c_in
		# c_in = (order*support_len)*c_in
		self.mlp = linear(c_in, c_out)
		self.dropout = dropout
		self.order = order

	def forward(self, x, support):
		# x (B, F, N, M)
		# a (B, M, N, N)
		out = [x]
		for a in support:
			x1 = self.nconv(x,a)
			out.append(x1)
			for k in range(2, self.order + 1):
				x2 = self.nconv(x1,a)
				out.append(x2)
				x1 = x2

		h = torch.cat(out, dim=1) # concat x and x_conv
		h = self.mlp(h)
		return F.relu(h)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        """
        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
	

class tPatchGNN(nn.Module):
	def __init__(self, args, supports = None, dropout = 0):
	
		super(tPatchGNN, self).__init__()
		self.device = args.device
		self.hid_dim = args.hid_dim
		self.history = float(max(getattr(args, "history", 1) or 1, 1))
		self.pred_window = float(max(getattr(args, "pred_window", 1) or 1, 1))
		self.pred_bins = max(int(round(self.pred_window)), 1)
		self.history_norm = self.history / max(self.history + self.pred_window, 1.0)
		self.hazard_loss_weight = float(getattr(args, "hazard_loss_weight", 0.7))
		self.aux_bce_weight = float(getattr(args, "aux_bce_weight", 0.2))
		self.d_txt = getattr(args, "d_txt", args.hid_dim)
		self.enable_text = bool(getattr(args, "enable_text", False))
		self.text_guided_graph = bool(getattr(args, "text_guided_graph", False) and self.enable_text)
		self.dbg_text_graph = bool(getattr(args, "dbg_text_graph", False) and self.text_guided_graph)
		if self.text_guided_graph and args.hid_dim < 2:
			raise ValueError("--text_guided_graph requires hid_dim >= 2.")
		self.base_N = args.C
		self.N = self.base_N
		self.M = args.npatch
		self.max_text_vars = int(getattr(args, "max_text_vars", 1))
		self.batch_size = None
		self.supports = supports
		self.n_layer = args.nlayer
		self._dbg_once_keys = set()

		### Intra-time series modeling ## 
		## Time embedding
		self.te_scale = nn.Linear(1, 1)
		self.te_periodic = nn.Linear(1, args.te_dim-1)

		## TTCN
		self.event_feat_dim = 1
		input_dim = self.event_feat_dim + args.te_dim
		ttcn_dim = args.hid_dim - 1
		self.ttcn_dim = ttcn_dim
		self.Filter_Generators = nn.Sequential(
				nn.Linear(input_dim, ttcn_dim, bias=True),
				nn.ReLU(inplace=True),
				nn.Linear(ttcn_dim, ttcn_dim, bias=True),
				nn.ReLU(inplace=True),
				nn.Linear(ttcn_dim, input_dim*ttcn_dim, bias=True))
		self.T_bias = nn.Parameter(torch.randn(1, ttcn_dim))
		
		d_model = args.hid_dim
		## Transformer
		self.ADD_PE = PositionalEncoding(d_model) 
		self.transformer_encoder = nn.ModuleList()
		for _ in range(self.n_layer):
			encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=args.n_heads, batch_first=True)
			self.transformer_encoder.append(nn.TransformerEncoder(encoder_layer, num_layers=args.tf_layer))			

		### Inter-time series modeling ###
		self.supports_len = 0
		if supports is not None:
			self.supports_len += len(supports)

		nodevec_dim = args.node_dim
		self.nodevec_dim = nodevec_dim
		if supports is None:
			self.supports = []

		self.nodevec1 = nn.Parameter(torch.randn(self.base_N, nodevec_dim, device=self.device), requires_grad=True)
		self.nodevec2 = nn.Parameter(torch.randn(nodevec_dim, self.base_N, device=self.device), requires_grad=True)

		self.nodevec_linear1 = nn.ModuleList()
		self.nodevec_linear2 = nn.ModuleList()
		self.nodevec_gate1 = nn.ModuleList()
		self.nodevec_gate2 = nn.ModuleList()
		for _ in range(self.n_layer):
			self.nodevec_linear1.append(nn.Linear(args.hid_dim, nodevec_dim))
			self.nodevec_linear2.append(nn.Linear(args.hid_dim, nodevec_dim))
			self.nodevec_gate1.append(nn.Sequential(
				nn.Linear(args.hid_dim+nodevec_dim, 1),
				nn.Tanh(),
				nn.ReLU()))
			self.nodevec_gate2.append(nn.Sequential(
				nn.Linear(args.hid_dim+nodevec_dim, 1),
				nn.Tanh(),
				nn.ReLU()))

		if self.text_guided_graph:
			self.numeric_value_proj = nn.Sequential(
				nn.Linear(1, args.hid_dim),
				nn.LayerNorm(args.hid_dim),
				nn.Dropout(args.dropout),
			)
			self.text_value_proj = nn.Sequential(
				nn.Linear(self.d_txt, args.hid_dim),
				nn.LayerNorm(args.hid_dim),
				nn.Dropout(args.dropout),
			)
			self.cross_modal_gate = nn.Linear(args.hid_dim * 2, args.hid_dim)
			self.text_alpha = nn.Parameter(torch.tensor(0.1))
			self.text_nodevec1 = nn.Parameter(
				torch.randn(self.max_text_vars, nodevec_dim, device=self.device),
				requires_grad=True,
			)
			self.text_nodevec2 = nn.Parameter(
				torch.randn(nodevec_dim, self.max_text_vars, device=self.device),
				requires_grad=True,
			)
		else:
			self.numeric_value_proj = None
			self.text_value_proj = None
			self.cross_modal_gate = None
			self.text_alpha = None
			self.text_nodevec1 = None
			self.text_nodevec2 = None
			
		self.supports_len +=1

		self.gconv = nn.ModuleList() # gragh conv
		for _ in range(self.n_layer):
			self.gconv.append(gcn(d_model, d_model, dropout, support_len=self.supports_len, order=args.hop))

		### Encoder output layer ###
		self.outlayer = args.outlayer
		enc_dim = args.hid_dim
		if(self.outlayer == "Linear"):
			self.temporal_agg = nn.Sequential(
				nn.Linear(args.hid_dim*self.M, enc_dim))
		
		elif(self.outlayer == "CNN"):
			self.temporal_agg = nn.Sequential(
				nn.Conv1d(d_model, enc_dim, kernel_size=self.M))

		### Decoder ###
		self.decoder = nn.Sequential(
			nn.Linear(enc_dim+args.te_dim, args.hid_dim),
			nn.ReLU(inplace=True),
			nn.Linear(args.hid_dim, args.hid_dim),
			nn.ReLU(inplace=True),
			nn.Linear(args.hid_dim, 1)
			)

		### Classification head (optional) ###
		self.n_labels = getattr(args, "n_labels", 0)
		self.use_risk_head = bool(
			self.n_labels == 1 and getattr(args, "task_level", None) == "window_cls"
		)
		if self.n_labels > 0:
			self.cls_head = nn.Sequential(
				nn.Linear(enc_dim, args.hid_dim),
				nn.ReLU(inplace=True),
				nn.Dropout(dropout),
				nn.Linear(args.hid_dim, self.n_labels),
			)
			if self.use_risk_head:
				self.cls_var_stat_proj = nn.Sequential(
					nn.Linear(3, args.hid_dim),
					nn.LayerNorm(args.hid_dim),
					nn.ReLU(inplace=True),
				)
				self.cls_var_attn = nn.Linear(args.hid_dim, 1)
				self.cls_patch_stat_proj = nn.Sequential(
					nn.Linear(3, args.hid_dim),
					nn.LayerNorm(args.hid_dim),
					nn.ReLU(inplace=True),
				)
				cls_encoder_layer = nn.TransformerEncoderLayer(
					d_model=args.hid_dim,
					nhead=max(1, getattr(args, "n_heads", 1)),
					batch_first=True,
				)
				self.cls_patch_encoder = nn.TransformerEncoder(
					cls_encoder_layer, num_layers=1
				)
				coarse_encoder_layer = nn.TransformerEncoderLayer(
					d_model=args.hid_dim,
					nhead=max(1, getattr(args, "n_heads", 1)),
					batch_first=True,
				)
				self.cls_coarse_encoder = nn.TransformerEncoder(
					coarse_encoder_layer, num_layers=1
				)
				self.cls_patch_pos = nn.Parameter(
					torch.randn(1, self.M, args.hid_dim, device=self.device) * 0.02
				)
				self.cls_coarse_pos = nn.Parameter(
					torch.randn(1, max(self.M - 1, 1), args.hid_dim, device=self.device) * 0.02
				)
				self.cls_patch_attn = nn.Linear(args.hid_dim, 1)
				self.cls_coarse_attn = nn.Linear(args.hid_dim, 1)
				self.cls_recency_scale = nn.Parameter(torch.tensor(1.0, device=self.device))
				self.cls_multiscale_proj = nn.Sequential(
					nn.Linear(args.hid_dim * 3, args.hid_dim),
					nn.LayerNorm(args.hid_dim),
					nn.ReLU(inplace=True),
					nn.Dropout(dropout),
				)
				self.hazard_head = nn.Sequential(
					nn.Linear(args.hid_dim, args.hid_dim),
					nn.ReLU(inplace=True),
					nn.Dropout(dropout),
					nn.Linear(args.hid_dim, self.pred_bins),
				)
			else:
				self.cls_var_stat_proj = None
				self.cls_var_attn = None
				self.cls_patch_stat_proj = None
				self.cls_patch_encoder = None
				self.cls_coarse_encoder = None
				self.cls_patch_pos = None
				self.cls_coarse_pos = None
				self.cls_patch_attn = None
				self.cls_coarse_attn = None
				self.cls_recency_scale = None
				self.cls_multiscale_proj = None
				self.hazard_head = None

		### Regression head (optional) ###
		self.is_regression = bool(getattr(args, "is_regression", False))
		if self.is_regression:
			self.reg_head = nn.Sequential(
				nn.Linear(enc_dim, args.hid_dim),
				nn.ReLU(inplace=True),
				nn.Dropout(dropout),
				nn.Linear(args.hid_dim, 1),
			)
		
	def LearnableTE(self, tt):
		# tt: (N*M*B, L, 1)
		out1 = self.te_scale(tt)
		out2 = torch.sin(self.te_periodic(tt))
		return torch.cat([out1, out2], -1)
	
	def TTCN(self, X_int, mask_X):
		# X_int: shape (B*N*M, L, F)
		# mask_X: shape (B*N*M, L, 1)

		N, Lx, _ = mask_X.shape
		Filter = self.Filter_Generators(X_int) # (N, Lx, F_in*ttcn_dim)
		Filter_mask = Filter * mask_X + (1 - mask_X) * (-1e8)
		# normalize along with sequence dimension
		Filter_seqnorm = F.softmax(Filter_mask, dim=-2)  # (N, Lx, F_in*ttcn_dim)
		Filter_seqnorm = Filter_seqnorm.view(N, Lx, self.ttcn_dim, -1) # (N, Lx, ttcn_dim, F_in)
		X_int_broad = X_int.unsqueeze(dim=-2).repeat(1, 1, self.ttcn_dim, 1)
		ttcn_out = torch.sum(torch.sum(X_int_broad * Filter_seqnorm, dim=-3), dim=-1) # (N, ttcn_dim)
		h_t = torch.relu(ttcn_out + self.T_bias) # (N, ttcn_dim)
		return h_t

	def _dbg_log_once(self, key, message):
		if self.dbg_text_graph and (key not in self._dbg_once_keys):
			print(message)
			self._dbg_once_keys.add(key)

	def _check_finite(self, name, x):
		if self.dbg_text_graph and (not torch.isfinite(x).all()):
			raise ValueError(f"[DBG text_graph] Non-finite values detected in {name}.")

	def _pad_seq_len(self, x, target_len):
		cur_len = x.size(3)
		if cur_len == target_len:
			return x
		if cur_len > target_len:
			return x[:, :, :, :target_len]
		pad_shape = list(x.shape)
		pad_shape[3] = target_len - cur_len
		pad = torch.zeros(*pad_shape, dtype=x.dtype, device=x.device)
		return torch.cat([x, pad], dim=3)

	def _build_nodevec(self, B, M, n_numeric_vars, n_text_vars, dtype, device):
		if n_numeric_vars > self.base_N:
			raise ValueError(f"n_numeric_vars={n_numeric_vars} exceeds base_N={self.base_N}")
		nodevec1 = self.nodevec1[:n_numeric_vars].to(device=device, dtype=dtype)
		nodevec2 = self.nodevec2[:, :n_numeric_vars].to(device=device, dtype=dtype)

		if n_text_vars > 0:
			if not self.text_guided_graph or self.text_nodevec1 is None or self.text_nodevec2 is None:
				raise ValueError("Text node embeddings are not initialized.")
			if n_text_vars > self.max_text_vars:
				raise ValueError(
					f"n_text_vars={n_text_vars} exceeds max_text_vars={self.max_text_vars}. "
					"Set a larger --max_text_vars for this configuration."
				)
			text_nodevec1 = self.text_nodevec1[:n_text_vars].to(device=device, dtype=dtype)
			text_nodevec2 = self.text_nodevec2[:, :n_text_vars].to(device=device, dtype=dtype)
			nodevec1 = torch.cat([nodevec1, text_nodevec1], dim=0)
			nodevec2 = torch.cat([nodevec2, text_nodevec2], dim=1)

		n_total = n_numeric_vars + n_text_vars
		nodevec1 = nodevec1.view(1, 1, n_total, self.nodevec_dim).repeat(B, M, 1, 1)
		nodevec2 = nodevec2.view(1, 1, self.nodevec_dim, n_total).repeat(B, M, 1, 1)
		return nodevec1, nodevec2

	def _prepare_text_as_nodes_inputs(self, X, truth_time_steps, mask, text_var_data, text_var_tp, text_var_mask):
		B, M, L_num, N_num = X.shape
		if text_var_data.dim() != 5:
			raise ValueError(
				f"text_var_data must be 5D [B, M, L_txt, N_txt, d_txt], got {text_var_data.shape}"
			)

		B_txt, M_txt, L_txt, N_txt, d_txt = text_var_data.shape
		if B_txt != B or M_txt != M:
			raise ValueError(
				f"text_var_data batch/patch mismatch: expected ({B}, {M}), got ({B_txt}, {M_txt})"
			)
		if d_txt != self.d_txt:
			raise ValueError(f"text embedding dim mismatch: expected {self.d_txt}, got {d_txt}")
		if text_var_tp.shape != (B, M, L_txt, N_txt):
			raise ValueError(
				f"text_var_tp must have shape {(B, M, L_txt, N_txt)}, got {tuple(text_var_tp.shape)}"
			)
		if text_var_mask.shape != (B, M, L_txt, N_txt):
			raise ValueError(
				f"text_var_mask must have shape {(B, M, L_txt, N_txt)}, got {tuple(text_var_mask.shape)}"
			)

		x_num = X.permute(0, 3, 1, 2).reshape(-1, L_num, 1)  # (B*N_num*M, L_num, 1)
		t_num = truth_time_steps.permute(0, 3, 1, 2).reshape(-1, L_num, 1)  # (B*N_num*M, L_num, 1)
		m_num = mask.permute(0, 3, 1, 2).reshape(-1, L_num, 1)  # (B*N_num*M, L_num, 1)
		te_num = self.LearnableTE(t_num)  # (B*N_num*M, L_num, F_te)
		x_num_in = torch.cat([x_num, te_num], dim=-1)
		self._check_finite("x_num_in", x_num_in)

		x_txt = text_var_data.permute(0, 3, 1, 2, 4).to(device=X.device, dtype=X.dtype)  # (B, N_txt, M, L_txt, d_txt)
		t_txt = text_var_tp.permute(0, 3, 1, 2).unsqueeze(-1).to(device=X.device, dtype=X.dtype)  # (B, N_txt, M, L_txt, 1)
		m_txt = text_var_mask.permute(0, 3, 1, 2).unsqueeze(-1).to(device=X.device, dtype=X.dtype)  # (B, N_txt, M, L_txt, 1)
		te_txt = self.LearnableTE(t_txt.reshape(-1, L_txt, 1)).view(B, N_txt, M, L_txt, -1)
		x_txt_proj = self.text_value_proj(x_txt)
		x_txt_in = torch.cat([x_txt_proj, te_txt], dim=-1)
		self._check_finite("x_txt_proj", x_txt_proj)
		self._check_finite("x_txt_in", x_txt_in)

		if self.dbg_text_graph:
			with torch.no_grad():
				txt_norm = x_txt_proj.norm(dim=-1)
				txt_mean = float(txt_norm.mean().item())
				txt_std = float(txt_norm.std(unbiased=False).item())
				txt_q90 = float(torch.quantile(txt_norm.reshape(-1), 0.9).item())
			self._dbg_log_once(
				"txt_proj_norm",
				f"[DBG text_graph][norm] text_proj(mean/std/q90)=({txt_mean:.4f}/{txt_std:.4f}/{txt_q90:.4f})",
			)

		# Convert event-level text embeddings into real graph nodes: one text node per
		# text variable, with a patch-level representation plus an explicit mask bit.
		text_core_events = x_txt_proj[..., :-1]
		txt_weighted_sum = (text_core_events * m_txt).sum(dim=3)  # (B, N_txt, M, hid_dim-1)
		txt_count = m_txt.sum(dim=3).clamp_min(1.0)  # (B, N_txt, M, 1)
		text_patch_mask = (m_txt.sum(dim=3) > 0).to(x_txt_proj.dtype)  # (B, N_txt, M, 1)
		text_patch_core = txt_weighted_sum / txt_count
		text_patch_repr = torch.cat([text_patch_core, text_patch_mask], dim=-1)
		text_patch_repr = text_patch_repr * text_patch_mask

		# Keep a patch-level summary context for gating numeric nodes.
		text_ctx_sum = (text_patch_repr * text_patch_mask).sum(dim=1)  # (B, M, hid_dim)
		text_ctx_count = text_patch_mask.sum(dim=1).clamp_min(1.0)  # (B, M, 1)
		text_ctx = text_ctx_sum / text_ctx_count
		text_ctx_mask = (text_patch_mask.sum(dim=1).squeeze(-1) > 0).to(text_ctx.dtype)  # (B, M)
		text_ctx = text_ctx * text_ctx_mask.unsqueeze(-1)
		self._check_finite("text_ctx", text_ctx)
		self._check_finite("text_patch_repr", text_patch_repr)

		if self.dbg_text_graph:
			with torch.no_grad():
				nonempty_ratio = float(text_ctx_mask.mean().item())
				empty_ratio = 1.0 - nonempty_ratio
			self._dbg_log_once(
				"text_empty_patch",
				f"[DBG text_graph][mask] text patch nonempty ratio={nonempty_ratio:.4f}, empty ratio={empty_ratio:.4f}",
			)

		self._check_finite("x_num_in_reshape", x_num_in)
		self._check_finite("m_num_reshape", m_num)

		return x_num_in, m_num, N_num, text_patch_repr, int(N_txt), text_ctx, text_ctx_mask

	def _encode_backbone(
		self,
		X,
		truth_time_steps,
		mask=None,
		text_var_data=None,
		text_var_tp=None,
		text_var_mask=None,
		return_patch=False,
	):
		"""
		Run the shared tPatchGNN encoder and optionally keep the patch tokens before
		the final temporal aggregation. Classification uses those patch tokens to
		build a recency-aware risk decoder without changing the forecasting path.
		"""
		B, _, L_in, N = X.shape
		self.batch_size = B

		if self.text_guided_graph:
			if text_var_data is None or text_var_tp is None or text_var_mask is None:
				raise ValueError("Missing text variable tensors for --text_guided_graph.")
			X_num, mask_num, n_numeric_vars, text_patch_repr, n_text_vars, text_ctx, text_ctx_mask = self._prepare_text_as_nodes_inputs(
				X,
				truth_time_steps,
				mask,
				text_var_data,
				text_var_tp,
				text_var_mask,
			)
			encoded = self.IMTS_Model(
				X_num,
				mask_num,
				n_vars=n_numeric_vars + n_text_vars,
				n_numeric_vars=n_numeric_vars,
				n_text_vars=n_text_vars,
				text_patch_repr=text_patch_repr,
				text_ctx=text_ctx,
				text_ctx_mask=text_ctx_mask,
				return_patch=return_patch,
			)
			if return_patch:
				h, h_patch = encoded
				h = h[:, :n_numeric_vars]
				h_patch = h_patch[:, :n_numeric_vars]
				return h, h_patch
			return encoded[:, :n_numeric_vars]

		X_flat = X.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
		truth_time_steps_flat = truth_time_steps.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
		mask_flat = mask.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
		te_his = self.LearnableTE(truth_time_steps_flat)
		X_flat = torch.cat([X_flat, te_his], dim=-1)
		return self.IMTS_Model(
			X_flat,
			mask_flat,
			n_vars=N,
			n_numeric_vars=N,
			n_text_vars=0,
			return_patch=return_patch,
		)

	def _compute_patch_statistics(self, truth_time_steps, mask):
		mask_float = mask.to(dtype=torch.float32)
		counts = mask_float.sum(dim=2).permute(0, 2, 1)  # (B, N, M)
		nonempty = counts > 0
		count_norm = counts / counts.amax(dim=-1, keepdim=True).clamp_min(1.0)

		last_tp = truth_time_steps.masked_fill(mask == 0, 0.0).max(dim=2).values.permute(0, 2, 1)
		recency = (last_tp / max(self.history_norm, 1e-6)).clamp(0.0, 1.0)
		recency = recency * nonempty.to(recency.dtype)
		return count_norm, recency, nonempty

	def _aggregate_patch_sequence(self, tokens, valid_mask, recency, pos_embed, encoder, attn_layer):
		if tokens.size(1) == 0:
			return tokens.new_zeros(tokens.size(0), self.hid_dim)

		x = tokens + pos_embed[:, :tokens.size(1)].to(dtype=tokens.dtype, device=tokens.device)
		x = encoder(x, src_key_padding_mask=~valid_mask)
		scores = attn_layer(x).squeeze(-1)
		scores = scores + F.softplus(self.cls_recency_scale) * recency
		scores = scores.masked_fill(~valid_mask, -1e4)
		attn = torch.softmax(scores, dim=-1)
		attn = attn * valid_mask.to(dtype=attn.dtype)
		attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
		return (attn.unsqueeze(-1) * x).sum(dim=1)

	def _risk_from_encoded_state(self, h, h_patch, truth_time_steps, mask):
		if not self.use_risk_head:
			h_pooled = h.mean(dim=1)
			logits = self.cls_head(h_pooled)
			return logits, h_pooled, {}

		count_norm, recency, patch_nonempty = self._compute_patch_statistics(
			truth_time_steps, mask
		)
		var_stats = torch.stack(
			[count_norm, recency, patch_nonempty.to(dtype=h_patch.dtype)],
			dim=-1,
		)
		var_tokens = h_patch + self.cls_var_stat_proj(var_stats)
		var_scores = self.cls_var_attn(var_tokens).squeeze(-1)
		var_scores = var_scores.masked_fill(~patch_nonempty, -1e4)
		var_attn = torch.softmax(var_scores, dim=1)
		var_attn = var_attn * patch_nonempty.to(dtype=var_attn.dtype)
		var_attn = var_attn / var_attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
		patch_tokens = (var_attn.unsqueeze(-1) * var_tokens).sum(dim=1)  # (B, M, H)

		patch_level_stats = torch.stack(
			[
				count_norm.mean(dim=1),
				recency.max(dim=1).values,
				patch_nonempty.to(dtype=h_patch.dtype).mean(dim=1),
			],
			dim=-1,
		)
		patch_tokens = patch_tokens + self.cls_patch_stat_proj(patch_level_stats)
		fine_mask = patch_nonempty.any(dim=1)
		fine_recency = patch_level_stats[..., 1]
		fine_summary = self._aggregate_patch_sequence(
			patch_tokens,
			fine_mask,
			fine_recency,
			self.cls_patch_pos,
			self.cls_patch_encoder,
			self.cls_patch_attn,
		)

		if patch_tokens.size(1) > 1:
			coarse_tokens = 0.5 * (patch_tokens[:, :-1] + patch_tokens[:, 1:])
			coarse_mask = fine_mask[:, :-1] | fine_mask[:, 1:]
			coarse_recency = torch.maximum(fine_recency[:, :-1], fine_recency[:, 1:])
			coarse_summary = self._aggregate_patch_sequence(
				coarse_tokens,
				coarse_mask,
				coarse_recency,
				self.cls_coarse_pos,
				self.cls_coarse_encoder,
				self.cls_coarse_attn,
			)
		else:
			coarse_summary = fine_summary

		h_pooled = h.mean(dim=1)
		risk_context = self.cls_multiscale_proj(
			torch.cat([h_pooled, fine_summary, coarse_summary], dim=-1)
		)
		hazard_logits = self.hazard_head(risk_context)
		hazard_probs = torch.sigmoid(hazard_logits)
		event_prob = 1.0 - torch.prod(1.0 - hazard_probs, dim=-1, keepdim=True)
		event_logit = torch.logit(event_prob.clamp(1e-5, 1.0 - 1e-5))
		aux_global_logit = self.cls_head(h_pooled)
		return event_logit, h_pooled, {
			"hazard_logits": hazard_logits,
			"hazard_probs": hazard_probs,
			"event_prob": event_prob,
			"aux_global_logit": aux_global_logit,
		}

	def IMTS_Model(
		self,
		x,
		mask_X,
		n_vars,
		n_numeric_vars,
		n_text_vars,
		text_patch_repr=None,
		text_ctx=None,
		text_ctx_mask=None,
		return_patch=False,
	):
		"""
		x (B*N*M, L, F)
		mask_X (B*N*M, L, 1)
		"""
		# mask for the patch
		mask_patch = (mask_X.sum(dim=1) > 0) # (B*N*M, 1)

		### TTCN for patch modeling ###
		x_patch = self.TTCN(x, mask_X) # (B*N*M, hid_dim-1)
		x_patch = torch.cat([x_patch, mask_patch],dim=-1) # (B*N*M, hid_dim)
		x_patch = x_patch.view(self.batch_size, n_numeric_vars, self.M, -1) # (B, N_num, M, hid_dim)
		B = x_patch.shape[0]
		if text_patch_repr is not None:
			expected = (B, n_text_vars, self.M, self.hid_dim)
			if tuple(text_patch_repr.shape) != expected:
				raise ValueError(
					f"text_patch_repr shape mismatch: expected {expected}, got {tuple(text_patch_repr.shape)}"
				)
			text_patch_repr = text_patch_repr.to(device=x_patch.device, dtype=x_patch.dtype)
			x_patch = torch.cat([x_patch, text_patch_repr], dim=1)
		B, N, M, D = x_patch.shape
		self._check_finite("x_patch", x_patch)

		x = x_patch
		for layer in range(self.n_layer):

			if(layer > 0): # residual
				x_last = x.clone()
				
			### Transformer for temporal modeling ###
			x = x.reshape(B*N, M, -1) # (B*N, M, F)
			x = self.ADD_PE(x)
			x = self.transformer_encoder[layer](x).view(x_patch.shape) # (B, N, M, F)
			self._check_finite(f"transformer_out_layer{layer}", x)

			if text_ctx is not None:
				x_numeric = x[:, :n_numeric_vars]
				text_expand = text_ctx.unsqueeze(1).expand(B, n_numeric_vars, M, self.hid_dim)
				if text_ctx_mask is not None:
					text_expand = text_expand * text_ctx_mask.view(B, 1, M, 1)
				gate = torch.sigmoid(self.cross_modal_gate(torch.cat([x_numeric, text_expand], dim=-1)))
				x_numeric = x_numeric + self.text_alpha * gate * text_expand
				if n_text_vars > 0:
					x = torch.cat([x_numeric, x[:, n_numeric_vars:]], dim=1)
				else:
					x = x_numeric
				self._check_finite(f"text_conditioned_x_layer{layer}", x)
				if self.dbg_text_graph:
					with torch.no_grad():
						gate_mean = float(gate.mean().item())
						gate_std = float(gate.std(unbiased=False).item())
						alpha = float(self.text_alpha.item())
						num_norm = x_numeric.norm(dim=-1)
						txt_norm = text_ctx.norm(dim=-1)
						num_mean = float(num_norm.mean().item())
						num_std = float(num_norm.std(unbiased=False).item())
						txt_mean = float(txt_norm.mean().item())
						txt_std = float(txt_norm.std(unbiased=False).item())
						num_q90 = float(torch.quantile(num_norm.reshape(-1), 0.9).item())
						txt_q90 = float(torch.quantile(txt_norm.reshape(-1), 0.9).item())
					self._dbg_log_once(
						f"text_gate_layer{layer}",
						f"[DBG text_graph][gate][layer={layer}] alpha={alpha:.6f} gate(mean/std)=({gate_mean:.6f}/{gate_std:.6f})",
					)
					self._dbg_log_once(
						f"num_txt_norm_layer{layer}",
						f"[DBG text_graph][norm][layer={layer}] num_patch(mean/std/q90)=({num_mean:.4f}/{num_std:.4f}/{num_q90:.4f}) "
						f"text_ctx(mean/std/q90)=({txt_mean:.4f}/{txt_std:.4f}/{txt_q90:.4f})",
					)

			### GNN for inter-time series modeling ###
			### time-adaptive graph structure learning ###
			nodevec1, nodevec2 = self._build_nodevec(
				B=B,
				M=M,
				n_numeric_vars=n_numeric_vars,
				n_text_vars=n_text_vars,
				dtype=x.dtype,
				device=x.device,
			)
			x_gate1 = self.nodevec_gate1[layer](torch.cat([x, nodevec1.permute(0, 2, 1, 3)], dim=-1))
			x_gate2 = self.nodevec_gate2[layer](torch.cat([x, nodevec2.permute(0, 3, 1, 2)], dim=-1))
			x_p1 = x_gate1 * self.nodevec_linear1[layer](x) # (B, M, N, 10)
			x_p2 = x_gate2 * self.nodevec_linear2[layer](x) # (B, M, N, 10)
			nodevec1 = nodevec1 + x_p1.permute(0,2,1,3) # (B, M, N, 10)
			nodevec2 = nodevec2 + x_p2.permute(0,2,3,1) # (B, M, 10, N)

			adp = F.softmax(F.relu(torch.matmul(nodevec1, nodevec2)), dim=-1) # (B, M, N, N) used
			self._check_finite(f"adp_layer{layer}", adp)
			if self.dbg_text_graph:
				with torch.no_grad():
					n_nodes = adp.size(-1)
					p = adp.clamp_min(1e-12)
					entropy = float((-(p * torch.log(p)).sum(dim=-1)).mean().item())
					topk = min(3, n_nodes)
					topk_share = float(adp.topk(topk, dim=-1).values.sum(dim=-1).mean().item())
					adp_mean = float(adp.mean().item())
					adp_min = float(adp.min().item())
					adp_max = float(adp.max().item())
				self._dbg_log_once(
					f"adp_stats_layer{layer}",
					f"[DBG text_graph][A][layer={layer}] N={n_nodes} mean/min/max="
					f"({adp_mean:.6f}/{adp_min:.6f}/{adp_max:.6f}) entropy={entropy:.6f} top{topk}_share={topk_share:.6f}",
				)
			new_supports = self.supports + [adp]

			# input x shape (B, F, N, M)
			x = self.gconv[layer](x.permute(0,3,1,2), new_supports) # (B, F, N, M)
			x = x.permute(0, 2, 3, 1) # (B, N, M, F)
			self._check_finite(f"gconv_out_layer{layer}", x)

			if(layer > 0): # residual addition
				x = x_last + x 

		patch_tokens = x

		### Output layer ###
		if(self.outlayer == "CNN"):
			x = x.reshape(self.batch_size*n_vars, self.M, -1).permute(0, 2, 1) # (B*N, F, M)
			x = self.temporal_agg(x) # (B*N, F, M) -> (B*N, F, 1)
			x = x.view(self.batch_size, n_vars, -1) # (B, N, F)

		elif(self.outlayer == "Linear"):
			x = x.reshape(self.batch_size, n_vars, -1) # (B, N, M*F)
			x = self.temporal_agg(x) # (B, N, hid_dim)

		if return_patch:
			return x, patch_tokens
		return x

	def forecasting(
		self,
		time_steps_to_predict,
		X,
		truth_time_steps,
		mask=None,
		text_var_data=None,
		text_var_tp=None,
		text_var_mask=None,
	):
		
		""" 
		time_steps_to_predict (B, L) [0, 1]
		X (B, M, L, N) 
		truth_time_steps (B, M, L, N) [0, 1]
		mask (B, M, L, N)

		To ====>

        X (B*N*M, L, 1)
		truth_time_steps (B*N*M, L, 1)
        mask_X (B*N*M, L, 1)
        """

		B = X.shape[0]
		h = self._encode_backbone(
			X,
			truth_time_steps,
			mask=mask,
			text_var_data=text_var_data,
			text_var_tp=text_var_tp,
			text_var_mask=text_var_mask,
			return_patch=False,
		)
		N_out = h.shape[1]

		""" Decoder """
		L_pred = time_steps_to_predict.shape[-1]
		h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1) # (B, N, Lp, F)
		time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N_out, 1, 1) # (B, N, Lp, 1)
		te_pred = self.LearnableTE(time_steps_to_predict) # (B, N, Lp, F_te)

		h = torch.cat([h, te_pred], dim=-1) # (B, N, Lp, F)
		self._check_finite("decoder_input_h", h)

		# (B, N, Lp, F) -> (B, N, Lp, 1) -> (B, Lp, N)
		outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1)
		self._check_finite("decoder_outputs", outputs)

		return outputs # (B, Lp, N)

	def classify(
		self,
		X,
		truth_time_steps,
		mask=None,
		text_var_data=None,
		text_var_tp=None,
		text_var_mask=None,
		return_h=False,
		return_aux=False,
	):
		"""
		Classification using the same encoder as forecasting.
		Returns logits of shape (B, n_labels).
		If return_h=True, returns (logits, h_pooled) for late fusion.
		"""
		if self.use_risk_head:
			h, h_patch = self._encode_backbone(
				X,
				truth_time_steps,
				mask=mask,
				text_var_data=text_var_data,
				text_var_tp=text_var_tp,
				text_var_mask=text_var_mask,
				return_patch=True,
			)
			risk_logits, h_pooled, aux = self._risk_from_encoded_state(
				h, h_patch, truth_time_steps, mask
			)
			if return_h:
				base_logits = aux.get("aux_global_logit", self.cls_head(h_pooled))
				if return_aux:
					return base_logits, h_pooled, aux
				return base_logits, h_pooled
			if return_aux:
				return risk_logits, aux
			return risk_logits

		h = self._encode_backbone(
			X,
			truth_time_steps,
			mask=mask,
			text_var_data=text_var_data,
			text_var_tp=text_var_tp,
			text_var_mask=text_var_mask,
			return_patch=False,
		)
		h_pooled = h.mean(dim=1)
		logits = self.cls_head(h_pooled)
		if return_h:
			if return_aux:
				return logits, h_pooled, {}
			return logits, h_pooled
		if return_aux:
			return logits, {}
		return logits

	def regress(
		self,
		X,
		truth_time_steps,
		mask=None,
		text_var_data=None,
		text_var_tp=None,
		text_var_mask=None,
		return_h=False,
	):
		"""
		Regression using the same encoder as forecasting.
		Returns predictions of shape (B, 1).
		If return_h=True, returns (preds, h_pooled) for late fusion.
		"""
		h = self._encode_backbone(
			X,
			truth_time_steps,
			mask=mask,
			text_var_data=text_var_data,
			text_var_tp=text_var_tp,
			text_var_mask=text_var_mask,
			return_patch=False,
		)

		h_pooled = h.mean(dim=1)  # (B, hid_dim)
		pred = self.reg_head(h_pooled)  # (B, 1)
		if return_h:
			return pred, h_pooled
		return pred
