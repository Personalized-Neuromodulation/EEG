import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Optional, Tuple, Union


class DynamicsModel(nn.Module):
	"""State transition f(g, s) using a GRU cell with step size."""

	def __init__(self, latent_dim: int, stim_dim: int):
		super().__init__()
		self.stim_dim = stim_dim
		input_size = stim_dim if stim_dim > 0 else 1
		self.gru = nn.GRUCell(input_size=input_size, hidden_size=latent_dim)

	def forward(
		self,
		g: torch.Tensor,
		s: Optional[torch.Tensor] = None,
		dt: float = 1.0,
	) -> torch.Tensor:
		if self.stim_dim > 0 and s is not None:
			s_in = s
		else:
			batch_size = g.shape[0]
			s_in = torch.zeros(batch_size, 1, device=g.device, dtype=g.dtype)
		g_gru = self.gru(s_in, g)
		return g + dt * (g_gru - g)



class ObservationEncoder(nn.Module):
	"""Channel-aware encoder for band-power residual tensors."""

	def __init__(self, per_channel_dim: int, n_channels: int, embed_dim: int = 128):
		super().__init__()
		self.n_channels = int(n_channels)
		in_dim = int(per_channel_dim) * self.n_channels
		self.out = nn.Sequential(
			nn.Linear(in_dim, embed_dim),
			nn.GELU(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 3:
			raise ValueError(f"Expected residual shape (B,C,F), got {tuple(x.shape)}")
		b, c, _ = x.shape
		if c != self.n_channels:
			raise ValueError(f"Expected {self.n_channels} channels, got {c}")
		return self.out(x.reshape(b, -1))


class Decoder(nn.Module):
	"""Near-linear structured decoder D(g) -> (B, C, F)."""

	def __init__(
		self,
		latent_dim: int,
		n_channels: int,
		per_channel_dim: int,
		hidden_dim: int = 192,
	):
		super().__init__()
		self.n_channels = int(n_channels)
		self.per_channel_dim = int(per_channel_dim)
		rank_dim = max(16, int(hidden_dim))
		self.in_proj = nn.Linear(latent_dim, rank_dim)
		self.out_proj = nn.Linear(rank_dim, self.n_channels * self.per_channel_dim)

	def forward(self, g: torch.Tensor) -> torch.Tensor:
		x = self.out_proj(self.in_proj(g))
		return x.view(g.shape[0], self.n_channels, self.per_channel_dim)


class UpdateModule(nn.Module):
	"""Constrained update: delta = K(g) * r, K does not depend on x."""

	def __init__(
		self,
		latent_dim: int,
		obs_dim: int,
		gate_type: str = "diag",
		hidden_dim: int = 192,
		low_rank: int = 8,
	):
		super().__init__()
		if gate_type not in {"diag", "lowrank"}:
			raise ValueError("gate_type must be 'diag' or 'lowrank'")
		self.gate_type = gate_type
		self.low_rank = low_rank
		self.proj = nn.Linear(obs_dim, latent_dim, bias=True)
		if gate_type == "diag":
			gate_out = latent_dim
			self.gate_net = nn.Sequential(
				nn.Linear(latent_dim, hidden_dim),
				nn.Tanh(),
				nn.Linear(hidden_dim, gate_out),
				nn.Sigmoid(),
			)
		else:
			self.u_net = nn.Sequential(
				nn.Linear(latent_dim, hidden_dim),
				nn.Tanh(),
				nn.Linear(hidden_dim, latent_dim * low_rank),
			)
			self.v_net = nn.Sequential(
				nn.Linear(latent_dim, hidden_dim),
				nn.Tanh(),
				nn.Linear(hidden_dim, low_rank * latent_dim),
			)

	def forward(
		self, g: torch.Tensor, residual: torch.Tensor
	) -> Tuple[torch.Tensor, torch.Tensor]:
		res_proj = self.proj(residual)
		if self.gate_type == "lowrank":
			batch_size = g.shape[0]
			u = self.u_net(g).view(batch_size, -1, self.low_rank)
			v = self.v_net(g).view(batch_size, self.low_rank, -1)
			mid = torch.bmm(v, res_proj.unsqueeze(-1)).squeeze(-1)
			delta = torch.bmm(u, mid.unsqueeze(-1)).squeeze(-1)
		else:
			gate = self.gate_net(g)
			delta = gate * res_proj
		l_delta = (delta ** 2).mean()
		return delta, l_delta


class OnlineStateSpaceModel(nn.Module):
	"""Predict-update online model for closed-loop EEG control."""

	def __init__(
		self,
		latent_dim: int,
		obs_dim: Union[int, Tuple[int, int]],
		stim_dim: int = 0,
		obs_embed_dim: int = 128,
		hidden_dim: int = 192,
		decoder_hidden_dim: int = 192,
		gate_type: str = "diag",
		low_rank: int = 8,
		update_every: int = 1,
		burn_in_steps: int = 0,
		delta_clip: Optional[float] = None,
		spec_power: float = 0.5,
		spatial_lambda: float = 0.1,
		sample_rate: float = 100.0,
		dt: float = 1.0,
	):
		super().__init__()
		if isinstance(obs_dim, tuple):
			self._raw_obs_shape = (int(obs_dim[0]), int(obs_dim[1]))
			dummy = torch.zeros(1, self._raw_obs_shape[0], self._raw_obs_shape[1])
		else:
			self._raw_obs_shape = (int(obs_dim),)
			dummy = torch.zeros(1, self._raw_obs_shape[0])
		self._band_edges = (
			(0.5, 2.0),
			(2.0, 4.0),
			(4.0, 6.0),
			(6.0, 8.0),
			(8.0, 10.0),
			(10.0, 12.0),
			(12.0, 14.0),
			(14.0, 16.0),
			(16.0, 20.0),
			(20.0, 24.0),
			(24.0, 30.0),
		)
		self._spec_power = float(spec_power)
		self._spatial_lambda = float(spatial_lambda)
		self._sample_rate = float(sample_rate)
		dummy_spec_ch = self._obs_to_spec_per_channel(dummy)
		self._n_channels = int(dummy_spec_ch.shape[1])
		self._per_channel_spec_dim = int(dummy_spec_ch.shape[2])
		self._spec_dim = self._n_channels * self._per_channel_spec_dim
		self.dynamics = DynamicsModel(latent_dim, stim_dim)
		self.obs_encoder = ObservationEncoder(self._per_channel_spec_dim, self._n_channels, obs_embed_dim)
		self.decoder = Decoder(
			latent_dim,
			self._n_channels,
			self._per_channel_spec_dim,
			decoder_hidden_dim,
		)
		self.update_module = UpdateModule(
			latent_dim,
			obs_embed_dim,
			gate_type=gate_type,
			hidden_dim=hidden_dim,
			low_rank=low_rank,
		)
		self._g_state: Optional[torch.Tensor] = None
		self._latent_dim = latent_dim
		self._update_every = max(1, int(update_every))
		self._burn_in_steps = max(0, int(burn_in_steps))
		self._delta_clip = delta_clip
		self._dt = float(dt)

	def _obs_to_spec_per_channel(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim == 2:
			x_in = x.unsqueeze(1)
		elif x.ndim == 3:
			x_in = x
		else:
			raise ValueError(f"Expected input shape (B,T) or (B,C,T), got {tuple(x.shape)}")

		batch_size, n_channels, seq_len = x_in.shape
		x_flat = x_in.reshape(batch_size * n_channels, seq_len)
		xf = torch.fft.rfft(x_flat, dim=-1)
		power = (xf.abs() ** 2) / max(1, seq_len)
		freqs = torch.fft.rfftfreq(seq_len, d=1.0 / self._sample_rate).to(x.device)
		band_feats = []
		for f_low, f_high in self._band_edges:
			mask = (freqs >= f_low) & (freqs < f_high)
			if bool(mask.any()):
				bp = power[:, mask].mean(dim=-1)
			else:
				bp = torch.zeros(power.shape[0], device=x.device, dtype=x.dtype)
			bp = torch.log1p(bp.pow(self._spec_power))
			band_feats.append(bp)
		bands = torch.stack(band_feats, dim=-1)
		return bands.reshape(batch_size, n_channels, -1)

	def _vec_to_channel_spec(self, vec: torch.Tensor) -> torch.Tensor:
		if vec.ndim != 2:
			raise ValueError(f"Expected vec shape (B,D), got {tuple(vec.shape)}")
		if vec.shape[1] != self._spec_dim:
			m = min(vec.shape[1], self._spec_dim)
			pad = self._spec_dim - m
			vec = vec[:, :m]
			if pad > 0:
				vec = F.pad(vec, (0, pad))
		return vec.view(vec.shape[0], self._n_channels, self._per_channel_spec_dim)

	def _obs_to_spec(self, x: torch.Tensor) -> torch.Tensor:
		spec_ch = self._obs_to_spec_per_channel(x)
		return spec_ch.reshape(spec_ch.shape[0], -1)

	def observation_to_spectrum(self, x: torch.Tensor) -> torch.Tensor:
		return self._obs_to_spec(x)

	def observation_to_spectrum_per_channel(self, x: torch.Tensor) -> torch.Tensor:
		return self._obs_to_spec_per_channel(x)

	def _bandpower_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		"""Channel-aware band-power reconstruction + spatial covariance consistency."""
		target_ch = self._obs_to_spec_per_channel(target)
		if pred.ndim == 2:
			pred_ch = self._vec_to_channel_spec(pred)
		else:
			pred_ch = pred
		if pred_ch.shape != target_ch.shape:
			c = min(pred_ch.shape[1], target_ch.shape[1])
			f = min(pred_ch.shape[2], target_ch.shape[2])
			pred_ch = pred_ch[:, :c, :f]
			target_ch = target_ch[:, :c, :f]
		l1 = F.l1_loss(pred_ch, target_ch)
		l2 = F.mse_loss(pred_ch, target_ch)
		pred_center = pred_ch - pred_ch.mean(dim=2, keepdim=True)
		target_center = target_ch - target_ch.mean(dim=2, keepdim=True)
		denom = max(1, pred_center.shape[2])
		cov_pred = torch.bmm(pred_center, pred_center.transpose(1, 2)) / denom
		cov_target = torch.bmm(target_center, target_center.transpose(1, 2)) / denom
		spatial = F.mse_loss(cov_pred, cov_target)
		return l1 + 0.5 * l2 + self._spatial_lambda * spatial

	def set_burn_in_steps(self, k: int) -> None:
		"""Set burn-in steps with update."""
		self._burn_in_steps = max(0, int(k))

	def reset_state(self) -> None:
		"""Clear persistent state for streaming use."""
		self._g_state = None

	def set_state(self, g: torch.Tensor) -> None:
		"""Manually set persistent state for streaming use."""
		self._g_state = g

	def get_state(self) -> Optional[torch.Tensor]:
		"""Get current persistent state for streaming use."""
		return self._g_state

	def set_dt(self, dt: float) -> None:
		"""Set integration step for dynamics."""
		self._dt = float(dt)

	def set_spec_power(self, p: float) -> None:
		"""Set spectral magnitude compression power."""
		self._spec_power = float(p)

	def update(
		self, g: torch.Tensor, x: torch.Tensor
	):
		x_spec_ch = self._obs_to_spec_per_channel(x)
		x_hat_ch = self.decoder(g)
		residual_ch = x_spec_ch - x_hat_ch
		residual_embed = self.obs_encoder(residual_ch)
		delta, l_delta = self.update_module(g, residual_embed)
		if self._delta_clip is not None and self._delta_clip > 0:
			norm = torch.norm(delta, dim=1, keepdim=True)
			scale = self._delta_clip / (norm + 1e-8)
			scale = torch.clamp(scale, max=1.0)
			delta = delta * scale
			l_delta = (delta ** 2).mean()
		g_plus = g + delta
		return g_plus, l_delta

	def predict(self, g: torch.Tensor, s: Optional[torch.Tensor] = None) -> torch.Tensor:
		return self.dynamics(g, s, dt=self._dt)

	def step(
		self, g: torch.Tensor, x: torch.Tensor, s: Optional[torch.Tensor] = None
	) -> Dict[str, torch.Tensor]:
		g_plus, l_delta = self.update(g, x)
		g_tilde = self.predict(g_plus, s)
		return {
			"filtered_state": g_plus,
			"prediction_state": g_tilde,
			"l_delta": l_delta,
		}

	def forward_chunk_loss(
		self,
		x_seq: torch.Tensor,
		s_seq: Optional[torch.Tensor] = None,
		update_dropout: float = 0.0,
		detach_state: bool = False,
		start_step: int = 0,
		k_steps: int = 1,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Compute chunk losses without storing full state traces."""
		if x_seq.ndim not in (3, 4):
			raise ValueError(f"Expected x_seq rank 3/4, got shape {tuple(x_seq.shape)}")
		t_steps = x_seq.shape[0]
		batch_size = x_seq.shape[1]
		k_steps = max(1, int(k_steps))
		if s_seq is None and self.dynamics.stim_dim > 0:
			s_seq = torch.zeros(t_steps, batch_size, self.dynamics.stim_dim, device=x_seq.device)
		initialized_from_obs = False
		if self._g_state is None:
			g0 = torch.zeros(batch_size, self._latent_dim, device=x_seq.device)
			if t_steps > 0:
				g0, _ = self.update(g0, x_seq[0])
				initialized_from_obs = True
			self._g_state = g0
		elif detach_state:
			self._g_state = self._g_state.detach()
		g = self._g_state

		loss_now = torch.zeros((), device=x_seq.device)
		loss_k = torch.zeros((), device=x_seq.device)
		delta_loss = torch.zeros((), device=x_seq.device)
		now_count = 0
		k_count = 0

		max_t = t_steps - k_steps
		for t in range(max_t):
			x_t = x_seq[t]
			s_t = None if s_seq is None else s_seq[t]

			t_global = start_step + t
			do_update = (t_global < self._burn_in_steps) or (t_global % self._update_every == 0)
			if initialized_from_obs and t == 0:
				do_update = False
			drop_update = (self.training and update_dropout > 0.0 and
					 torch.rand(()) < update_dropout)
			use_update = do_update and not drop_update
			if use_update:
				g_plus, l_delta = self.update(g, x_t)
			else:
				l_delta = torch.zeros((), device=x_seq.device)
				g_plus = g

			if use_update:
				# Current-step reconstruction loss using updated state
				x_now_pred = self.decoder(g_plus)
				loss_now = loss_now + self._bandpower_loss(x_now_pred, x_t)
				now_count += 1

			# multi-step prediction loss (no updates within rollout)
			g_pred = g_plus
			for k in range(1, k_steps + 1):
				s_k = None if s_seq is None else s_seq[t + k - 1]
				g_pred = self.predict(g_pred, s_k)
				x_target = x_seq[t + k]
				x_pred = self.decoder(g_pred)
				loss_k = loss_k + self._bandpower_loss(x_pred, x_target)
				k_count += 1

			# One-step state evolution for filtering
			g_tilde = self.predict(g_plus, s_t)
			delta_loss = delta_loss + l_delta
			g = g_tilde

		self._g_state = g.detach() if detach_state else g
		now_denom = max(1, now_count)
		k_denom = max(1, k_count)
		delta_denom = max(1, t_steps - k_steps)
		return loss_k / k_denom, loss_now / now_denom, delta_loss / delta_denom
