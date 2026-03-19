import os
import glob
import torch
from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, time, timedelta
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import pyedflib
except ImportError:
    pyedflib = None

from model import OnlineStateSpaceModel


def pca_project_3d(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    u, s, v = torch.pca_lowrank(x, q=3)
    return x @ v[:, :3]


def rollout_predictions(
    model: OnlineStateSpaceModel,
    obs_seq: torch.Tensor,
) -> torch.Tensor:
    """Infer predicted latent states using update + predict."""
    obs_seq = obs_seq.to(next(model.parameters()).device)
    t_steps = obs_seq.shape[0]
    latent_dim = model._latent_dim
    g = torch.zeros(1, latent_dim, device=obs_seq.device)
    preds = []
    for t in range(t_steps - 1):
        x_t = obs_seq[t].unsqueeze(0)
        do_update = (t < model._burn_in_steps) or (t % model._update_every == 0)
        if do_update:
            g_plus, _ = model.update(g, x_t)
        else:
            g_plus = g
        g_tilde = model.predict(g_plus)
        preds.append(g_tilde.squeeze(0))
        g = g_tilde
    return torch.stack(preds, dim=0)


def fit_linear_probe(
    latents: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit linear probe Wg+b -> targets and return metrics."""
    latents = latents.detach().cpu()
    targets = targets.detach().cpu()
    ones = torch.ones(latents.shape[0], 1)
    feat = torch.cat([latents, ones], dim=1)
    coef = torch.linalg.lstsq(feat, targets).solution
    weight = coef[:-1].T
    bias = coef[-1]
    pred = feat @ coef
    mse = ((pred - targets) ** 2).mean()
    ss_tot = ((targets - targets.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    ss_res = ((targets - pred) ** 2).sum(dim=0)
    r2 = 1.0 - ss_res / ss_tot.clamp_min(1e-8)
    return weight, bias, mse, r2


def cca_align(
    latents: torch.Tensor,
    targets: torch.Tensor,
    n_components: int = 3,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CCA align latents and targets; return aligned coords and correlations."""
    x = latents.detach().cpu().numpy()
    y = targets.detach().cpu().numpy()
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    n_samples = x.shape[0]
    if n_samples < 2:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0,))

    n_components = min(n_components, x.shape[1], y.shape[1])

    cxx = (x.T @ x) / (n_samples - 1) + eps * np.eye(x.shape[1])
    cyy = (y.T @ y) / (n_samples - 1) + eps * np.eye(y.shape[1])
    cxy = (x.T @ y) / (n_samples - 1)

    def inv_sqrt(mat: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(mat)
        vals = np.clip(vals, eps, None)
        return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

    wx = inv_sqrt(cxx)
    wy = inv_sqrt(cyy)
    m = wx @ cxy @ wy
    u, s, vt = np.linalg.svd(m, full_matrices=False)

    ax = wx @ u[:, :n_components]
    ay = wy @ vt.T[:, :n_components]

    x_c = x @ ax
    y_c = y @ ay

    return x_c, y_c, s[:n_components]


class LorenzDataset(Dataset):
    def __init__(self, sequences: torch.Tensor):
        self.sequences = sequences

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]

def _pick_channel_indices(labels: List[str], priority: List[str], max_channels: int) -> List[int]:
    labels_norm = [lab.replace(" ", "").upper() for lab in labels]
    picked: List[int] = []
    for token in priority:
        tok = token.replace(" ", "").upper()
        for i, lab in enumerate(labels_norm):
            if tok in lab and i not in picked:
                picked.append(i)
                if len(picked) >= max_channels:
                    return picked
    return picked


def _parse_clock_hhmm(clock_text: str) -> time:
    clock_text = clock_text.strip()
    for fmt in ("%H:%M", "%H.%M"):
        try:
            return datetime.strptime(clock_text, fmt).time()
        except ValueError:
            pass
    raise ValueError(f"Invalid clock format '{clock_text}', expected HH:MM or HH.MM")


def _clock_in_window(cur: time, start: time, end: time) -> bool:
    if start < end:
        return start <= cur < end
    return (cur >= start) or (cur < end)


def _night_key(ts: datetime, start: time, end: time) -> date:
    if start < end:
        return ts.date()
    return ts.date() - timedelta(days=1) if ts.time() < end else ts.date()


def _slice_by_clock_and_night(
    patches: torch.Tensor,
    start_dt: datetime,
    patch_seconds: float,
    night_start: time,
    night_end: time,
) -> List[torch.Tensor]:
    n = patches.shape[0]
    if n == 0:
        return []

    valid = []
    keys = []
    for i in range(n):
        ts = start_dt + timedelta(seconds=float(i) * float(patch_seconds))
        if _clock_in_window(ts.time(), night_start, night_end):
            valid.append(i)
            keys.append(_night_key(ts, night_start, night_end))

    if not valid:
        return []

    segments: List[torch.Tensor] = []
    seg_start = 0
    for k in range(1, len(valid)):
        contiguous = (valid[k] == valid[k - 1] + 1)
        same_night = (keys[k] == keys[k - 1])
        if not contiguous or not same_night:
            idx = valid[seg_start:k]
            segments.append(patches[idx])
            seg_start = k
    idx = valid[seg_start:]
    segments.append(patches[idx])
    return segments


def build_real_dataset(
    dataset_dir: str,
    total_steps: int,
    stride_steps: int = 0,
    max_files: int = 0,
    seed: int = 0,
    channel_priority: Optional[List[str]] = None,
    obs_scale: float = 1.0,
    patch_seconds: float = 5.0,
    unified_fs: float = 100.0,
    night_start_clock: str = "00:00",
    night_end_clock: str = "06:00",
    single_night_only: bool = False,
    windows_per_night: int = 0,
    use_ss2_bio: bool = True,
    use_sleep_edf: bool = True,
    single_record_channel_only: bool = True,
    max_channels: int = 6,
    max_windows_total: int = 2048,
) -> torch.Tensor:
    if stride_steps <= 0: 
        stride_steps = total_steps
    if channel_priority is None:
        channel_priority = [
            "EEG F3-CLE",
            "F3-CLE",
            "EEG F4-CLE",
            "F4-CLE",
            "EEG C3-A2",
            "C3-A2",
            "EEG C4-A1",
            "C4-A1",
            "EEG O1-CLE",
            "O1-CLE",
            "EEG O2-CLE",
            "O2-CLE",
            "F3",
            "F4",
            "C3",
            "C4",
            "O1",
            "O2",
        ]
    rng = np.random.default_rng(seed)
    target_patch_width = max(1, int(round(unified_fs * patch_seconds)))
    night_start = _parse_clock_hhmm(night_start_clock)
    night_end = _parse_clock_hhmm(night_end_clock)
    psg_paths = sorted(glob.glob(os.path.join(dataset_dir, "SS2_bio", "* PSG.edf"))) if use_ss2_bio else []
    sleep_paths = sorted(glob.glob(os.path.join(dataset_dir, "sleep-edf", "*.npz"))) if use_sleep_edf else []
    if max_files > 0:
        psg_paths = psg_paths[:max_files]
        sleep_paths = sleep_paths[:max_files]
    if not psg_paths and not sleep_paths:
        raise ValueError(f"No SS2_bio EDF or sleep-edf NPZ files found under {dataset_dir}")

    sequences = []
    full_sequences = []
    locked_source_channel: Optional[str] = None

    def _accept_source_channel(key: str) -> bool:
        nonlocal locked_source_channel
        if not single_record_channel_only:
            return True
        if locked_source_channel is None:
            locked_source_channel = key
            return True
        return key == locked_source_channel

    def _resample_patch_width(x: torch.Tensor, target_width: int) -> torch.Tensor:
        if x.shape[-1] == target_width:
            return x
        if x.ndim == 2:
            x_1d = x.unsqueeze(1)  # (n_points, 1, width)
            x_1d = F.interpolate(x_1d, size=target_width, mode="linear", align_corners=False)
            return x_1d.squeeze(1)
        if x.ndim == 3:
            n_points, n_ch, width = x.shape
            x_1d = x.reshape(n_points * n_ch, 1, width)
            x_1d = F.interpolate(x_1d, size=target_width, mode="linear", align_corners=False)
            return x_1d.reshape(n_points, n_ch, target_width)
        raise ValueError(f"Unexpected rank in _resample_patch_width: {x.ndim}")

    def _normalize_sequence(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return (x - x.mean()) / (x.std() + 1e-6)
        if x.ndim == 3:
            mean = x.mean(dim=(0, 2), keepdim=True)
            std = x.std(dim=(0, 2), keepdim=True)
            return (x - mean) / (std + 1e-6)
        raise ValueError(f"Unexpected rank in _normalize_sequence: {x.ndim}")

    ss2_used = 0
    for path in psg_paths:
        if pyedflib is None:
            raise ImportError("pyedflib is required to read SS2_bio EDF files")
        with pyedflib.EdfReader(path) as edf:
            labels = edf.getSignalLabels()
            idx_list = _pick_channel_indices(labels, channel_priority, max_channels=max(1, int(max_channels)))
            if not idx_list:
                continue
            source_channel_key = (
                f"SS2_bio::{os.path.basename(path)}::ch" + "-".join(str(i) for i in idx_list)
            )
            if not _accept_source_channel(source_channel_key):
                continue
            fs_list = [float(edf.getSampleFrequency(i)) for i in idx_list]
            if any(f <= 0 for f in fs_list):
                continue
            fs = fs_list[0]
            if any(abs(f - fs) > 1e-6 for f in fs_list[1:]):
                continue
            sig_list = [edf.readSignal(i).astype(np.float32) for i in idx_list]
            min_len = min(sig.size for sig in sig_list)
            signal = np.stack([sig[:min_len] for sig in sig_list], axis=0)  # (C, N)
            start_dt = edf.getStartdatetime()

        if fs <= 0:
            continue
        samples_per_point = max(1, int(round(fs * patch_seconds)))
        n_points = signal.shape[1] // samples_per_point
        if n_points < 2:
            continue
        signal = signal[:, :n_points * samples_per_point]
        x = signal.reshape(signal.shape[0], n_points, samples_per_point).transpose(1, 0, 2)
        x = torch.from_numpy(x)
        x = _resample_patch_width(x, target_patch_width)
        x = _normalize_sequence(x)
        night_segments = _slice_by_clock_and_night(
            x,
            start_dt=start_dt,
            patch_seconds=patch_seconds,
            night_start=night_start,
            night_end=night_end,
        )
        full_sequences.extend(seg for seg in night_segments if seg.shape[0] >= 2)
        ss2_used += 1

    sleep_used = 0
    for path in sleep_paths:
        try:
            record = np.load(path, allow_pickle=True)
        except Exception:
            continue
        if "x" not in record or "fs" not in record:
            continue
        x_raw = record["x"]
        if x_raw.ndim == 3:
            source_channel_key = f"sleep-edf::{os.path.basename(path)}::ch0"
            if not _accept_source_channel(source_channel_key):
                continue
            x_raw = x_raw[..., 0]
        if x_raw.ndim != 2:
            continue
        if x_raw.ndim == 2:
            source_channel_key = f"sleep-edf::{os.path.basename(path)}::ch0"
            if not _accept_source_channel(source_channel_key):
                continue
        fs = float(record["fs"])
        if fs <= 0:
            continue
        if "header_raw" not in record:
            continue
        header_raw = record["header_raw"].item()
        dt_text = header_raw.get("date_time", "")
        try:
            start_dt = datetime.fromisoformat(str(dt_text))
        except ValueError:
            continue

        epoch_data = x_raw.astype(np.float32)
        epoch_samples = epoch_data.shape[1]
        samples_per_point = max(1, int(round(fs * patch_seconds)))

        if samples_per_point <= epoch_samples:
            n_sub = epoch_samples // samples_per_point
            if n_sub < 1:
                continue
            trimmed = epoch_data[:, : n_sub * samples_per_point]
            x = trimmed.reshape(-1, samples_per_point)
        else:
            signal = epoch_data.reshape(-1)
            n_points = signal.size // samples_per_point
            if n_points < 2:
                continue
            signal = signal[:n_points * samples_per_point]
            x = signal.reshape(n_points, samples_per_point)

        if x.shape[0] < 2:
            continue
        x = torch.from_numpy(x)
        x = x.unsqueeze(1)
        x = _resample_patch_width(x, target_patch_width)
        x = _normalize_sequence(x)
        night_segments = _slice_by_clock_and_night(
            x,
            start_dt=start_dt,
            patch_seconds=patch_seconds,
            night_start=night_start,
            night_end=night_end,
        )
        full_sequences.extend(seg for seg in night_segments if seg.shape[0] >= 2)
        sleep_used += 1

    if not full_sequences:
        raise ValueError("No sequences could be built from SS2_bio/sleep-edf files")

    if single_night_only:
        if not full_sequences:
            raise ValueError("No night sequences available for single-night training")
        if total_steps <= 0:
            raise ValueError("single_night_only=True requires total_steps > 0")
        picked_idx = int(rng.integers(0, len(full_sequences)))
        picked = full_sequences[picked_idx]
        if picked.shape[0] < total_steps:
            raise ValueError(
                f"Selected night too short for total_steps={total_steps}, length={picked.shape[0]}"
            )
        max_start = picked.shape[0] - total_steps
        all_starts = np.arange(max_start + 1)
        if windows_per_night > 0:
            n_windows = windows_per_night
            if n_windows > all_starts.size:
                raise ValueError(
                    f"windows_per_night={n_windows} exceeds available unique windows={all_starts.size}"
                )
            starts = rng.choice(all_starts, size=n_windows, replace=False)
        else:
            if max_windows_total > 0 and all_starts.size > max_windows_total:
                starts = rng.choice(all_starts, size=max_windows_total, replace=False)
                print(
                    "Single-night mode memory guard: "
                    f"available_windows={all_starts.size}, sampled_windows={starts.size}"
                )
            else:
                starts = all_starts
        rng.shuffle(starts)

        for start in starts.tolist():
            chunk = picked[start:start + total_steps]
            sequences.append(chunk)
        print(
            "Single-night mode: "
            f"picked_night_index={picked_idx}, night_len_steps={picked.shape[0]}, "
            f"window_steps={total_steps}, total_possible_windows={all_starts.size}, "
            f"used_windows={len(sequences)}"
        )
    elif total_steps <= 0:
        min_len = min(seq.shape[0] for seq in full_sequences)
        for seq in full_sequences:
            sequences.append(seq[:min_len])
    else:
        if stride_steps <= 0:
            stride_steps = total_steps
        for seq in full_sequences:
            if seq.shape[0] < total_steps:
                continue
            max_start = seq.shape[0] - total_steps
            for start in range(0, max_start + 1, stride_steps):
                chunk = seq[start:start + total_steps]
                sequences.append(chunk)
    if not sequences:
        raise ValueError("No sequences could be built from SS2_bio/sleep-edf files")

    if max_windows_total > 0 and len(sequences) > max_windows_total:
        pick_idx = rng.choice(len(sequences), size=max_windows_total, replace=False)
        sequences = [sequences[i] for i in pick_idx.tolist()]
        print(
            "Memory guard: "
            f"sampled {len(sequences)} windows from larger candidate set"
        )

    rng.shuffle(sequences)
    all_seq = torch.stack(sequences, dim=0)
    all_seq = all_seq * obs_scale
    print(
        "Loaded dataset summary: "
        f"SS2_bio records={ss2_used}, sleep-edf records={sleep_used}, "
        f"total sequences={all_seq.shape[0]}, patch width={all_seq.shape[-1]}, "
        f"clock window={night_start_clock}-{night_end_clock}, unified_fs={unified_fs}, "
        f"single_night_only={single_night_only}, "
        f"single_record_channel_only={single_record_channel_only}, "
        f"locked_source_channel={locked_source_channel}"
    )
    return all_seq


def train_lorenz_model(
    latent_dim: int = 64,
    obs_dim: Optional[int] = None,
    stim_dim: int = 0,
    total_steps: int = 2000,
    burn_in_steps: int = 150,
    chunk_len: int = 200,
    dataset_dt: float = 0.01,
    obs_scale: float = 1.0,
    obs_nl_scale: float = 0.3,
    noise_std: float = 0.05,
    obs_embed_dim: int = 128,
    decoder_hidden_dim: int = 256,
    spec_power: float = 0.5,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    train_windows: int = 0,
    val_windows: int = 0,
    batch_size: int = 64,
    epochs: int = 400,
    lr: float = 1e-3,
    update_dropout: float = 0,
    k_steps_max: int = 15,
    curriculum_epochs: int = 260,
    alpha_now: float = 0.4,
    beta_delta: float = 1e-6,
    update_every_schedule: Tuple[int, int, int, int] = (10, 10, 10, 10),
    update_every_milestones: Tuple[float, float, float] = (0.3, 0.6, 0.75),
    dt_schedule: Tuple[float, float, float, float] = (0.1, 0.1, 0.1, 0.1),
    dt_milestones: Tuple[float, float, float] = (0.3, 0.6, 0.8),
    dataset_root: str = "Latent -Dynamical-System/real_data/dataset",
    patch_seconds: float = 5.0,
    unified_fs: float = 100.0,
    stride_steps: int = 0,
    max_files: int = 1,
    channel_priority: Optional[List[str]] = None,
    night_start_clock: str = "00:00",
    night_end_clock: str = "06:00",
    single_night_only: bool = True,
    windows_per_night: int = 0,
    use_ss2_bio: bool = True,
    use_sleep_edf: bool = False,
    single_record_channel_only: bool = True,
    max_channels: int = 6,
    max_windows_total: int = 2048,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 1,
    resume_from_checkpoint: bool = True,
    num_workers: int = 0,
    seed: int = 0,
    device: str = "cuda",
) -> None:
    if chunk_len <= 0:
        raise ValueError("chunk_len must be > 0")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    all_data = build_real_dataset(
        dataset_root,
        total_steps,
        stride_steps=stride_steps,
        max_files=max_files,
        seed=seed,
        channel_priority=channel_priority,
        obs_scale=obs_scale,
        patch_seconds=patch_seconds,
        unified_fs=unified_fs,
        night_start_clock=night_start_clock,
        night_end_clock=night_end_clock,
        single_night_only=single_night_only,
        windows_per_night=windows_per_night,
        use_ss2_bio=use_ss2_bio,
        use_sleep_edf=use_sleep_edf,
        single_record_channel_only=single_record_channel_only,
        max_channels=max_channels,
        max_windows_total=max_windows_total,
    )
    proj = None
    if obs_dim is None:
        if all_data.ndim == 4:
            obs_dim = (all_data.shape[-2], all_data.shape[-1])
        else:
            obs_dim = all_data.shape[-1]
    total_samples = all_data.shape[0]
    if total_samples == 0:
        raise ValueError("No training samples available after preprocessing")

    if train_windows > 0 or val_windows > 0:
        if train_windows <= 0 or val_windows <= 0:
            raise ValueError("When using scalar split, both train_windows and val_windows must be > 0")
        total_needed = int(train_windows + val_windows)
        if total_needed > total_samples:
            raise ValueError(
                f"train_windows + val_windows = {total_needed} exceeds available unique windows = {total_samples}"
            )
        perm = torch.randperm(total_samples)
        picked = all_data[perm[:total_needed]]
        train_data = picked[:train_windows]
        val_data = picked[train_windows:train_windows + val_windows]
        print(
            f"Scalar split: train_windows={train_windows}, val_windows={val_windows}, "
            f"available_windows={total_samples}"
        )
    else:
        train_frac = float(train_frac)
        val_frac = float(val_frac)
        if train_frac <= 0 or val_frac <= 0 or (train_frac + val_frac) >= 1.0:
            raise ValueError("train_frac and val_frac must be > 0 and train_frac + val_frac < 1.0")
        n_train = max(1, int(total_samples * train_frac))
        n_val = max(1, int(total_samples * val_frac))
        if n_train + n_val > total_samples:
            n_val = max(1, total_samples - n_train)
        if n_train + n_val > total_samples:
            n_train = max(1, total_samples - n_val)
        perm = torch.randperm(total_samples)
        all_data = all_data[perm]
        train_data = all_data[:n_train]
        val_data = all_data[n_train:n_train + n_val]
        print(
            f"Ratio split: train_frac={train_frac:.2f}, val_frac={val_frac:.2f}, "
            f"train_windows={len(train_data)}, val_windows={len(val_data)}, "
            f"unused_windows={total_samples - len(train_data) - len(val_data)}"
        )

    train_dataset = LorenzDataset(train_data)
    val_dataset = LorenzDataset(val_data)
    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty; reduce total_steps or stride_steps")
    if batch_size > len(train_dataset):
        batch_size = len(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(max(0, int(num_workers)) > 0),
    )
    if len(val_dataset) == 0:
        raise ValueError("Val dataset is empty; reduce total_steps or stride_steps")
    val_batch = min(batch_size, len(val_dataset))
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch,
        shuffle=False,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(max(0, int(num_workers)) > 0),
    )

    model = OnlineStateSpaceModel(
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        stim_dim=stim_dim,
        obs_embed_dim=obs_embed_dim,
        hidden_dim=256,
        decoder_hidden_dim=decoder_hidden_dim,
        gate_type="diag",
        low_rank=8,
        update_every=update_every_schedule[0],
        burn_in_steps=burn_in_steps,
        delta_clip=8.0,
        spec_power=spec_power,
        sample_rate=unified_fs,
        dt=dt_schedule[0],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")

    best_val = float("inf")
    best_state = None
    no_improve = 0
    patience = 30
    prev_phase: Optional[Tuple[int, int]] = None
    milestone_activated = False
    skipped_batches = 0
    start_epoch = 1

    if resume_from_checkpoint and os.path.exists(latest_ckpt_path):
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val = float(ckpt.get("best_val", best_val))
        no_improve = int(ckpt.get("no_improve", no_improve))
        prev_phase = ckpt.get("prev_phase", prev_phase)
        milestone_activated = bool(ckpt.get("milestone_activated", milestone_activated))
        if "best_state" in ckpt:
            best_state = ckpt["best_state"]
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if start_epoch <= epochs:
            print(f"Resumed from checkpoint: {latest_ckpt_path} (start_epoch={start_epoch})")
        else:
            print("Checkpoint epoch already >= target epochs; training loop will be skipped.")

    for epoch in range(start_epoch, epochs + 1):
        if len(update_every_schedule) != 4 or len(update_every_milestones) != 3:
            raise ValueError("update_every_schedule must have 4 items and milestones 3 items")
        if len(dt_schedule) != 4 or len(dt_milestones) != 3:
            raise ValueError("dt_schedule must have 4 items and milestones 3 items")
        progress = (epoch - 1) / max(1, epochs - 1)
        if progress < update_every_milestones[0]:
            update_phase = 0
        elif progress < update_every_milestones[1]:
            update_phase = 1
        elif progress < update_every_milestones[2]:
            update_phase = 2
        else:
            update_phase = 3
        model._update_every = update_every_schedule[update_phase]

        if progress < dt_milestones[0]:
            dt_phase = 0
        elif progress < dt_milestones[1]:
            dt_phase = 1
        elif progress < dt_milestones[2]:
            dt_phase = 2
        else:
            dt_phase = 3
        model.set_dt(dt_schedule[dt_phase])

        current_phase = (update_phase, dt_phase)
        if prev_phase is None:
            prev_phase = current_phase
        elif current_phase != prev_phase:
            milestone_activated = True
            no_improve = 0
            prev_phase = current_phase
            print(
                f"Milestone changed at epoch {epoch}: "
                f"update_every={model._update_every}, dt={model._dt:.4f}. "
                f"Early-stop watch restarted ({patience} epochs)."
            )
        ramp_epochs = max(1, curriculum_epochs)
        k_max = 1 + (k_steps_max - 1) * min(epoch - 1, ramp_epochs) // ramp_epochs
        model.train()
        train_loss = 0.0
        train_k = 0.0
        train_now = 0.0
        train_delta = 0.0
        for batch_idx, x_seq in enumerate(train_loader):
            x_seq = x_seq.to(device)
            if x_seq.ndim == 3:
                x_seq = x_seq.permute(1, 0, 2)  # (T, B, W)
            elif x_seq.ndim == 4:
                x_seq = x_seq.permute(1, 0, 2, 3)  # (T, B, C, W)
            else:
                raise ValueError(f"Unexpected batch shape: {tuple(x_seq.shape)}")
            n_steps = x_seq.shape[0]
            n_chunks = (n_steps + chunk_len - 1) // chunk_len
            model.reset_state()
            optimizer.zero_grad(set_to_none=True)
            seq_step = 0
            debug_done = True
            debug_target_global = -1
            batch_failed = False
            for start in range(0, n_steps, chunk_len):
                end = start + chunk_len
                x_chunk = x_seq[start:end]
                k_steps = k_max

                chunk_start_global = seq_step
                chunk_end_global = seq_step + x_chunk.shape[0] - 1

                if (
                    batch_idx == 0
                    and not debug_done
                    and chunk_start_global <= debug_target_global <= chunk_end_global
                ):
                    with torch.no_grad():
                        g_probe = model.get_state()
                        if g_probe is None:
                            g_probe = torch.zeros(x_chunk.shape[1], model._latent_dim, device=x_chunk.device)

                        probe_idx = debug_target_global - chunk_start_global
                        for tt in range(probe_idx):
                            x_t_probe = x_chunk[tt]
                            t_global_probe = seq_step + tt
                            do_update_probe = (
                                t_global_probe < model._burn_in_steps
                            ) or (t_global_probe % model._update_every == 0)
                            if do_update_probe:
                                g_plus_probe, _ = model.update(g_probe, x_t_probe)
                            else:
                                g_plus_probe = g_probe
                            g_probe = model.predict(g_plus_probe, None)

                        x_dbg = x_chunk[probe_idx]
                        x_dbg_spec = model.observation_to_spectrum_per_channel(x_dbg)
                        t_global_dbg = seq_step + probe_idx
                        do_update_dbg = (
                            t_global_dbg < model._burn_in_steps
                        ) or (t_global_dbg % model._update_every == 0)

                        x_hat_pre = model.decoder(g_probe)
                        residual_dbg = x_dbg_spec - x_hat_pre
                        residual_embed_dbg = model.obs_encoder(residual_dbg)
                        if do_update_dbg:
                            g_plus_dbg, _ = model.update(g_probe, x_dbg)
                        else:
                            g_plus_dbg = g_probe
                        delta_dbg = g_plus_dbg - g_probe
                        x_hat_post = model.decoder(g_plus_dbg)
                        print(
                            f"debug[t={probe_idx}] | x std="
                            f"{x_dbg.std():.4f} | "
                            "x_spec mean/std="
                            f"{x_dbg_spec.mean():.4f}/{x_dbg_spec.std():.4f} | "
                            "x_hat_pre mean/std="
                            f"{x_hat_pre.mean():.4f}/{x_hat_pre.std():.4f} | "
                            "x_hat_post mean/std="
                            f"{x_hat_post.mean():.4f}/{x_hat_post.std():.4f} | "
                            "res mean/std="
                            f"{residual_dbg.mean():.4f}/{residual_dbg.std():.4f} | "
                            "res_embed norm="
                            f"{residual_embed_dbg.norm(dim=1).mean():.4f} | "
                            "g norm="
                            f"{g_probe.norm(dim=1).mean():.4f} | "
                            "delta norm="
                            f"{delta_dbg.norm(dim=1).mean():.4f} | "
                            "g_plus norm="
                            f"{g_plus_dbg.norm(dim=1).mean():.4f}"
                        )
                        debug_done = True

                try:
                    loss_k, loss_now, delta_loss = model.forward_chunk_loss(
                        x_chunk,
                        s_seq=None,
                        update_dropout=update_dropout,
                        detach_state=True,
                        start_step=seq_step,
                        k_steps=k_steps,
                    )
                except RuntimeError as err:
                    err_msg = str(err)
                    if (
                        "CUDA out of memory" in err_msg
                        or "CUBLAS_STATUS_INTERNAL_ERROR" in err_msg
                        or "cublasSgemm" in err_msg
                    ):
                        skipped_batches += 1
                        batch_failed = True
                        optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print(f"Warning: skipped batch {batch_idx} due to CUDA forward error: {err_msg[:160]}")
                        break
                    raise

                loss = loss_k + alpha_now * loss_now + beta_delta * delta_loss
                if not torch.isfinite(loss):
                    skipped_batches += 1
                    batch_failed = True
                    optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break

                try:
                    (loss / n_chunks).backward()
                except RuntimeError as err:
                    err_msg = str(err)
                    if (
                        "CUDA out of memory" in err_msg
                        or "CUBLAS_STATUS_INTERNAL_ERROR" in err_msg
                        or "cublasSgemm" in err_msg
                    ):
                        skipped_batches += 1
                        batch_failed = True
                        optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print(f"Warning: skipped batch {batch_idx} due to CUDA backward error: {err_msg[:160]}")
                        break
                    raise

                train_loss += loss.item() / n_chunks
                train_k += loss_k.item() / n_chunks
                train_now += loss_now.item() / n_chunks
                train_delta += delta_loss.item() / n_chunks
                seq_step += x_chunk.shape[0]

            if batch_failed:
                continue

            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            except RuntimeError as err:
                err_msg = str(err)
                if (
                    "CUBLAS_STATUS_INTERNAL_ERROR" in err_msg
                    or "CUDA out of memory" in err_msg
                    or "cublasSgemm" in err_msg
                ):
                    skipped_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"Warning: skipped batch {batch_idx} due to CUDA error: {err_msg[:160]}")
                    continue
                raise

        avg_train = train_loss / len(train_loader)
        avg_train_k = train_k / len(train_loader)
        avg_train_now = train_now / len(train_loader)
        avg_train_delta = train_delta / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_k = 0.0
        val_now = 0.0
        val_delta = 0.0
        with torch.no_grad():
            for x_seq in val_loader:
                x_seq = x_seq.to(device)
                if x_seq.ndim == 3:
                    x_seq = x_seq.permute(1, 0, 2)
                elif x_seq.ndim == 4:
                    x_seq = x_seq.permute(1, 0, 2, 3)
                else:
                    raise ValueError(f"Unexpected val batch shape: {tuple(x_seq.shape)}")
                n_steps = x_seq.shape[0]
                n_chunks = (n_steps + chunk_len - 1) // chunk_len

                model.reset_state()
                batch_loss = 0.0
                seq_step = 0
                for start in range(0, n_steps, chunk_len):
                    end = start + chunk_len
                    x_chunk = x_seq[start:end]
                    loss_k, loss_now, delta_loss = model.forward_chunk_loss(
                        x_chunk,
                        s_seq=None,
                        update_dropout=0.0,
                        detach_state=True,
                        start_step=seq_step,
                        k_steps=k_steps_max,
                    )
                    loss = loss_k + alpha_now * loss_now + beta_delta * delta_loss

                    batch_loss += loss.item()
                    val_k += loss_k.item() / n_chunks
                    val_now += loss_now.item() / n_chunks
                    val_delta += delta_loss.item() / n_chunks
                    seq_step += x_chunk.shape[0]

                val_loss += batch_loss / n_chunks

        avg_val = val_loss / len(val_loader)
        avg_val_k = val_k / len(val_loader)
        avg_val_now = val_now / len(val_loader)
        avg_val_delta = val_delta / len(val_loader)
        print(
            f"Epoch {epoch:03d} | train={avg_train:.6f} | val={avg_val:.6f} "
            f"| train_k={avg_train_k:.6f} train_now={avg_train_now:.6f} train_delta={avg_train_delta:.6f} "
            f"| val_k={avg_val_k:.6f} val_now={avg_val_now:.6f} val_delta={avg_val_delta:.6f} "
            f"| skipped_batches={skipped_batches}"
        )
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_state = {
                "state_dict": model.state_dict(),
                "latent_dim": latent_dim,
                "obs_dim": obs_dim,
                "proj": proj,
                "obs_mean": None,
                "obs_std": None,
            }
            no_improve = 0
        else:
            no_improve += 1
            if milestone_activated and no_improve >= patience:
                print(f"Early stop: no val improvement for {patience} epochs")
                break

        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val": best_val,
            "no_improve": no_improve,
            "prev_phase": prev_phase,
            "milestone_activated": milestone_activated,
            "best_state": best_state,
        }
        torch.save(ckpt_payload, latest_ckpt_path)
        if checkpoint_every > 0 and (epoch % checkpoint_every == 0):
            epoch_ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pth")
            torch.save(ckpt_payload, epoch_ckpt_path)

    model_path = os.path.join(project_root, "lorenz_model.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if best_state is None:
        best_state = {
            "state_dict": model.state_dict(),
            "latent_dim": latent_dim,
            "obs_dim": obs_dim,
            "proj": proj,
            "obs_mean": None,
            "obs_std": None,
        }
    torch.save(best_state, model_path)
    model.load_state_dict(best_state["state_dict"])

    model.eval()
    with torch.no_grad():
        sample = val_data[0] if len(val_data) > 0 else train_data[0]
        obs_seq = sample.to(device)
        pred_states = rollout_predictions(model, obs_seq)
        pred_obs = model.decoder(pred_states)
        true_obs_spec = model.observation_to_spectrum_per_channel(obs_seq[:-1])

    pred_pca = pca_project_3d(pred_states.detach().cpu())
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.plot(
        pred_pca[:, 0].numpy(),
        pred_pca[:, 1].numpy(),
        pred_pca[:, 2].numpy(),
        color="tab:orange",
        label="Pred (PCA)",
    )
    ax3.set_title("Predicted latent (PCA 3D)")
    ax3.legend()
    fig3.tight_layout()
    fig3_path = os.path.join(project_root, "pred_latent_pca.png")
    fig3.savefig(fig3_path, dpi=150)

    fig2 = plt.figure(figsize=(8, 4))
    ax2 = fig2.add_subplot(111)
    ax2.plot(
        pred_obs[:, 0, 0].cpu(),
        color="tab:orange",
        label="Obs-spec (pred)",
        alpha=0.7,
        zorder=1,
    )
    ax2.plot(
        true_obs_spec[:, 0, 0].cpu(),
        color="tab:blue",
        label="Obs-spec (true)",
        alpha=0.7,
        zorder=3,
    )
    ax2.set_title("Band-power space: True vs Predicted")
    ax2.legend()
    fig2.tight_layout()
    fig2_path = os.path.join(project_root, "obs_pred_vs_true.png")
    fig2.savefig(fig2_path, dpi=150)


if __name__ == "__main__":
    train_lorenz_model()
