"""Inference-only components extracted from the EventEdgeTCN reference release."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _as_time_index(
    time_index: Optional[np.ndarray],
    n: int,
) -> Optional[np.ndarray]:
    if time_index is None:
        return None
    t_abs = np.asarray(time_index, dtype=np.int64)
    if len(t_abs) != n:
        raise ValueError(f"time_index length {len(t_abs)} != sequence length {n}")
    return t_abs


def _span_time_contiguous(t_abs: np.ndarray, start: int, end_exclusive: int) -> bool:
    """True iff every consecutive pair in ``[start, end_exclusive)`` is adjacent in time."""
    if end_exclusive - start <= 1:
        return True
    for k in range(start + 1, end_exclusive):
        if int(t_abs[k]) != int(t_abs[k - 1]) + 1:
            return False
    return True


def derive_state_from_power(
    power: np.ndarray,
    threshold: float = 20.0,
    min_on: int = 10,
    min_off: int = 2,
    time_index: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Threshold + min ON/OFF morphology.

    When ``time_index`` is set, ON/OFF runs are measured only within dense-time
    contiguous spans; ``min_off`` never fills across a missing-time hard break.
    """
    state = (np.asarray(power) >= threshold).astype(np.int64)
    n = len(state)
    t_abs = _as_time_index(time_index, n)

    i = 0
    while i < n:
        if state[i] == 1:
            j = i + 1
            while j < n and state[j] == 1:
                if t_abs is not None and int(t_abs[j]) != int(t_abs[j - 1]) + 1:
                    break
                j += 1
            if (j - i) < min_on:
                state[i:j] = 0
            i = j
        else:
            i += 1

    i = 0
    while i < n:
        if state[i] == 0:
            j = i + 1
            while j < n and state[j] == 0:
                if t_abs is not None and int(t_abs[j]) != int(t_abs[j - 1]) + 1:
                    break
                j += 1
            left_on = i > 0 and state[i - 1] == 1
            right_on = j < n and state[j] == 1
            # Require contiguous time from the left ON through the OFF span to the
            # right ON sample; otherwise a missing bin hard-breaks morphology.
            can_fill = left_on and right_on and (j - i) < min_off
            if can_fill and t_abs is not None:
                can_fill = _span_time_contiguous(t_abs, i - 1, j + 1)
            if can_fill:
                state[i:j] = 1
            i = j
        else:
            i += 1

    return state.astype(np.int64)


def find_activation_segments(
    state: np.ndarray,
    time_index: Optional[np.ndarray] = None,
) -> List[Tuple[int, int]]:
    """Inclusive ON segments on an evaluation sequence.

    If ``time_index`` is provided (absolute dense-timeline coordinates for each
    sequence position), a run is cut whenever consecutive ON samples are not
    adjacent in time. This implements missing-label / discarded-bin hard breaks
    after the valid-time sequence \(\mathcal{V}\) has dropped those bins.
    """
    segs = []
    state = np.asarray(state).astype(np.int64)
    n = len(state)
    t_abs = _as_time_index(time_index, n)
    i = 0
    while i < n:
        if state[i] == 1:
            j = i + 1
            while j < n and state[j] == 1:
                if t_abs is not None and int(t_abs[j]) != int(t_abs[j - 1]) + 1:
                    break
                j += 1
            segs.append((i, j - 1))
            i = j
        else:
            i += 1
    return segs


def segments_to_state(length: int, segs: Sequence[Tuple[int, int]]) -> np.ndarray:
    state = np.zeros(length, dtype=np.int64)
    for s, e in segs:
        state[s:e + 1] = 1
    return state


def merge_segments(
    segs: Sequence[Tuple[int, int]],
    merge_gap: int,
    time_index: Optional[np.ndarray] = None,
) -> List[Tuple[int, int]]:
    """Merge nearby ON segments; never across a dense-timeline discontinuity."""
    if not segs:
        return []
    t_abs = None if time_index is None else np.asarray(time_index, dtype=np.int64)
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        last_s, last_e = merged[-1]
        index_gap = s - last_e - 1
        can_merge = index_gap <= merge_gap
        if can_merge and t_abs is not None:
            # Path from end of previous ON through intervening OFFs to new onset.
            can_merge = _span_time_contiguous(t_abs, last_e, s + 1)
        if can_merge:
            merged[-1][1] = max(last_e, e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def constrain_cycle_lengths(
    segs: Sequence[Tuple[int, int]],
    min_cycle_len: int,
    max_cycle_len: int,
    time_index: Optional[np.ndarray] = None,
) -> List[Tuple[int, int]]:
    """Drop segments outside ``[min_cycle_len, max_cycle_len]`` (dense bins if timed)."""
    t_abs = None if time_index is None else np.asarray(time_index, dtype=np.int64)
    kept = []
    for s, e in segs:
        if t_abs is None:
            dur = e - s + 1
        else:
            dur = int(t_abs[e]) - int(t_abs[s]) + 1
        if dur < min_cycle_len:
            continue
        if dur > max_cycle_len:
            continue
        kept.append((s, e))
    return kept


def decode_event_state(
    probs: np.ndarray,
    threshold: float,
    min_on: int,
    min_off: int,
    merge_gap: int,
    min_cycle_len: int,
    max_cycle_len: int,
    time_index: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Frozen event decoder; morphology respects missing-time hard breaks when timed."""
    pred_state = (np.asarray(probs) >= threshold).astype(np.int64)
    pred_state = derive_state_from_power(
        pred_state.astype(np.float32),
        threshold=0.5,
        min_on=min_on,
        min_off=min_off,
        time_index=time_index,
    )
    segs = find_activation_segments(pred_state, time_index=time_index)
    segs = merge_segments(segs, merge_gap=merge_gap, time_index=time_index)
    segs = constrain_cycle_lengths(
        segs,
        min_cycle_len=min_cycle_len,
        max_cycle_len=max_cycle_len,
        time_index=time_index,
    )
    return segments_to_state(len(pred_state), segs)


def extract_aux_batch(raw_norm: np.ndarray, ws: int, chunk: int = 200_000) -> np.ndarray:
    """Vectorized aux for dense endpoints ``i = ws .. n-1`` (same features as extract_aux_features)."""
    n = len(raw_norm)
    n_win = n - ws
    if n_win <= 0:
        return np.zeros((0, 12), dtype=np.float32)

    out = np.zeros((n_win, 12), dtype=np.float32)
    x = np.arange(ws, dtype=np.float32)
    x_mean = float(x.mean())
    x_var = float(((x - x_mean) ** 2).sum())
    q = max(ws // 4, 1)
    rn = np.asarray(raw_norm, dtype=np.float32)

    for start in range(0, n_win, chunk):
        end = min(start + chunk, n_win)
        m = end - start
        base = rn[start : start + m + ws - 1]
        windows = np.lib.stride_tricks.as_strided(
            base,
            shape=(m, ws),
            strides=(base.strides[0], base.strides[0]),
        )
        w = np.array(windows, dtype=np.float32, copy=True)

        mean = w.mean(axis=1)
        std = w.std(axis=1) + 1e-6
        mx = w.max(axis=1)
        mn = w.min(axis=1)
        d = np.diff(w, axis=1)
        slope = ((w - mean[:, None]) * (x - x_mean)).sum(axis=1) / max(x_var, 1e-6)
        q1 = np.quantile(w, 0.25, axis=1)
        q3 = np.quantile(w, 0.75, axis=1)
        head = w[:, :q].mean(axis=1)
        tail = w[:, -q:].mean(axis=1)

        out[start:end, 0] = mean
        out[start:end, 1] = std
        out[start:end, 2] = mx
        out[start:end, 3] = mn
        out[start:end, 4] = d.mean(axis=1)
        out[start:end, 5] = d.std(axis=1) + 1e-6
        out[start:end, 6] = slope
        out[start:end, 7] = mx - mn
        out[start:end, 8] = np.abs(d).mean(axis=1)
        out[start:end, 9] = q1
        out[start:end, 10] = q3
        out[start:end, 11] = tail - head

    return out.astype(np.float32)


@dataclass
class ModelConfig:
    name: str = "training_eventedgetcn_core_event_tcn"
    ws: int = 240
    in_ch: int = 1
    aux_dim: int = 12
    stem_ch: int = 32
    block_ch: int = 48
    num_blocks: int = 5
    kernel_size: int = 7
    dropout: float = 0.10
    norm: str = "bn"
    power_scale: float = 2000.0
    use_aux: bool = True
    use_se: bool = True


def make_norm_1d(channels: int, kind: str = "bn") -> nn.Module:
    if kind == "gn":
        groups = 8 if channels % 8 == 0 else 4 if channels % 4 == 0 else 1
        return nn.GroupNorm(groups, channels)
    return nn.BatchNorm1d(channels)


class DSResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1, dropout: float = 0.1, norm: str = "bn"):
        super().__init__()
        pad = ((kernel_size - 1) // 2) * dilation

        self.dw1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            groups=channels,
        )
        self.pw1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.n1 = make_norm_1d(channels, norm)

        self.dw2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            groups=channels,
        )
        self.pw2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.n2 = make_norm_1d(channels, norm)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x

        x = self.dw1(x)
        x = self.pw1(x)
        x = self.n1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.dw2(x)
        x = self.pw2(x)
        x = self.n2(x)
        x = self.dropout(x)

        return F.relu(x + res)


class SE1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        s = x.mean(dim=-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).unsqueeze(-1)
        return x * s


class EdgeEventTCN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.stem = nn.Sequential(
            nn.Conv1d(cfg.in_ch, cfg.stem_ch, kernel_size=cfg.kernel_size, padding=cfg.kernel_size // 2),
            make_norm_1d(cfg.stem_ch, cfg.norm),
            nn.ReLU(),
            nn.Conv1d(cfg.stem_ch, cfg.block_ch, kernel_size=1),
            make_norm_1d(cfg.block_ch, cfg.norm),
            nn.ReLU(),
        )

        dilations = [1, 2, 4, 8, 16][: cfg.num_blocks]
        self.blocks = nn.ModuleList(
            [
                DSResBlock(
                    channels=cfg.block_ch,
                    kernel_size=cfg.kernel_size,
                    dilation=d,
                    dropout=cfg.dropout,
                    norm=cfg.norm,
                )
                for d in dilations
            ]
        )

        self.se = SE1D(cfg.block_ch, reduction=8)

        self.use_aux = bool(getattr(cfg, "use_aux", True))
        if self.use_aux:
            self.aux = nn.Sequential(
                nn.Linear(cfg.aux_dim, 24),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(24, 16),
                nn.ReLU(),
            )
            fusion_in = cfg.block_ch + 16
        else:
            self.aux = None
            fusion_in = cfg.block_ch

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 48),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(48, 32),
            nn.ReLU(),
        )

        self.state_head = nn.Linear(32, 1)
        self.power_head = nn.Linear(32, 1)

    def encode(self, seq, aux):
        """Shared trunk (+ optional aux fusion) used by power-activation variants."""
        if seq.dim() == 2:
            seq = seq.unsqueeze(1)

        x = self.stem(seq)
        for blk in self.blocks:
            x = blk(x)
        if bool(getattr(self.cfg, "use_se", True)):
            x = self.se(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        if self.aux is not None:
            auxf = self.aux(aux)
            z = self.fusion(torch.cat([x, auxf], dim=1))
        else:
            z = self.fusion(x)
        return z

    def forward(self, seq, aux):
        z = self.encode(seq, aux)
        state_logit = self.state_head(z).squeeze(1)
        power_pred = F.relu(self.power_head(z).squeeze(1))
        return state_logit, power_pred

