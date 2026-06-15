"""Построение вектора наблюдения для инференса (без полного цикла Gym)."""

from __future__ import annotations

import numpy as np


def build_observation_vector(
    features: np.ndarray,
    window: int,
    t: int,
    w: float,
    balance: float,
    w_min: float,
    w_max: float,
) -> np.ndarray:
    if t < window:
        raise ValueError("t должен быть >= window")
    sl = features[t - window : t, :].reshape(-1)
    log_b = float(np.log(np.clip(balance, 1e-12, 1e12)))
    obs = np.concatenate(
        [sl, [float(np.clip(w, w_min, w_max)), log_b]]
    ).astype(np.float32)
    return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)


def apply_max_delta(w_old: float, w_raw: float, w_min: float, w_max: float, max_delta: float | None) -> float:
    w_raw = float(np.clip(w_raw, w_min, w_max))
    if max_delta is None or max_delta <= 0:
        return w_raw
    dw = float(np.clip(w_raw - w_old, -max_delta, max_delta))
    return float(np.clip(w_old + dw, w_min, w_max))
