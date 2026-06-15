"""Загрузка и простая причинная подготовка OHLCV для среды."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_ohlcv_csv(path: str | Path) -> tuple[pd.DataFrame, np.ndarray | None]:
    """
    Загружает CSV с колонками open, high, low, close, volume.

    Опционально (для перпетуалов): колонка ``funding_rate`` или ``funding``
    — ставка funding на эту свечу (как на бирже, обычно очень мала, знак:
    положительный — лонги платят шортам за интервал свечи).

    Возвращает (df_ohlcv, funding_или_None). Длина funding совпадает с числом строк.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В CSV не хватает колонок: {missing}. Нужны: {required}")
    out = df[list(required)].astype(np.float64).copy()
    funding = None
    for name in ("funding_rate", "funding"):
        if name in df.columns:
            funding = np.asarray(df[name].values, dtype=np.float64)
            if len(funding) != len(out):
                raise ValueError(
                    f"Колонка {name}: длина {len(funding)} != числу свечей {len(out)}"
                )
            funding = np.nan_to_num(funding, nan=0.0)
            break
    return out, funding


def build_features(
    df: pd.DataFrame,
    window: int,
    eps: float = 1e-8,
    min_std: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Признаки только из прошлого относительно текущего шага t:
    лог-доходность close, нормированный объём, доли high-low к close.
    Возвращает:
      X — массив формы (T, F) признаков по каждому времени;
      valid — булев массив длины T: True там, где окно window полностью в прошлом.
    """
    close = df["close"].values
    vol = df["volume"].values
    hl = (df["high"].values - df["low"].values) / np.maximum(df["close"].values, eps)

    log_ret = np.zeros_like(close)
    log_ret[1:] = np.log(np.maximum(close[1:], eps) / np.maximum(close[:-1], eps))

    v_mean = pd.Series(vol).rolling(window, min_periods=window).mean().values
    v_std = np.maximum(
        pd.Series(vol).rolling(window, min_periods=window).std().values, min_std
    )
    vol_z = (vol - v_mean) / (v_std + eps)

    hl_mean = pd.Series(hl).rolling(window, min_periods=window).mean().values
    hl_std = np.maximum(
        pd.Series(hl).rolling(window, min_periods=window).std().values, min_std
    )
    hl_z = (hl - hl_mean) / (hl_std + eps)

    lr_mean = pd.Series(log_ret).rolling(window, min_periods=window).mean().values
    lr_std = np.maximum(
        pd.Series(log_ret).rolling(window, min_periods=window).std().values, min_std
    )
    lr_z = (log_ret - lr_mean) / (lr_std + eps)

    X = np.column_stack([lr_z, vol_z, hl_z])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.clip(X, -10.0, 10.0, out=X)

    valid = np.isfinite(X).all(axis=1)
    valid[: window] = False
    return X, valid
