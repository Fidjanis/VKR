"""Создаёт пример OHLCV для проверки пайплайна (синтетический ряд)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    ret = rng.normal(0.0002, 0.01, size=n)
    close = 100 * np.exp(np.cumsum(ret))
    noise = rng.normal(0, 0.002, size=n)
    high = close * (1 + np.abs(noise) + 0.002)
    low = close * (1 - np.abs(noise) - 0.002)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.lognormal(10, 0.5, size=n)

    # Корень проекта ppo_crypto_agent (на уровень выше папки scripts)
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "sample_ohlcv.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    # open_time UTC ms — для склейки с funding Binance (merge_binance_funding.py)
    base_ms = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp() * 1000)
    open_time = base_ms + np.arange(n, dtype=np.int64) * 3_600_000

    pd.DataFrame(
        {
            "open_time": open_time,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    ).to_csv(out, index=False)
    print(f"Записано: {out}")


if __name__ == "__main__":
    main()
