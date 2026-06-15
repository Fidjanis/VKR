"""
Подтягивает исторические ставки funding USDT-M с Binance и добавляет колонку
``funding_rate`` в CSV со свечами.

Требуется колонка ``open_time`` — время открытия свечи в миллисекундах (UTC),
как в выгрузке klines с Binance.

Важно: биржа начисляет funding раз в ~8 часов на весь нотионал; ставка в API —
за весь интервал. Если у вас часовые свечи и вы применяете ставку на каждую
свечу в среде, завысите списание в ~8 раз. Используйте --funding-scale 0.125
(1/8) как грубую поправку или работайте с 8-часовыми свечами.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BINANCE_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"


def fetch_funding_history(symbol: str, start_ms: int, end_ms: int) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (funding_time_ms, funding_rate) отсортировано по времени."""
    times: list[int] = []
    rates: list[float] = []
    cursor = start_ms
    while cursor <= end_ms:
        q = urllib.parse.urlencode(
            {
                "symbol": symbol.upper().replace("/", ""),
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            }
        )
        url = f"{BINANCE_FUNDING}?{q}"
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                chunk = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise SystemExit(f"Binance HTTP {e.code}: {e.read()[:500]!r}") from e
        except urllib.error.URLError as e:
            raise SystemExit(f"Сеть / URL: {e}") from e
        if not chunk:
            break
        for row in chunk:
            times.append(int(row["fundingTime"]))
            rates.append(float(row["fundingRate"]))
        last_t = times[-1]
        if last_t >= end_ms or len(chunk) < 1000:
            break
        cursor = last_t + 1
    if not times:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    order = np.argsort(times)
    return np.asarray(times, dtype=np.int64)[order], np.asarray(rates, dtype=np.float64)[order]


def assign_funding_to_candles(
    open_time: np.ndarray,
    ft: np.ndarray,
    fr: np.ndarray,
) -> np.ndarray:
    """Для каждой свечи — последняя ставка funding с fundingTime <= open_time."""
    out = np.zeros(len(open_time), dtype=np.float64)
    if len(ft) == 0:
        return out
    idx = np.searchsorted(ft, open_time, side="right") - 1
    mask = idx >= 0
    out[mask] = fr[idx[mask]]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="CSV со свечами + open_time (ms)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--symbol", type=str, default="BTCUSDT", help="Например BTCUSDT")
    p.add_argument(
        "--funding-scale",
        type=float,
        default=1.0,
        help="Множитель к ставке (для 1h свечей Binance попробуйте 0.125)",
    )
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.lower().strip() for c in df.columns]
    if "open_time" not in df.columns:
        raise SystemExit(
            "В CSV должна быть колонка open_time (мс UTC), как у klines Binance. "
            "Скачайте фьючерсные klines с https://data.binance.vision/ или API."
        )
    ot = np.asarray(df["open_time"].values, dtype=np.int64)
    start_ms = int(ot.min())
    end_ms = int(ot.max()) + 86_400_000

    print(f"Загрузка funding {args.symbol} с {start_ms} по {end_ms} …")
    ft, fr = fetch_funding_history(args.symbol, start_ms, end_ms)
    print(f"Получено {len(ft)} записей funding")

    assigned = assign_funding_to_candles(ot, ft, fr) * float(args.funding_scale)
    out_df = df.copy()
    out_df["funding_rate"] = assigned
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Записано: {args.output.resolve()}")


if __name__ == "__main__":
    main()
