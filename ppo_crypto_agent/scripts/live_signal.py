"""
Сигнал по последним свечам: только число w, без отправки ордеров.

Источник данных:
  • по умолчанию — публичный REST Binance USDT-M (без ccxt, без load_markets);
  • при сбоях сети / блокировке — локальный CSV: --from-csv …

Пример:
  py -3 scripts\\live_signal.py --model runs\\my_run2\\best\\best_model.zip --symbol BTC/USDT:USDT --short --max-delta-w 0.12

  py -3 scripts\\live_signal.py --model runs\\my_run2\\best\\best_model.zip --from-csv data\\sample_with_funding.csv --short --max-delta-w 0.12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.data import build_features, load_ohlcv_csv  # noqa: E402
from utils.inference import apply_max_delta, build_observation_vector  # noqa: E402


def _symbol_to_binance_pair(symbol: str) -> str:
    s = symbol.strip().upper()
    if ":" in s:
        s = s.split(":")[0]
    if "/" in s:
        base, quote = s.split("/", 1)
        return f"{base.strip()}{quote.strip()}"
    return s.replace("/", "")


def fetch_klines_binance(
    symbol: str,
    timeframe: str,
    limit: int,
    *,
    base_url: str = "https://fapi.binance.com",
    retries: int = 4,
    timeout_sec: int = 45,
) -> pd.DataFrame:
    pair = _symbol_to_binance_pair(symbol)
    q = urllib.parse.urlencode(
        {"symbol": pair, "interval": timeframe, "limit": min(limit, 1500)}
    )
    url = f"{base_url.rstrip('/')}/fapi/v1/klines?{q}"
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ppo-crypto-agent/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                raw = json.loads(resp.read().decode())
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    else:
        raise SystemExit(
            f"Не удалось загрузить свечи с Binance после {retries} попыток: {last_err}\n"
            "Попробуйте VPN/другую сеть или режим без сети:\n"
            "  --from-csv data\\sample_with_funding.csv"
        ) from last_err

    rows = []
    for k in raw:
        rows.append(
            {
                "open_time": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
        )
    return pd.DataFrame(rows)


def load_tail_from_csv(path: Path, tail: int) -> pd.DataFrame:
    df, _f = load_ohlcv_csv(path)
    if len(df) > tail:
        df = df.iloc[-tail:].copy()
    return df.reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="Без интернета: взять последние свечи из локального CSV (как при обучении)",
    )
    p.add_argument("--symbol", type=str, default="BTC/USDT:USDT")
    p.add_argument("--timeframe", type=str, default="1h")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument(
        "--base-url",
        type=str,
        default="https://fapi.binance.com",
        help="При необходимости другой endpoint (например тестовая сеть Binance)",
    )
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--short", action="store_true")
    p.add_argument("--max-delta-w", dest="max_delta_w", type=float, default=None)
    p.add_argument("--current-weight", type=float, default=0.0)
    p.add_argument("--balance", type=float, default=1.0)
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (ROOT / model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    if args.from_csv is not None:
        csv_path = Path(args.from_csv)
        if not csv_path.is_absolute():
            csv_path = (ROOT / csv_path).resolve()
        df = load_tail_from_csv(csv_path, args.limit)
        src = f"локальный CSV {csv_path.name}"
    else:
        df = fetch_klines_binance(
            args.symbol,
            args.timeframe,
            args.limit,
            base_url=args.base_url,
        )
        src = f"Binance {args.symbol} {args.timeframe}"

    df_feat = df[["open", "high", "low", "close", "volume"]].astype(np.float64)
    close = df_feat["close"].values
    n = len(close)
    if n < args.window + 5:
        raise SystemExit("Слишком мало свечей, увеличьте --limit или CSV")

    X, valid = build_features(df_feat, args.window)
    w_min, w_max = (-1.0, 1.0) if args.short else (0.0, 1.0)

    t = n - 2
    while t >= args.window and not valid[t]:
        t -= 1
    if t < args.window:
        raise SystemExit("Нет валидного t для признаков")

    obs = build_observation_vector(
        X, args.window, t, args.current_weight, args.balance, w_min, w_max
    )

    model = PPO.load(str(model_path), env=None)
    raw, _ = model.predict(obs, deterministic=True)
    w_raw = float(np.asarray(raw, dtype=np.float64).flat[0])
    w_exec = apply_max_delta(
        args.current_weight, w_raw, w_min, w_max, args.max_delta_w
    )

    print("=== Сигнал (не ордер) ===")
    print(f"Источник: {src}  свечей в расчёте: {n}")
    if "open_time" in df.columns:
        print(f"Время последней свечи (ms): {int(df['open_time'].iloc[-1])}")
    print(f"Сырой выход политики w: {w_raw:.6f}")
    print(f"С учётом max |Δw| (если задано): {w_exec:.6f}")
    print(
        "\nДальше вручную: сравните w_exec с лимитами биржи, плечом, маржой. "
        "Для защиты диплома используйте testnet и малый объём."
    )


if __name__ == "__main__":
    main()
