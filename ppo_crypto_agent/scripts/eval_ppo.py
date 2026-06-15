"""Проверка сохранённой PPO: метрики на отрезке данных (как при eval при обучении)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.crypto_env import CryptoTradingEnv  # noqa: E402
from utils.data import build_features, load_ohlcv_csv  # noqa: E402


def make_env(
    df,
    window,
    commission,
    turnover_penalty,
    start,
    end,
    *,
    short: bool,
    funding_per_step: float,
    funding_series: np.ndarray | None,
    max_abs_delta_w: float | None,
):
    close = df["close"].values
    X, valid = build_features(df, window=window)
    w_min, w_max = (-1.0, 1.0) if short else (0.0, 1.0)

    def _init():
        return Monitor(
            CryptoTradingEnv(
                close=close,
                features=X,
                valid=valid,
                window=window,
                commission=commission,
                turnover_penalty=turnover_penalty,
                start_idx=start,
                end_idx=end,
                max_episode_steps=None,
                w_min=w_min,
                w_max=w_max,
                funding_per_step=funding_per_step,
                funding_rates=funding_series,
                max_abs_delta_w=max_abs_delta_w,
            )
        )

    return _init


def buy_hold_balance(
    close: np.ndarray,
    t_start: int,
    t_end: int,
    commission: float,
) -> float:
    """Один раз купили на всё в t_start, держим до t_end (индекс последней цены в эпизоде)."""
    fee = commission
    return float((1.0 - fee) * close[t_end] / max(close[t_start], 1e-12))


def main() -> None:
    p = argparse.ArgumentParser(description="Оценка сохранённой PPO на CSV")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--model", type=Path, default=None, help="ppo_final.zip или best_model.zip")
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--commission", type=float, default=0.0004)
    p.add_argument("--turnover-penalty", type=float, default=0.01)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--episodes", type=int, default=3, help="Сколько полных прогонов оценки")
    p.add_argument(
        "--segment",
        choices=("val", "train"),
        default="val",
        help="val — хвост после train (как EvalCallback); train — только train-интервал",
    )
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Случайные действия из политики (если deterministic даёт оборот 0 — сравните)",
    )
    p.add_argument(
        "--short",
        action="store_true",
        help="Среда long/short w∈[-1,1] (должно совпадать с обучением модели)",
    )
    p.add_argument(
        "--funding-per-step",
        type=float,
        default=0.0,
        help="Как при обучении (0=выкл)",
    )
    p.add_argument(
        "--max-delta-w",
        dest="max_delta_w",
        type=float,
        default=None,
        help="Как при обучении (ограничение |Δw| за шаг)",
    )
    args = p.parse_args()

    csv_path = Path(args.csv).expanduser()
    if not csv_path.is_absolute():
        csv_path = (ROOT / csv_path).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    save_dir = ROOT / "runs" / "ppo_crypto"
    if args.model is None:
        best = save_dir / "best" / "best_model.zip"
        final = save_dir / "ppo_final.zip"
        if best.is_file():
            model_path = best
        elif final.is_file():
            model_path = final
        else:
            raise FileNotFoundError(
                f"Не найдена модель. Укажите --model или положите файл в:\n"
                f"  {best}\n  {final}"
            )
    else:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = (ROOT / model_path).resolve()
        if not model_path.is_file():
            # SB3 EvalCallback пишет в …/best/best_model.zip, не в …/best_model.zip
            candidates = [
                model_path.parent / "best" / "best_model.zip",
                model_path.parent / "ppo_final.zip",
            ]
            found = next((p for p in candidates if p.is_file()), None)
            if found is not None:
                print(
                    f"Файл не найден: {model_path}\n"
                    f"Используется: {found}"
                )
                model_path = found
            else:
                raise FileNotFoundError(
                    f"Нет файла модели: {model_path}\n"
                    "Ожидаются, например:\n"
                    f"  {model_path.parent / 'best' / 'best_model.zip'}\n"
                    f"  {model_path.parent / 'ppo_final.zip'}"
                )

    df, funding_csv = load_ohlcv_csv(csv_path)
    n = len(df)
    split = int(n * args.train_ratio)
    split = max(split, args.window + 50)
    split = min(split, n - 50)
    t_end = n - 2

    if args.segment == "val":
        start, end = split, t_end
    else:
        start, end = args.window, split

    close = df["close"].values
    env_fn = make_env(
        df,
        args.window,
        args.commission,
        args.turnover_penalty,
        start,
        end,
        short=args.short,
        funding_per_step=args.funding_per_step,
        funding_series=funding_csv,
        max_abs_delta_w=args.max_delta_w,
    )
    vec = DummyVecEnv([env_fn])

    print(f"Модель: {model_path}")
    print(f"CSV: {csv_path} (строк: {n})")
    print(f"Сегмент: {args.segment}  индексы [{start}, {end}]  длина эпизода ~ {end - start}")

    model = PPO.load(str(model_path), env=vec)

    det = not args.stochastic
    mean_r, std_r = evaluate_policy(
        model,
        vec,
        n_eval_episodes=args.episodes,
        deterministic=det,
        render=False,
    )
    print(
        f"\n=== SB3 evaluate_policy ({args.episodes} эпизодов, "
        f"{'deterministic' if det else 'stochastic'}) ==="
    )
    print(f"Средняя суммарная награда за эпизод: {mean_r:.4f} +/- {std_r:.4f}")

    bh = buy_hold_balance(close, start, end, args.commission)
    print("\n=== Эталон buy & hold (одна покупка в начале сегмента, без ребаланса) ===")
    print(f"Итоговый множитель капитала (прибл.): {bh:.6f}")

    X, valid = build_features(df, args.window)
    w_min, w_max = (-1.0, 1.0) if args.short else (0.0, 1.0)
    raw = CryptoTradingEnv(
        close=close,
        features=X,
        valid=valid,
        window=args.window,
        commission=args.commission,
        turnover_penalty=args.turnover_penalty,
        start_idx=start,
        end_idx=end,
        max_episode_steps=None,
        w_min=w_min,
        w_max=w_max,
        funding_per_step=args.funding_per_step,
        funding_rates=funding_csv,
        max_abs_delta_w=args.max_delta_w,
    )
    obs, _ = raw.reset(seed=0)
    total_reward = 0.0
    turnover_sum = 0.0
    steps = 0
    while True:
        action, _ = model.predict(obs, deterministic=det)
        obs, r, terminated, truncated, info = raw.step(action)
        total_reward += float(r)
        turnover_sum += float(info.get("turnover", 0.0))
        steps += 1
        if terminated or truncated:
            break

    fin_bal = float(raw._balance)  # noqa: SLF001 — итог симуляции
    print("\n=== Один полный прогон агента (детально) ===")
    print(f"Шагов: {steps}")
    print(f"Сумма наград за эпизод: {total_reward:.6f}")
    print(f"Итоговый баланс (из среды): {fin_bal:.6f}")
    print(f"Суммарный оборот |Δw|: {turnover_sum:.4f}")
    print(f"Лог-доходность vs начало: {np.log(max(fin_bal, 1e-12)):.6f}")
    print(f"Buy&hold множитель (см. выше): {bh:.6f}")

    if turnover_sum < 1e-6 and not args.stochastic:
        print(
            "\n(Замечание) Оборот 0 при deterministic — часто среднее действие политики "
            "оказывается ≤0 и после обрезки в [0,1] получается полная наличность. "
            "Запустите с --stochastic или переобучите с меньшим --turnover-penalty."
        )

    if model_path.parent.name == "best":
        run_dir = model_path.parent.parent
    else:
        run_dir = model_path.parent
    tb_dir = run_dir / "tb"
    if not tb_dir.is_dir():
        tb_dir = ROOT / "runs" / "ppo_crypto" / "tb"
    print(
        "\nTensorBoard (если обучали с --tensorboard):\n"
        f"  py -3 -m tensorboard.main --logdir {tb_dir}"
    )


if __name__ == "__main__":
    main()
