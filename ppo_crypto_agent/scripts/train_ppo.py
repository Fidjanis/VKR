"""Обучение PPO на CSV со свечами (колонки: open, high, low, close, volume)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import sys

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
        env = CryptoTradingEnv(
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
        return Monitor(env)

    return _init


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True, help="Путь к OHLCV CSV")
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--commission", type=float, default=0.0004)
    p.add_argument("--turnover-penalty", type=float, default=0.01)
    p.add_argument("--train-ratio", type=float, default=0.7, help="Доля строк под train")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=Path, default=ROOT / "runs" / "ppo_crypto")
    p.add_argument(
        "--tensorboard",
        action="store_true",
        help="Логи в TensorBoard (нужен пакет tensorboard)",
    )
    p.add_argument(
        "--short",
        action="store_true",
        help="Long/short: позиция w в [-1, 1] (−1 шорт, 0 вне рынка, +1 лонг)",
    )
    p.add_argument(
        "--funding-per-step",
        type=float,
        default=0.0,
        help="Константа funding за шаг, если в CSV нет колонки funding_rate",
    )
    p.add_argument(
        "--max-delta-w",
        dest="max_delta_w",
        type=float,
        default=None,
        help="Ограничить |Δw| за шаг (например 0.12), чтобы снизить переторговку",
    )
    args = p.parse_args()

    csv_path = Path(args.csv).expanduser()
    if not csv_path.is_absolute():
        csv_path = (ROOT / csv_path).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Нет файла: {csv_path}\n"
            "Сначала создайте пример данных командой:\n"
            "  py -3 scripts\\make_sample_csv.py"
        )

    df, funding_csv = load_ohlcv_csv(csv_path)
    n = len(df)
    split = int(n * args.train_ratio)
    split = max(split, args.window + 50)
    split = min(split, n - 50)

    train_env = DummyVecEnv(
        [
            make_env(
                df,
                args.window,
                args.commission,
                args.turnover_penalty,
                start=None,
                end=split,
                short=args.short,
                funding_per_step=args.funding_per_step,
                funding_series=funding_csv,
                max_abs_delta_w=args.max_delta_w,
            )
        ]
    )
    eval_env = DummyVecEnv(
        [
            make_env(
                df,
                args.window,
                args.commission,
                args.turnover_penalty,
                start=split,
                end=n - 2,
                short=args.short,
                funding_per_step=args.funding_per_step,
                funding_series=funding_csv,
                max_abs_delta_w=args.max_delta_w,
            )
        ]
    )

    args.save.mkdir(parents=True, exist_ok=True)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(args.save / "best"),
        log_path=str(args.save / "logs"),
        eval_freq=max(500, min(10_000, args.timesteps // 5)),
        deterministic=True,
        render=False,
    )

    tb_log = str(args.save / "tb") if args.tensorboard else None
    if args.tensorboard:
        try:
            import tensorboard  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "Включён --tensorboard, но пакет не установлен. Выполните:\n"
                "  py -3 -m pip install tensorboard\n"
                "или уберите флаг --tensorboard."
            ) from e

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=args.seed,
        tensorboard_log=tb_log,
        policy_kwargs=dict(ortho_init=True),
    )
    model.learn(total_timesteps=args.timesteps, callback=eval_cb)
    model.save(str(args.save / "ppo_final"))
    print(f"Модель: {args.save / 'ppo_final.zip'}")


if __name__ == "__main__":
    main()
