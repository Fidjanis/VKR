"""
Торговая среда Gymnasium: одна пара, целевая позиция w в активе,
награда — лог-доходность портфеля с учётом комиссии от оборота.

По умолчанию только long: w ∈ [0, 1]. Режим long/short: w ∈ [-1, 1]
(−1 полный шорт, 0 вне рынка, +1 полный лонг). Доходность за шаг:
множитель капитала = 1 + w * (r − 1), где r = P_{t+1}/P_t.

Funding: либо константа ``funding_per_step``, либо массив ``funding_rates[t]``
(например из истории Binance, см. scripts/merge_binance_funding.py). Положительная
ставка: лонг (w>0) платит, шорт (w<0) получает, множитель баланса ``1 - rate*w``.
Опционально ``max_abs_delta_w``: ограничение |Δw| за шаг (снижает переторговку,
ближе к поэтапному исполнению).
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CryptoTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        close: np.ndarray,
        features: np.ndarray,
        valid: np.ndarray,
        window: int,
        commission: float = 0.0004,
        turnover_penalty: float = 0.01,
        initial_balance: float = 1.0,
        max_episode_steps: int | None = 2048,
        start_idx: int | None = None,
        end_idx: int | None = None,
        *,
        w_min: float = 0.0,
        w_max: float = 1.0,
        funding_per_step: float = 0.0,
        funding_rates: np.ndarray | None = None,
        max_abs_delta_w: float | None = None,
    ) -> None:
        super().__init__()
        self.close = np.asarray(close, dtype=np.float64)
        self.features = np.asarray(features, dtype=np.float32)
        self.valid = np.asarray(valid, dtype=bool)
        self.window = int(window)
        self.commission = float(commission)
        self.turnover_penalty = float(turnover_penalty)
        self.initial_balance = float(initial_balance)
        self.max_episode_steps = max_episode_steps
        self._forced_start = start_idx
        self._forced_end = end_idx
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.funding_per_step = float(funding_per_step)
        self.funding_rates: np.ndarray | None = None
        if funding_rates is not None:
            fr = np.asarray(funding_rates, dtype=np.float64)
            if len(fr) != len(self.close):
                raise ValueError("funding_rates должен совпадать по длине с close")
            self.funding_rates = fr
        self.max_abs_delta_w = (
            None
            if max_abs_delta_w is None or float(max_abs_delta_w) <= 0
            else float(max_abs_delta_w)
        )
        if self.w_min >= self.w_max:
            raise ValueError("Нужно w_min < w_max")

        n = len(self.close)
        self._global_start = self.window
        self._global_end = n - 2
        if self._forced_start is not None:
            self._global_start = max(self._global_start, int(self._forced_start))
        if self._forced_end is not None:
            self._global_end = min(self._global_end, int(self._forced_end))

        feat_dim = self.features.shape[1]
        obs_dim = self.window * feat_dim + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.float32(self.w_min),
            high=np.float32(self.w_max),
            shape=(1,),
            dtype=np.float32,
        )

        self._t = 0
        self._steps = 0
        self._w = 0.0
        self._balance = self.initial_balance

    def _obs_at(self, t: int) -> np.ndarray:
        sl = self.features[t - self.window : t, :].reshape(-1)
        log_b = float(np.log(np.clip(self._balance, 1e-12, 1e12)))
        obs = np.concatenate(
            [sl, [float(np.clip(self._w, self.w_min, self.w_max)), log_b]]
        ).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        low = self._global_start
        high = self._global_end - 1
        if high <= low:
            raise ValueError("Слишком короткий ряд для заданного window и диапазона индексов.")

        if self._forced_start is not None and self._forced_end is not None:
            self._episode_start = int(self._forced_start)
        else:
            candidates = np.where(self.valid[low : high + 1])[0] + low
            if len(candidates) == 0:
                self._episode_start = int(self.np_random.integers(low, high))
            else:
                self._episode_start = int(self.np_random.choice(candidates))

        self._t = self._episode_start
        self._steps = 0
        self._w = 0.0
        self._balance = self.initial_balance
        return self._obs_at(self._t), {}

    def step(self, action: Any) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        w_old = self._w
        raw = float(np.asarray(action, dtype=np.float64).flat[0])
        if not np.isfinite(raw):
            raw = w_old
        w_new = float(np.clip(np.nan_to_num(raw, nan=w_old), self.w_min, self.w_max))
        if self.max_abs_delta_w is not None:
            dw = w_new - w_old
            dw = float(np.clip(dw, -self.max_abs_delta_w, self.max_abs_delta_w))
            w_new = float(np.clip(w_old + dw, self.w_min, self.w_max))
        turnover = abs(w_new - w_old)

        ret = self.close[self._t + 1] / max(self.close[self._t], 1e-12)
        ret = float(np.clip(ret, 1e-6, 1e6))
        # Унифицированно long и short: множитель = 1 + w * (r − 1)
        port_ret = 1.0 + w_new * (ret - 1.0)
        # комиссия как доля капитала, пропорциональная обороту (упрощённая модель)
        fee = self.commission * turnover
        growth = max(port_ret * (1.0 - fee), 1e-12)

        prev_b = self._balance
        self._balance = max(prev_b * growth, 1e-12)
        if self.funding_rates is not None:
            fr = float(self.funding_rates[self._t])
        else:
            fr = self.funding_per_step
        if fr != 0.0:
            # положительный funding: лонг (w>0) платит, шорт (w<0) получает
            self._balance *= max(1.0 - fr * w_new, 1e-12)
        self._w = w_new

        reward = float(np.log(self._balance / max(prev_b, 1e-12)))
        reward -= self.turnover_penalty * turnover
        reward = float(np.clip(np.nan_to_num(reward, nan=0.0), -50.0, 50.0))

        self._t += 1
        self._steps += 1

        terminated = self._t >= self._global_end
        truncated = bool(
            self.max_episode_steps is not None and self._steps >= self.max_episode_steps
        )

        info = {
            "turnover": turnover,
            "weight": self._w,
            "balance": self._balance,
        }
        return self._obs_at(self._t), reward, terminated, truncated, info
