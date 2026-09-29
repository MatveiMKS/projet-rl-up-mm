import numpy as np
import pandas as pd
import pytest

from tdqn_crypto import augmentation, baselines, metrics
from tdqn_crypto.agent import TDQN, normalization_coefficients, process_state
from tdqn_crypto.config import AgentConfig, EnvConfig
from tdqn_crypto.data import split_by_dates
from tdqn_crypto.env import TradingEnv


def synthetic_prices(n=200, start_price=30_000.0, seed=0):
    rng = np.random.default_rng(seed)
    close = start_price * np.cumprod(1 + rng.normal(0.001, 0.03, n))
    spread = np.abs(rng.normal(0, 0.02, n)) * close
    return pd.DataFrame({
        "Open": np.r_[close[0], close[:-1]], "High": close + spread, "Low": close - spread,
        "Close": close, "Volume": rng.uniform(1e9, 5e10, n),
    }, index=pd.date_range("2020-01-01", periods=n, freq="D", name="Date"))


def test_observation_size_matches_reference():
    cfg = EnvConfig()
    df = synthetic_prices()
    obs = TradingEnv(df, cfg).reset()
    state = process_state(obs, normalization_coefficients(df))
    assert state.shape == (cfg.observation_size,) == (117,)


def test_buy_and_hold_tracks_price_with_fractional_shares():
    df = synthetic_prices()
    cfg = EnvConfig(transaction_costs=0.0)
    history = baselines.buy_and_hold(df, cfg)
    t0 = cfg.state_length
    expected = cfg.initial_cash * df["Close"].iloc[-1] / df["Close"].iloc[t0]
    assert history["Money"].iloc[-1] == pytest.approx(expected)


def test_integer_shares_leave_cash_idle():
    df = synthetic_prices()
    cfg = EnvConfig(transaction_costs=0.0, fractional_shares=False)
    env = TradingEnv(df, cfg)
    env.step(1)
    assert env.shares == np.floor(cfg.initial_cash / df["Close"].iloc[cfg.state_length])
    assert env.cash[cfg.state_length] > 0


def test_other_action_matches_real_step():
    """La transition « action opposée » renvoyée dans info doit coïncider avec
    ce qu'aurait donné un vrai pas avec cette action."""
    df = synthetic_prices()
    rng = np.random.default_rng(1)
    for _ in range(20):
        actions = rng.integers(0, 2, 15)
        a, b = TradingEnv(df), TradingEnv(df)
        for act in actions[:-1]:
            a.step(int(act))
            b.step(int(act))
        _, _, _, info = a.step(int(actions[-1]))
        _, reward, _, _ = b.step(int(1 - actions[-1]))
        assert info["other_reward"] == pytest.approx(reward)


def test_noise_is_relative_to_price():
    df = synthetic_prices(n=2000, start_price=80_000.0)
    noisy = augmentation.add_noise(df, stdev_pct=1.0, rng=np.random.default_rng(0))
    rel = (noisy["Close"] / df["Close"] - 1).iloc[1:]
    assert rel.std() == pytest.approx(0.01, rel=0.1)


def test_low_pass_filter_is_causal():
    df = synthetic_prices()
    full = augmentation.low_pass_filter(df, 5)
    truncated = augmentation.low_pass_filter(df.iloc[:100], 5)
    pd.testing.assert_series_equal(full["Close"].iloc[:100], truncated["Close"])


def test_split_by_dates():
    df = synthetic_prices(n=400)
    a, b, c = split_by_dates(df, "2020-06-01", "2020-09-01")
    assert len(a) + len(b) + len(c) == len(df)
    assert a.index.max() < pd.Timestamp("2020-06-01") <= b.index.min()
    assert b.index.max() < pd.Timestamp("2020-09-01") <= c.index.min()


def test_sharpe_uses_annualisation_factor():
    r = pd.Series([0.01, -0.005, 0.02, 0.0])
    assert metrics.sharpe_ratio(r, 365) / metrics.sharpe_ratio(r, 252) == pytest.approx(np.sqrt(365 / 252))


def test_tdqn_smoke():
    df = synthetic_prices(n=260)
    train, validation = df.iloc[:200], df.iloc[200:]
    agent = TDQN(config=AgentConfig(neurons=32, batch_size=8), device="cpu")
    history = agent.train(train, validation, episodes=2, verbose=False)
    assert len(history) == 2 and "validation_sharpe" in history[0]
    result = agent.test(validation, context=train)
    assert len(result) == len(validation) + 1
    assert result["Money"].iloc[0] == EnvConfig().initial_cash
