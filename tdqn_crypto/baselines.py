"""Stratégies de comparaison (papier §5.3).

Réécriture de `classicalStrategy.py` : Buy and Hold (l.106) et Sell and Hold
(l.228), exécutées dans le même environnement que l'agent pour avoir les mêmes
frais. Les stratégies à moyennes mobiles (l.350, l.624) sont à ajouter.
"""

import pandas as pd

from .config import EnvConfig
from .env import TradingEnv


def constant_position(df: pd.DataFrame, action: int, env_config: EnvConfig = None,
                      context: pd.DataFrame = None) -> pd.DataFrame:
    """Garde la même action tout l'horizon (1 = Buy and Hold, 0 = Sell and Hold).
    `context` a le même rôle que dans `TDQN.test`, pour comparer sur les mêmes jours."""
    env_config = env_config or EnvConfig()
    L = env_config.state_length
    full = df if context is None else pd.concat([context.iloc[-L:], df])
    env = TradingEnv(full, env_config)
    done = False
    while not done:
        _, _, done, _ = env.step(action)
    history = env.history()
    return history.iloc[L - 1:] if context is not None else history


def buy_and_hold(df, env_config=None, context=None):
    return constant_position(df, 1, env_config, context)


def sell_and_hold(df, env_config=None, context=None):
    return constant_position(df, 0, env_config, context)
