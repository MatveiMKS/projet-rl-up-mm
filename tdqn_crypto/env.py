"""Environnement de trading (papier §3.3-3.4).

Réécriture de `tradingEnv.py` (classe `TradingEnv`, l.46-423). Mêmes règles :
- deux actions : 1 = long, 0 = short (éq. 15) ;
- long : tout le cash est converti en actifs ; short : on vend à découvert la
  même quantité, sous la contrainte de solvabilité (éq. 13, `computeLowerBound`) ;
- récompense = variation relative de la valeur du portefeuille (éq. 19) ;
- `step` calcule aussi la transition de l'action opposée, que l'agent stocke
  pour mieux explorer (§4.2, `tradingEnv.py:298-358`).

Chronologie, identique au code de référence : l'observation au pas t couvre
les jours [t-L, t-1], l'ordre s'exécute au Close du jour t, et la récompense
est la variation de valeur entre les Close t-1 et t. Elle mesure donc surtout
la position tenue depuis la veille, plus les frais de l'ordre du jour ; l'effet
du nouvel ordre sur les prix arrive au pas suivant, via la position incluse
dans l'état.

Différences avec la référence :
- quantités fractionnaires possibles (voir `EnvConfig.fractional_shares`) ;
- une seule fonction de transition sert aux deux actions. Dans la référence,
  la branche « action opposée » long→short utilise `self.numberOfShares` au
  lieu de la quantité recalculée (`tradingEnv.py:344`) ;
- pas de dépendance à `gym` : `reset()` renvoie l'observation, `step()` le
  quadruplet (observation, récompense, fin, info).
"""

import math

import numpy as np
import pandas as pd

from .config import EnvConfig


class TradingEnv:
    def __init__(self, data: pd.DataFrame, config: EnvConfig = None):
        self.config = config or EnvConfig()
        if len(data) <= self.config.state_length:
            raise ValueError("Pas assez de données pour une seule observation.")
        self.data = data
        self.dates = data.index
        self.close = data["Close"].to_numpy(dtype=float)
        self.low = data["Low"].to_numpy(dtype=float)
        self.high = data["High"].to_numpy(dtype=float)
        self.volume = data["Volume"].to_numpy(dtype=float)
        self.reset()

    def __len__(self):
        return len(self.close)

    # ------------------------------------------------------------------ état
    def reset(self):
        n, cash = len(self.close), float(self.config.initial_cash)
        self.position = np.zeros(n, dtype=int)   # -1 short, 0 neutre, 1 long
        self.action = np.zeros(n, dtype=int)     # 1 achat, -1 vente, 0 rien
        self.cash = np.full(n, cash)
        self.holdings = np.zeros(n)
        self.money = np.full(n, cash)
        self.returns = np.zeros(n)
        self.shares = 0.0                        # quantité détenue (valeur absolue)
        self.t = self.config.state_length
        self.done = False
        return self.observation()

    def set_starting_point(self, t: int):
        """Démarre l'épisode au pas `t` (`tradingEnv.py:403-423`)."""
        self.t = int(np.clip(t, self.config.state_length, len(self.close)))
        self.done = self.t >= len(self.close)
        return self.observation()

    def observation(self, position: int = None):
        """Fenêtre brute [Close, Low, High, Volume] des L derniers jours et
        position courante ; la normalisation est faite par l'agent."""
        lo, hi = self.t - self.config.state_length, self.t
        if position is None:
            position = int(self.position[self.t - 1])
        return {
            "close": self.close[lo:hi],
            "low": self.low[lo:hi],
            "high": self.high[lo:hi],
            "volume": self.volume[lo:hi],
            "position": position,
        }

    # ------------------------------------------------------------ transition
    def _quantity(self, x: float) -> float:
        return x if self.config.fractional_shares else float(math.floor(x))

    def _lower_bound(self, cash: float, signed_shares: float, price: float) -> float:
        """Borne inférieure du nombre d'actifs à échanger (annexe du papier ;
        `tradingEnv.py:193-211`)."""
        eps, c = self.config.epsilon, self.config.transaction_costs
        delta = -cash - signed_shares * price * (1 + eps) * (1 + c)
        if delta < 0:
            return delta / (price * (2 * c + eps * (1 + c)))
        return delta / (price * eps * (1 + c))

    def _transition(self, action: int, t: int):
        """Résultat de `action` au pas t, sans modifier l'environnement.
        Renvoie (position, cash, quantité, valeur, récompense)."""
        c = self.config.transaction_costs
        price, prev_price = self.close[t], self.close[t - 1]
        prev_pos, prev_cash, prev_shares = self.position[t - 1], self.cash[t - 1], self.shares
        forced_reward = None

        if action == 1:                                        # LONG
            position = 1
            if prev_pos == 1:
                cash, shares = prev_cash, prev_shares
            else:
                cash = prev_cash
                if prev_pos == -1:                             # rachat du short
                    cash -= prev_shares * price * (1 + c)
                shares = self._quantity(cash / (price * (1 + c)))
                cash -= shares * price * (1 + c)
            holdings = shares * price
        elif action == 0:                                      # SHORT
            position = -1
            if prev_pos == -1:
                bound = self._lower_bound(prev_cash, -prev_shares, prev_price)
                if bound <= 0:
                    cash, shares = prev_cash, prev_shares
                else:                                          # rachat partiel imposé
                    to_buy = min(self._quantity(bound), prev_shares)
                    shares = prev_shares - to_buy
                    cash = prev_cash - to_buy * price * (1 + c)
                    forced_reward = (prev_price - price) / prev_price
            else:
                cash = prev_cash
                if prev_pos == 1:                              # vente du long
                    cash += prev_shares * price * (1 - c)
                # Même dimensionnement que la référence : cash/(p(1+c)).
                shares = self._quantity(cash / (price * (1 + c)))
                cash += shares * price * (1 - c)
            holdings = -shares * price
        else:
            raise ValueError("Action interdite : 1 (long) ou 0 (short).")

        money = cash + holdings
        prev_money = self.money[t - 1]
        reward = forced_reward if forced_reward is not None else (money - prev_money) / prev_money
        return position, cash, shares, money, reward

    def step(self, action: int):
        t = self.t
        position, cash, shares, money, reward = self._transition(action, t)
        other = 1 - action
        o_position, _, _, _, o_reward = self._transition(other, t)

        prev_pos = self.position[t - 1]
        self.position[t], self.cash[t], self.holdings[t] = position, cash, money - cash
        self.money[t] = money
        self.returns[t] = (money - self.money[t - 1]) / self.money[t - 1]
        if position != prev_pos:
            self.action[t] = position
        self.shares = shares

        self.t += 1
        self.done = self.t >= len(self.close)
        info = {
            "other_action": other,
            "other_observation": self.observation(position=o_position),
            "other_reward": o_reward,
            "other_done": self.done,
        }
        return self.observation(), reward, self.done, info

    # ------------------------------------------------------------- résultats
    def history(self) -> pd.DataFrame:
        """Journal de l'activité, au format attendu par `metrics`."""
        return pd.DataFrame({
            "Close": self.close, "Position": self.position, "Action": self.action,
            "Cash": self.cash, "Holdings": self.holdings, "Money": self.money,
            "Returns": self.returns,
        }, index=self.dates)
