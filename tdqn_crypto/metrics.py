"""Indicateurs de performance (papier §5.2, Table 1).

Réécriture de `tradingPerformance.py` (`PerformanceEstimator`, l.24).

ADAPTATION : le facteur d'annualisation est un paramètre (`periods_per_year`).
La référence fixe 252 séances (l.120, 143, 168) ; le BTC cote 365 jours par an.
Écart assumé : le rendement annualisé est calculé sur la valeur composée du
portefeuille, là où la référence compose la *somme* des rendements journaliers
(`tradingPerformance.py:81-106`).
"""

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 365, risk_free: float = 0.0) -> float:
    mean, std = returns.mean(), returns.std()
    if mean == 0 or std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(periods_per_year) * (mean - risk_free) / std)


def sortino_ratio(returns: pd.Series, periods_per_year: int = 365, risk_free: float = 0.0) -> float:
    downside = np.std(returns[returns < 0])
    mean = returns.mean()
    if mean == 0 or downside == 0 or np.isnan(downside):
        return 0.0
    return float(np.sqrt(periods_per_year) * (mean - risk_free) / downside)


def max_drawdown(money: pd.Series):
    """Perte maximale depuis un sommet (%) et sa durée (en pas)."""
    capital = money.to_numpy()
    trough = int(np.argmax(np.maximum.accumulate(capital) - capital))
    if trough == 0:
        return 0.0, 0
    peak = int(np.argmax(capital[:trough]))
    return float(100 * (capital[peak] - capital[trough]) / capital[peak]), trough - peak


def profitability(history: pd.DataFrame):
    """% de trades gagnants et ratio gain moyen / perte moyenne
    (`tradingPerformance.py:213-272`). Un trade va d'un changement de position
    au suivant, le dernier se clôt à la fin de l'horizon."""
    trades = np.flatnonzero(history["Action"].to_numpy() != 0)
    if len(trades) == 0:
        return 0.0, 0.0
    money = history["Money"].to_numpy()
    marks = np.append(money[trades], money[-1])
    deltas = np.diff(marks)
    gains, losses = deltas[deltas >= 0], -deltas[deltas < 0]
    ratio = (gains.mean() if len(gains) else 0.0) / losses.mean() if len(losses) else float("inf")
    return float(100 * len(gains) / len(deltas)), float(ratio)


def performance(history: pd.DataFrame, periods_per_year: int = 365) -> dict:
    returns, money = history["Returns"], history["Money"]
    days = max((history.index[-1] - history.index[0]).days, 1)
    total = money.iloc[-1] / money.iloc[0]
    annual = 100 * (total ** (365 / days) - 1) if total > 0 else -100.0
    dd, ddd = max_drawdown(money)
    prof, pl_ratio = profitability(history)
    return {
        "PnL": float(money.iloc[-1] - money.iloc[0]),
        "Annualized return (%)": float(annual),
        "Annualized volatility (%)": float(100 * np.sqrt(periods_per_year) * returns.std()),
        "Sharpe ratio": sharpe_ratio(returns, periods_per_year),
        "Sortino ratio": sortino_ratio(returns, periods_per_year),
        "Max drawdown (%)": dd,
        "Max drawdown duration (steps)": ddd,
        "Profitability (%)": prof,
        "Avg profit/loss ratio": pl_ratio,
        "Skewness": float(returns.skew()),
    }
