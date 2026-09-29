"""Augmentation et filtrage des séries de prix (papier §4.3).

Réécriture de `dataAugmentation.py`. Chaque fonction prend et renvoie un
DataFrame OHLCV, sans modifier l'entrée.
"""

import itertools

import numpy as np
import pandas as pd


def shift_volume(df: pd.DataFrame, magnitude: float = 0.0) -> pd.DataFrame:
    """Décale la série des volumes (`dataAugmentation.py:55-78`). Le papier parle
    de « signal shifting » ; le code ne décale que le volume."""
    out = df.copy()
    if magnitude < 0:
        magnitude = max(-out["Volume"].min(), magnitude)
    out["Volume"] += magnitude
    return out


def stretch(df: pd.DataFrame, factor: float = 1.0) -> pd.DataFrame:
    """Multiplie les rendements journaliers par `factor` (`dataAugmentation.py:81-105`,
    technique absente du texte du papier)."""
    if factor == 1.0:
        return df.copy()
    out = df.copy()
    returns = df["Close"].pct_change().fillna(0.0).to_numpy() * factor
    close = df["Close"].iloc[0] * np.cumprod(1.0 + returns)
    ratio = close / df["Close"].to_numpy()
    out["Close"] = close
    out["Low"] = df["Low"].to_numpy() * ratio
    out["High"] = df["High"].to_numpy() * ratio
    out["Open"] = out["Close"].shift(1).fillna(df["Open"].iloc[0])
    return out


def add_noise(df: pd.DataFrame, stdev_pct: float = 0.0, rng=None) -> pd.DataFrame:
    """Bruit gaussien multiplicatif d'écart-type `stdev_pct` %.

    ADAPTATION : dans `dataAugmentation.py:125-134`, l'écart-type vaut
    `stdev * prix / 100` puis est appliqué en `prix *= 1 + bruit / 100` ; le
    bruit relatif vaut donc `stdev * prix / 10 000` et explose pour le BTC.
    Ici il est relatif et indépendant du niveau de prix.
    """
    if stdev_pct == 0:
        return df.copy()
    rng = rng or np.random.default_rng()
    out = df.copy()
    n = len(df)
    price_noise = 1.0 + rng.normal(0.0, stdev_pct / 100, n)
    volume_noise = 1.0 + rng.normal(0.0, stdev_pct / 100, n)
    price_noise[0] = volume_noise[0] = 1.0
    for col in ("Close", "Low", "High"):
        out[col] = df[col].to_numpy() * price_noise
    out["Volume"] = df["Volume"].to_numpy() * volume_noise
    out["Open"] = out["Close"].shift(1).fillna(df["Open"].iloc[0])
    return out


def low_pass_filter(df: pd.DataFrame, order: int = 5) -> pd.DataFrame:
    """Moyenne glissante causale d'ordre `order` (`dataAugmentation.py:141-169`).
    Les `order` premières valeurs restent brutes."""
    out = df.copy()
    for col in ("Close", "Low", "High", "Volume"):
        smoothed = df[col].rolling(window=order).mean()
        smoothed.iloc[:order] = df[col].iloc[:order]
        out[col] = smoothed
    out["Open"] = out["Close"].shift(1).fillna(df["Open"].iloc[0])
    return out


def generate(df: pd.DataFrame, shift_range, stretch_range, filter_range, noise_range,
             rng=None) -> list:
    """Produit cartésien des plages, comme `dataAugmentation.py:172-193`."""
    return [
        add_noise(low_pass_filter(stretch(shift_volume(df, s), k), f), n, rng)
        for s, k, f, n in itertools.product(shift_range, stretch_range, filter_range, noise_range)
    ]
