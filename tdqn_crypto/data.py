"""Chargement des données OHLCV journalières.

Inspiré de `dataDownloader.py` (Yahoo Finance via `pandas_datareader`, l.158-222)
et du cache CSV de `tradingEnv.py:102-119`. `pandas_datareader` ne fonctionne
plus avec Yahoo ; on passe par `yfinance`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download_yahoo(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Télécharge les prix journaliers depuis Yahoo Finance (`end` exclu)."""
    import yfinance as yf

    raw = yf.download(symbol, start=start, end=end, interval="1d",
                      auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError(f"Aucune donnée Yahoo pour {symbol} entre {start} et {end}.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs(symbol, axis=1, level="Ticker")
    df = raw[COLUMNS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Même nettoyage que `tradingEnv.py:121-127` : les zéros sont traités comme
    des valeurs manquantes, interpolées (5 pas max), puis complétées."""
    df = df[COLUMNS].astype(float).replace(0.0, np.nan)
    df = df.interpolate(method="linear", limit=5, limit_area="inside")
    return df.ffill().bfill().fillna(0.0)


def load_prices(symbol: str, start: str, end: str, cache_dir: str = "data") -> pd.DataFrame:
    """Charge depuis le cache CSV, sinon télécharge puis met en cache."""
    path = Path(cache_dir) / f"{symbol}_{start}_{end}.csv"
    if path.exists():
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
    else:
        df = download_yahoo(symbol, start, end)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
    return clean(df)


def split_by_dates(df: pd.DataFrame, *boundaries: str) -> list:
    """Découpe chronologiquement : `split_by_dates(df, "2021", "2023")` renvoie
    [avant 2021, 2021-2022, à partir de 2023]. Chaque borne appartient au bloc
    qui commence à cette date."""
    cuts = [pd.Timestamp(b) for b in boundaries]
    edges = [df.index.min()] + cuts
    parts = []
    for i, lo in enumerate(edges):
        hi = cuts[i] if i < len(cuts) else None
        part = df[df.index >= lo] if hi is None else df[(df.index >= lo) & (df.index < hi)]
        parts.append(part.copy())
    return parts
