"""Entraîne un agent TDQN sur un actif et compare-le aux baselines.

Exemples :
    python scripts/run_tdqn.py                         # BTC-USD, réglages par défaut
    python scripts/run_tdqn.py --episodes 5            # essai rapide
    python scripts/run_tdqn.py --csv data/btc.csv      # données locales (Date,Open,High,Low,Close,Volume)
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tdqn_crypto import baselines, metrics  # noqa: E402
from tdqn_crypto.agent import TDQN  # noqa: E402
from tdqn_crypto.config import (AgentConfig, AugmentationConfig, DataConfig,  # noqa: E402
                                EnvConfig, TrainConfig)
from tdqn_crypto.data import clean, load_prices, split_by_dates  # noqa: E402


def parse_args():
    d, t = DataConfig(), TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default=d.symbol)
    p.add_argument("--start", default=d.start)
    p.add_argument("--validation-start", default=d.validation_start)
    p.add_argument("--test-start", default=d.test_start)
    p.add_argument("--end", default=d.end)
    p.add_argument("--csv", help="Fichier CSV local à utiliser au lieu de Yahoo Finance.")
    p.add_argument("--episodes", type=int, default=t.episodes)
    p.add_argument("--seed", type=int, default=t.seed)
    p.add_argument("--periods-per-year", type=int, default=t.periods_per_year)
    p.add_argument("--costs", type=float, default=EnvConfig().transaction_costs)
    p.add_argument("--integer-shares", action="store_true",
                   help="Quantités entières, comme le papier (éq. 14).")
    p.add_argument("--device", default=t.device)
    p.add_argument("--out", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    data_cfg = DataConfig(args.symbol, args.start, args.validation_start, args.test_start, args.end)
    env_cfg = EnvConfig(transaction_costs=args.costs, fractional_shares=not args.integer_shares)
    agent_cfg, aug_cfg = AgentConfig(), AugmentationConfig()

    if args.csv:
        df = clean(pd.read_csv(args.csv, index_col="Date", parse_dates=True))
        df = df[(df.index >= args.start) & (df.index < args.end)]
    else:
        df = load_prices(args.symbol, args.start, args.end, data_cfg.cache_dir)
    train, validation, test = split_by_dates(df, args.validation_start, args.test_start)
    print(f"{args.symbol} : train {len(train)} j, validation {len(validation)} j, test {len(test)} j")

    agent = TDQN(env_cfg, agent_cfg, aug_cfg, device=args.device, seed=args.seed)
    history = agent.train(train, validation, episodes=args.episodes,
                          periods_per_year=args.periods_per_year)

    run_dir = Path(args.out) / f"{args.symbol}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    agent.save(run_dir / "tdqn.pt")

    rows = {}
    for name, part, context in (("validation", validation, train),
                                ("test", test, pd.concat([train, validation]))):
        runs = {
            "TDQN": agent.test(part, context=context),
            "Buy and Hold": baselines.buy_and_hold(part, env_cfg, context),
            "Sell and Hold": baselines.sell_and_hold(part, env_cfg, context),
        }
        for strategy, result in runs.items():
            rows[(name, strategy)] = metrics.performance(result, args.periods_per_year)
            result.to_csv(run_dir / f"{name}_{strategy.replace(' ', '_')}.csv")

    table = pd.DataFrame(rows).round(3)
    print(table.to_string())
    table.to_csv(run_dir / "performance.csv")
    (run_dir / "config.json").write_text(json.dumps({
        "data": asdict(data_cfg), "env": asdict(env_cfg), "agent": asdict(agent_cfg),
        "augmentation": asdict(aug_cfg), "episodes": args.episodes, "seed": args.seed,
        "periods_per_year": args.periods_per_year, "csv": args.csv,
    }, indent=2))
    print(f"Résultats dans {run_dir}")


if __name__ == "__main__":
    main()
