"""Réglages de l'expérience, regroupés en un seul endroit.

Le code de référence disperse ces valeurs en variables globales
(`tradingSimulator.py:40-59`, `TDQN.py:40-78`). On garde les mêmes valeurs par
défaut, sauf là où le passage au BTC impose une adaptation (signalée par
« ADAPTATION »).
"""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    symbol: str = "BTC-USD"
    # ADAPTATION : le papier (§5.1) utilise 6 ans de train (2012-2017) et 2 ans
    # de test (2018-2019). L'historique Yahoo du BTC-USD commence le 2014-09-17.
    # On ajoute un bloc de validation chronologique entre train et test, que le
    # papier mentionne (§5.1) mais que le code publié n'a pas.
    start: str = "2015-01-01"
    validation_start: str = "2021-01-01"
    test_start: str = "2023-01-01"
    end: str = "2025-01-01"
    cache_dir: str = "data"


@dataclass
class EnvConfig:
    state_length: int = 30               # tradingSimulator.py:45
    transaction_costs: float = 0.001     # 0,1 %, tradingSimulator.py:50-51
    initial_cash: float = 100_000.0      # tradingSimulator.py:54
    # Variation de prix maximale supposée d'un pas à l'autre, utilisée pour la
    # contrainte de solvabilité d'une position short (papier §3.4.2, éq. 13 ;
    # `tradingEnv.py:153`). Le BTC dépasse parfois 10 % en un jour : à revoir.
    epsilon: float = 0.1
    # ADAPTATION : le papier impose un nombre entier d'actions (éq. 14,
    # `math.floor` dans `tradingEnv.py`). Avec un BTC à plusieurs dizaines de
    # milliers de dollars et 100 000 $ de capital, cela laisse une grosse part
    # du capital en cash. On autorise donc les quantités fractionnaires.
    fractional_shares: bool = True

    @property
    def observation_size(self) -> int:
        # 4 séries de (L-1) variations + la position (tradingSimulator.py:46)
        return 1 + (self.state_length - 1) * 4


@dataclass
class AgentConfig:
    # Valeurs de `TDQN.py:40-78`
    gamma: float = 0.4
    learning_rate: float = 1e-4
    target_update: int = 1000
    learning_update_period: int = 1
    capacity: int = 100_000
    batch_size: int = 32
    neurons: int = 512
    dropout: float = 0.2
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 10_000
    sticky_alpha: float = 0.1
    filter_order: int = 5
    gradient_clipping: float = 1.0
    reward_clipping: float = 1.0
    l2_factor: float = 1e-6


@dataclass
class AugmentationConfig:
    # Plages actives par défaut dans `dataAugmentation.py:25-28` :
    # seul le filtre passe-bas d'ordre 5 est utilisé.
    shift_range: list = field(default_factory=lambda: [0.0])
    stretch_range: list = field(default_factory=lambda: [1.0])
    filter_range: list = field(default_factory=lambda: [5])
    # ADAPTATION : ici l'écart-type est un pourcentage relatif du prix
    # (voir augmentation.add_noise), contrairement au code de référence.
    noise_range: list = field(default_factory=lambda: [0.0])


@dataclass
class TrainConfig:
    episodes: int = 50                   # tradingSimulator.py:59
    seed: int = 0
    # ADAPTATION : le BTC cote 365 jours par an, contre 252 séances pour les
    # actions (`tradingPerformance.py:120,143,168`).
    periods_per_year: int = 365
    device: str = "auto"
