"""Agent TDQN (papier §4).

Réécriture de `TDQN.py` : réseau (`DQN`, l.178), mémoire de rejeu
(`ReplayMemory`, l.84) et agent (`TDQN`, l.265). Même algorithme : Double DQN,
perte de Huber, clipping du gradient et de la récompense, Adam avec L2,
ε-greedy à décroissance exponentielle et « sticky actions ».

Différence principale : l'entraînement évalue la politique sur un bloc de
validation après chaque épisode et garde les poids du meilleur épisode
(early stopping, papier §4.3). La référence calculait à la place le Sharpe du
test à chaque épisode, pour ses courbes (`TDQN.py:739-743`).
"""

import copy
import math
import random
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import augmentation
from .config import AgentConfig, AugmentationConfig, EnvConfig
from .env import TradingEnv
from .metrics import sharpe_ratio


class DQN(nn.Module):
    """5 couches linéaires, BatchNorm, LeakyReLU, Dropout, init. Xavier."""

    def __init__(self, n_inputs: int, n_outputs: int, neurons: int = 512, dropout: float = 0.2):
        super().__init__()
        sizes = [n_inputs] + [neurons] * 4
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.norms = nn.ModuleList(nn.BatchNorm1d(neurons) for _ in range(4))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(neurons, n_outputs)
        for layer in [*self.hidden, self.out]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        for linear, norm in zip(self.hidden, self.norms):
            x = self.dropout(F.leaky_relu(norm(linear(x))))
        return self.out(x)


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, *experience):
        self.memory.append(experience)

    def sample(self, batch_size: int):
        return map(np.array, zip(*random.sample(self.memory, batch_size)))

    def __len__(self):
        return len(self.memory)


def normalization_coefficients(df):
    """Bornes min-max calculées sur tout un jeu de données (`TDQN.py:383-418`)."""
    close, low, high, volume = (df[c].to_numpy(dtype=float) for c in ("Close", "Low", "High", "Volume"))
    returns = np.abs(np.diff(close) / close[:-1])
    return [
        (0.0, returns.max()),                 # rendements du Close
        (0.0, np.abs(high - low).max()),      # amplitude High-Low
        (0.0, 1.0),                           # position du Close dans [Low, High]
        (volume.min(), volume.max()),         # volume
    ]


def _minmax(x, bounds, constant):
    lo, hi = bounds
    return (x - lo) / (hi - lo) if hi != lo else np.full_like(x, constant)


def process_state(obs, coefficients) -> np.ndarray:
    """Transforme la fenêtre brute en vecteur de taille 4(L-1)+1 (`TDQN.py:421-470`)."""
    close, low, high, volume = obs["close"], obs["low"], obs["high"], obs["volume"]
    returns = np.diff(close) / close[:-1]
    spread = np.abs(high - low)[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        close_pos = np.where(spread != 0, np.abs(close[1:] - low[1:]) / spread, 0.5)
    return np.concatenate([
        _minmax(returns, coefficients[0], 0.0),
        _minmax(spread, coefficients[1], 0.0),
        _minmax(close_pos, coefficients[2], 0.5),
        _minmax(volume[1:], coefficients[3], 0.0),
        [obs["position"]],
    ]).astype(np.float32)


class TDQN:
    def __init__(self, env_config: EnvConfig = None, config: AgentConfig = None,
                 aug_config: AugmentationConfig = None, device: str = "auto", seed: int = 0):
        self.env_config = env_config or EnvConfig()
        self.config = config or AgentConfig()
        self.aug_config = aug_config or AugmentationConfig()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        c, n_obs = self.config, self.env_config.observation_size
        self.policy = DQN(n_obs, 2, c.neurons, c.dropout).to(self.device)
        self.target = DQN(n_obs, 2, c.neurons, c.dropout).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.policy.eval()
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=c.learning_rate,
                                          weight_decay=c.l2_factor)
        self.memory = ReplayMemory(c.capacity)
        self.iterations = 0
        self.coefficients = None

    # ------------------------------------------------------------- actions
    def epsilon(self) -> float:
        c = self.config
        return c.epsilon_end + (c.epsilon_start - c.epsilon_end) * math.exp(-self.iterations / c.epsilon_decay)

    @torch.no_grad()
    def act(self, state: np.ndarray):
        """Action gloutonne et valeurs Q."""
        q = self.policy(torch.as_tensor(state, device=self.device).unsqueeze(0)).squeeze(0)
        return int(q.argmax().item()), q.cpu().numpy()

    def act_epsilon_greedy(self, state: np.ndarray, previous_action: int) -> int:
        """ε-greedy avec sticky actions (`TDQN.py:530-564`)."""
        if random.random() > self.epsilon():
            action = self.act(state)[0] if random.random() > self.config.sticky_alpha else previous_action
        else:
            action = random.randrange(2)
        self.iterations += 1
        return action

    # --------------------------------------------------------- apprentissage
    def learn(self):
        """Une mise à jour Double DQN (`TDQN.py:567-619`)."""
        c = self.config
        if len(self.memory) < c.batch_size:
            return
        self.policy.train()
        state, action, reward, next_state, done = self.memory.sample(c.batch_size)
        tensor = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=self.device)
        state, next_state = tensor(state), tensor(next_state)
        action, reward, done = tensor(action, torch.long), tensor(reward), tensor(done)

        current = self.policy(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Comme la référence : la politique (en mode train) choisit, la cible évalue.
            next_action = self.policy(next_state).argmax(1, keepdim=True)
            next_q = self.target(next_state).gather(1, next_action).squeeze(1)
            expected = reward + c.gamma * next_q * (1 - done)

        loss = F.smooth_l1_loss(current, expected)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), c.gradient_clipping)
        self.optimizer.step()
        if self.iterations % c.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())
        self.policy.eval()

    def _clip(self, reward: float) -> float:
        return float(np.clip(reward, -self.config.reward_clipping, self.config.reward_clipping))

    def train(self, train_df, validation_df=None, episodes: int = 50,
              periods_per_year: int = 365, verbose: bool = True):
        """Entraîne l'agent (`TDQN.py:622-779`). Renvoie l'historique par épisode.

        Si `validation_df` est fourni, les poids retenus sont ceux du meilleur
        Sharpe de validation ; sinon ceux du dernier épisode.
        """
        a = self.aug_config
        train_sets = augmentation.generate(train_df, a.shift_range, a.stretch_range,
                                           a.filter_range, a.noise_range, self.rng)
        envs = [TradingEnv(df, self.env_config) for df in train_sets]
        coefficients = [normalization_coefficients(df) for df in train_sets]
        # Coefficients du test : train filtré, comme `TDQN.py:797-802`.
        self.coefficients = normalization_coefficients(
            augmentation.low_pass_filter(train_df, self.config.filter_order))

        history, best_score, best_weights = [], -np.inf, None
        steps = 0
        for episode in range(episodes):
            total_reward = 0.0
            for env, coeffs in zip(envs, coefficients):
                env.reset()
                # Départ aléatoire (`TDQN.py:679`), borné pour éviter un épisode vide.
                obs = env.set_starting_point(random.randrange(len(env) - 1))
                state = process_state(obs, coeffs)
                previous_action, done = 0, env.done
                while not done:
                    action = self.act_epsilon_greedy(state, previous_action)
                    obs, reward, done, info = env.step(action)
                    reward = self._clip(reward)
                    next_state = process_state(obs, coeffs)
                    self.memory.push(state, action, reward, next_state, float(done))
                    # Astuce d'exploration : on stocke aussi l'action opposée (§4.2).
                    self.memory.push(state, info["other_action"], self._clip(info["other_reward"]),
                                     process_state(info["other_observation"], coeffs),
                                     float(info["other_done"]))
                    steps += 1
                    if steps % self.config.learning_update_period == 0:
                        self.learn()
                    state, previous_action = next_state, action
                    total_reward += reward

            record = {"episode": episode, "total_reward": total_reward,
                      "train_sharpe": sharpe_ratio(self.test(train_df)["Returns"], periods_per_year)}
            if validation_df is not None:
                score = sharpe_ratio(self.test(validation_df, context=train_df)["Returns"], periods_per_year)
                record["validation_sharpe"] = score
                if score > best_score:
                    best_score, best_weights = score, copy.deepcopy(self.policy.state_dict())
            history.append(record)
            if verbose:
                print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in record.items()), flush=True)

        if best_weights is not None:
            self.policy.load_state_dict(best_weights)
        return history

    def test(self, df, context=None):
        """Politique gloutonne, poids figés (`TDQN.py:782-826`).

        L'agent observe la série filtrée (moyenne glissante causale) mais
        exécute ses ordres sur les prix bruts. `context` : données qui précèdent
        `df` (en pratique le train), dont les L derniers jours servent de
        premier état ; sans elles, les L premiers jours de `df` ne sont pas tradés.
        Normalisation avec les coefficients du train, comme la référence.
        """
        if self.coefficients is None:
            raise RuntimeError("Entraîner (ou charger) l'agent avant de le tester.")
        L = self.env_config.state_length
        full = df if context is None else pd.concat([context.iloc[-L:], df])
        smoothed = TradingEnv(augmentation.low_pass_filter(full, self.config.filter_order), self.env_config)
        raw = TradingEnv(full, self.env_config)
        obs, done = smoothed.reset(), False
        raw.reset()
        while not done:
            action, _ = self.act(process_state(obs, self.coefficients))
            obs, _, done, _ = smoothed.step(action)
            raw.step(action)
        result = raw.history()
        return result.iloc[L - 1:] if context is not None else result

    # ---------------------------------------------------------- sauvegarde
    def save(self, path):
        torch.save({"policy": self.policy.state_dict(), "coefficients": self.coefficients}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy"])
        self.target.load_state_dict(checkpoint["policy"])
        self.coefficients = checkpoint["coefficients"]
