# Projet RL : TDQN appliqué au trading de cryptomonnaies

Projet de recherche étudiant qui reproduit l'agent TDQN de Théate & Ernst,
*An Application of Deep Reinforcement Learning to Algorithmic Trading*
([arXiv 2004.06627](https://arxiv.org/abs/2004.06627)), et l'adapte au BTC.
Le papier (version markdown) est dans `2004.06627v3_markdown/`.

## Crédits et licence du code de référence

Ce code s'inspire du dépôt des auteurs,
[ThibautTheate/An-Application-of-Deep-Reinforcement-Learning-to-Algorithmic-Trading](https://github.com/ThibautTheate/An-Application-of-Deep-Reinforcement-Learning-to-Algorithmic-Trading)
(commit `370b0fe`). Ce dépôt ne contient **aucune licence**, donc aucun droit
de copie n'est accordé par défaut. Le code ici est une **réécriture** qui suit
le papier et la logique de ce dépôt, sans en copier les fichiers ; chaque
module cite les sections du papier et les fichiers/lignes de référence dont il
s'inspire. Si vous utilisez ce travail, citez le papier :

```
@misc{theate2020application,
  title={An Application of Deep Reinforcement Learning to Algorithmic Trading},
  author={Th{\'e}ate, Thibaut and Ernst, Damien},
  year={2020},
  eprint={2004.06627},
  archivePrefix={arXiv}
}
```

## Organisation

| Fichier | Rôle | Équivalent dans la référence |
|---|---|---|
| `tdqn_crypto/config.py` | Tous les réglages (dataclasses) | variables globales de `tradingSimulator.py:40-59` et `TDQN.py:40-78` |
| `tdqn_crypto/data.py` | Téléchargement Yahoo (`yfinance`), cache CSV, nettoyage, découpage chronologique | `dataDownloader.py`, `tradingEnv.py:102-127` |
| `tdqn_crypto/env.py` | Environnement long/short avec frais et contrainte de solvabilité | `tradingEnv.py` |
| `tdqn_crypto/augmentation.py` | Filtre passe-bas, bruit, décalage, étirement | `dataAugmentation.py` |
| `tdqn_crypto/agent.py` | Réseau, mémoire de rejeu, agent TDQN (Double DQN) | `TDQN.py` |
| `tdqn_crypto/metrics.py` | Sharpe, Sortino, drawdown, etc. | `tradingPerformance.py` |
| `tdqn_crypto/baselines.py` | Buy and Hold, Sell and Hold | `classicalStrategy.py` |
| `scripts/run_tdqn.py` | Entraînement + évaluation en ligne de commande | `main.py`, `tradingSimulator.py` |

## Adaptations au BTC

1. **Validation.** Découpage train 2015-2020 / validation 2021-2022 /
   test 2023-2024 (réglable). Après chaque épisode, l'agent est évalué sur la
   validation et on garde les poids du meilleur Sharpe de validation. Le papier
   mentionne une validation (§5.1) ; le code publié n'en a pas et trace le
   Sharpe du test pendant l'entraînement (`TDQN.py:739-743`).
2. **365 jours par an** pour annualiser Sharpe, Sortino et volatilité (la
   référence utilise 252 séances, `tradingPerformance.py:120,143,168`).
3. **Quantités fractionnaires** (désactivables avec `--integer-shares`). Le
   papier impose un nombre entier d'actions (éq. 14) ; avec un BTC à plusieurs
   dizaines de milliers de dollars, cela immobilise une grosse part du capital.
4. **Bruit relatif.** Le bruit de `dataAugmentation.py:125-134` dépend du
   niveau de prix (≈ `stdev × prix / 10 000`) ; ici l'écart-type est un
   pourcentage fixe. Comme dans la référence, le bruit est désactivé par défaut.
5. **Correction** : dans la référence, la transition « action opposée »
   long→short utilise la mauvaise quantité (`tradingEnv.py:344`). Ici une seule
   fonction calcule les deux transitions.

## Points ouverts

- Contrainte de solvabilité des shorts : elle suppose une variation journalière
  d'au plus ε = 10 % (`tradingEnv.py:153`). Le BTC dépasse ce seuil certains
  jours, et une position short prolongée peut alors mener à une valeur de
  portefeuille négative (observé sur données synthétiques avec Sell and Hold).
- Stratégies à moyennes mobiles (`classicalStrategy.py:350,624`) pas encore portées.
- Le papier évalue sur plusieurs graines (§6.1) ; le script n'en lance qu'une.

## Utilisation

```bash
pip install -r requirements.txt
python -m pytest tests                       # tests sur données synthétiques
python scripts/run_tdqn.py --episodes 5      # essai rapide sur BTC-USD
python scripts/run_tdqn.py                   # 50 épisodes, comme le papier
```

Les données sont mises en cache dans `data/`, les résultats (historique
d'entraînement, modèle, performances validation/test contre Buy and Hold et
Sell and Hold) dans `results/`. `--csv fichier.csv` permet d'utiliser un
fichier local (colonnes `Date,Open,High,Low,Close,Volume`).
