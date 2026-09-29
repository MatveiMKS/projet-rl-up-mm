# Projet RL : TDQN appliqué au Bitcoin

Exercice de prise en main du code de Théate & Ernst,
*An Application of Deep Reinforcement Learning to Algorithmic Trading*
([arXiv 2004.06627](https://arxiv.org/abs/2004.06627)), appliqué au BTC.
Le papier (version markdown) est dans `2004.06627v3_markdown/`.

## Origine du code

Les fichiers Python viennent tels quels du dépôt des auteurs,
[ThibautTheate/An-Application-of-Deep-Reinforcement-Learning-to-Algorithmic-Trading](https://github.com/ThibautTheate/An-Application-of-Deep-Reinforcement-Learning-to-Algorithmic-Trading)
(commit `370b0fe`). Ce dépôt ne contient pas de licence ; tous les droits
restent à ses auteurs.

## Modifications par rapport à l'original

Toutes sont marquées `ADAPTATION` dans le code.

- `tradingSimulator.py` : dates 2017-01-01 → 2023-01-01 (train) et
  2023-01-01 → 2025-01-01 (test), soit 6 ans / 2 ans comme le papier (§5.1) ;
  ajout d'un dictionnaire `cryptos` (`Bitcoin`, `Ethereum`).
- `TDQN.py` : la date de fin `'2020-1-1'` écrite en dur (l.658 et l.905) est
  remplacée par `'2025-1-1'`.
- `dataDownloader.py` : téléchargement Yahoo via `yfinance`, car
  `pandas_datareader` ne fonctionne plus avec Yahoo.
- `main.py` : actif par défaut `Bitcoin`.
- `requirements.txt` : versions qui tournent sous Python 3.11. pandas est figé
  en 1.5.3, car le code écrit dans les DataFrames par affectation en chaîne
  (`data['Cash'][t] = ...`), ce que pandas 2 et 3 n'appliquent plus.
- Dossiers `Figures/` et `Data/` créés (le code y écrit graphiques et cache CSV).

Non modifié, à garder en tête : les indicateurs annualisent avec 252 jours
(`tradingPerformance.py:120,143,168`) alors que le BTC cote 365 jours par an ;
les quantités sont entières (`math.floor` dans `tradingEnv.py`).

## Utilisation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py                               # TDQN sur Bitcoin, 50 épisodes
python main.py -strategy "Buy and Hold"      # baseline
```
