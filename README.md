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

**Pour faire tourner le code sur le BTC**
- `tradingSimulator.py` : train 2017-2021, validation 2022, test 2023-2024 ;
  ajout d'un dictionnaire `cryptos` (`Bitcoin`, `Ethereum`).
- `TDQN.py` : date de fin `'2020-1-1'` écrite en dur remplacée (`plotExpectedPerformance`, l.921).
- `dataDownloader.py` : téléchargement Yahoo via `yfinance`, car
  `pandas_datareader` ne fonctionne plus avec Yahoo.
- `main.py` : actif par défaut `Bitcoin`.
- `requirements.txt` : versions qui tournent sous Python 3.11. pandas est figé
  en 1.5.3, car le code écrit dans les DataFrames par affectation en chaîne
  (`data['Cash'][t] = ...`), ce que pandas 2 et 3 n'appliquent plus.

**Pour que les résultats aient plus de sens sur le BTC (on s'éloigne du papier)**
1. **Validation** (`tradingSimulator.py`, `TDQN.training`) : TDQN s'entraîne
   jusqu'à `validationDate`, est évalué sur la validation après chaque épisode,
   et garde les poids du meilleur Sharpe de validation. L'original traçait le
   Sharpe du test pendant l'entraînement et gardait le dernier épisode.
   Les stratégies classiques s'entraînent toujours jusqu'à `splitingDate`.
2. **365 jours par an** pour annualiser volatilité, Sharpe et Sortino
   (`tradingDaysPerYear` dans `tradingPerformance.py`, 252 dans l'original).
3. **Sens de l'action 0** (`positionMode` dans `tradingEnv.py`) :
   - `'longShort'` (défaut) : long / short, comme le papier ;
   - `'longShortFunding'` : long / short, avec un coût d'emprunt quotidien sur
     les shorts (`dailyShortCost`, 0,03 % par jour, valeur indicative à calibrer) ;
   - `'longCash'` : long / tout en cash, sans vente à découvert.
4. **Seuil de solvabilité des shorts** (`shortEpsilon` dans `tradingEnv.py`) :
   variation de prix maximale supposée en un pas, fixée à 0,1 dans l'original.
   Par défaut (`'auto'`), c'est la plus forte hausse journalière des données
   d'entraînement (au moins 0,1), reprise telle quelle en validation et en test.
   Sans effet en mode `'longCash'`.
5. **Correction d'un bug** : dans l'original, la transition de l'action
   opposée (astuce d'exploration, §4.2) utilisait, pour le cas long → short, le
   nombre d'actions déjà mis à jour au lieu de l'ancien. Une seule fonction
   (`computeTransition`) calcule désormais les deux transitions.
6. **Quantités fractionnaires** (`fractionalShares` dans `tradingEnv.py`,
   `True` par défaut) au lieu d'un nombre entier d'actions.
7. **Bruit relatif** (`dataAugmentation.noiseAddition`) : l'écart-type est
   `stdev` % du prix, au lieu de `stdev × prix / 10 000` qui explosait au prix
   du BTC. Le bruit reste désactivé par défaut (`noiseRange = [0]`).

Pour changer un réglage, modifier la variable globale en haut du fichier
concerné (comme les autres réglages du code d'origine).

## Utilisation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py                               # TDQN sur Bitcoin, 50 épisodes
python main.py -strategy "Buy and Hold"      # baseline
```
