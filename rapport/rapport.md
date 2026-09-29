# TDQN appliqué au Bitcoin : reprise de Théate & Ernst

*Projet de recherche étudiant — [auteurs] — [date]*

> Contrainte : 2 pages maximum. Les références `fichier:ligne` renvoient au
> code de la PR #1 (branche `claude/tdqn-btc-scaffold-u14at8`, commit `6c62017`),
> les § au papier arXiv 2004.06627v3.

## 1. Objectif

Le papier de Théate & Ernst propose TDQN, un agent d'apprentissage par renforcement
profond qui choisit chaque jour une position (long ou short) pour maximiser le ratio
de Sharpe (§3.5). Il est évalué sur 30 actions et indices, sur 2018-2019 (§5.1), et
cite lui-même le Bitcoin comme terrain possible (§3.1). Nous reprenons le code publié
par les auteurs (commit `370b0fe`) et l'adaptons au BTC, un actif plus volatil, coté
tous les jours et avec un historique plus court.

## 2. La méthode TDQN

- **Observation** : les 30 derniers jours de prix et de volume, plus la position
  courante (§3.4.1, éq. 5 ; `stateLength = 30`, `tradingSimulator.py:47`).
  Le code n'utilise pas le prix d'ouverture : Close, Low, High, Volume.
- **Actions** : deux positions, long ou short, avec une quantité déduite du capital
  et d'une contrainte de solvabilité des shorts (§3.4.2, éq. 13-15).
- **Récompense** : le rendement journalier du portefeuille (§3.4.3).
- **Algorithme** : DQN (§4.1) avec Double DQN, perte de Huber, clipping des
  gradients, Adam, initialisation Xavier et régularisation (Dropout, L2, early
  stopping) (§4.3). Astuce d'exploration : à chaque pas, la transition de l'action
  opposée est aussi stockée en mémoire, puisqu'elle se calcule sans risque (§4.2).
- **Données** : prétraitement par filtre passe-bas et variations journalières, plus
  augmentation de données pour compenser le peu d'historique (§4.3).

Résultats du papier : sur Apple, TDQN obtient un Sharpe de 1,48 contre 1,24 pour
buy & hold (Table 4, §6.1) ; sur Tesla, 0,26 contre 0,51 (Table 5, §6.2). Les auteurs
soulignent une forte variance entre entraînements et un surapprentissage (§6).

## 3. Adaptations au Bitcoin

Toutes sont marquées `ADAPTATION` dans le code.

| Changement | Raison | Où |
|---|---|---|
| Train 2017-2021, validation 2022, test 2023-2024 | Historique BTC plus court ; le papier mentionne une validation (§5.1) absente du code | `tradingSimulator.py:40-45` |
| Garder les poids du meilleur Sharpe de validation | L'original suivait le Sharpe du **test** pendant l'entraînement et gardait le dernier épisode | `TDQN.py:742-765` |
| Annualisation sur 365 jours au lieu de 252 | Le BTC se négocie tous les jours | `tradingPerformance.py:24-27` |
| Trois modes : long/short, long/short avec coût d'emprunt (0,03 %/jour, indicatif), long/cash | Le short sur crypto a un coût, ou n'est pas toujours possible | `tradingEnv.py:40-50` |
| Seuil de solvabilité des shorts calculé sur le train (plus forte hausse journalière, au moins 0,1) | L'original fixe 0,1, trop faible pour le BTC | `tradingEnv.py:52-56`, `tradingSimulator.py:454-455` |
| Quantités fractionnaires | Un BTC peut valoir une grande part du capital (éq. 14 impose des entiers) | `tradingEnv.py:58-60` |
| Bruit d'augmentation relatif au prix | L'original donnait un bruit relatif de `stdev × prix / 10 000`, soit plusieurs centaines de % au prix du BTC | `dataAugmentation.py:127-136` |
| Correction d'un bug | Pour la transition opposée long → short, `tradingEnv.py:344` (original) utilisait `self.numberOfShares`, déjà mis à jour par l'action jouée | `tradingEnv.py:268` (`computeTransition`) |

## 4. Protocole expérimental

Données journalières BTC-USD de Yahoo Finance (via `yfinance`), disponibles depuis
le 17/09/2014. Capital initial 100 000 $, frais 0,1 % par transaction
(`tradingSimulator.py:54-58`). Comparaison avec les stratégies de référence du papier
(§5.2) : buy & hold, sell & hold, suivi de tendance et retour à la moyenne sur
moyennes mobiles. Indicateurs du papier (Table 3) : Sharpe, Sortino, rendement et
volatilité annualisés, drawdown maximal. Vu la variance de TDQN, plusieurs graines
d'entraînement sont nécessaires.

## 5. Résultats

*À compléter après les runs sur données réelles.*

| Stratégie | Sharpe | Rendement annualisé | Drawdown max |
|---|---|---|---|
| Buy & hold | | | |
| TDQN (long/short) | | | |
| TDQN (long/cash) | | | |

## 6. Discussion et limites

*À compléter.* Pistes : sensibilité aux frais et au mode de position, instabilité
entre graines, changement de régime entre la validation et le test.

## Références

- T. Théate, D. Ernst, *An Application of Deep Reinforcement Learning to Algorithmic
  Trading*, arXiv:2004.06627v3.
- Code des auteurs : github.com/ThibautTheate/An-Application-of-Deep-Reinforcement-Learning-to-Algorithmic-Trading
