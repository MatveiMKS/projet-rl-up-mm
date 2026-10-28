# TDQN appliqué au Bitcoin : reprise de Théate & Ernst

*Projet de recherche étudiant — [auteurs] — [date]*

## 1. Objectif et contexte

Théate et Ernst proposent TDQN, un agent d'apprentissage par renforcement profond
qui décide chaque jour comment se positionner sur un marché (§3). Ils l'évaluent
sur 30 actions et indices (§5.1) et citent le Bitcoin comme terrain possible
(§3.1). Nous reprenons leur code et l'appliquons au BTC.

Être **long**, c'est acheter l'actif : on gagne s'il monte. Être **short** (vente
à découvert), c'est emprunter l'actif pour le vendre, puis le racheter plus tard :
on gagne s'il baisse, mais la perte n'est pas bornée. On peut aussi rester **en
cash**. Chaque transaction paie des **frais**. Le critère principal du papier est
le **ratio de Sharpe** : rendement moyen divisé par sa volatilité, ramené à
l'année (§3.5). Le **drawdown maximal** est la plus forte perte entre un sommet et
le creux suivant. La référence est **buy & hold** : acheter au début et ne plus
rien faire.

## 2. Méthode et adaptations

L'agent observe les prix et volumes des 30 derniers jours et sa position (§3.4.1),
choisit chaque jour long ou short (§3.4.2) et reçoit comme récompense le rendement
journalier du portefeuille (§3.4.3). Il apprend avec DQN (§4.1) et, à chaque pas,
apprend aussi de l'action qu'il n'a pas jouée, dont le résultat se calcule à
partir des prix (§4.2), et s'entraîne sur des séries de prix artificiellement
perturbées (§4.3).

Le BTC s'échangeant tous les jours, les indicateurs sont annualisés sur 365 jours
au lieu de 252. L'agent peut acheter des fractions de BTC. La contrainte qui
garantit qu'un short reste remboursable utilise la plus forte hausse journalière
observée plutôt qu'une valeur fixe. Nous avons corrigé une erreur dans le calcul
du résultat de l'action non jouée. Enfin, comme le short est coûteux ou
indisponible sur crypto, nous avons ajouté un mode **long ou cash** où l'action «
short » est remplacée par la sortie du marché. Le papier mentionne une période de
validation (§5.1) sans la décrire : après chaque passe d'entraînement, l'agent est
évalué sur cette période et l'on garde les poids qui y obtiennent le meilleur
Sharpe.

## 3. Protocole

Données : prix journaliers du BTC en dollars, capital 100 000 $, frais de 0,1 %.
Entraînement de 2017 à 2020, validation 2021-2022, test 2023-2024. Ces périodes
diffèrent fortement : sur la validation, buy & hold perd 44 % (+58 % en 2021, −65
% en 2022, Sharpe −0,03) ; sur le test, il gagne 462 % (Sharpe 2,01).

Nous avons réglé TDQN par phases, une famille d'hyperparamètres à la fois, avec 3
graines aléatoires par variante (5 pour les configurations finales) et 50 passes
d'entraînement. Le critère de choix est le Sharpe de validation moyen sur les 10
dernières passes, pour ne jamais choisir en regardant le test. Nous avons ensuite
comparé TDQN à **PPO**, un algorithme qui apprend directement une politique au
lieu d'estimer la valeur de chaque action.

## 4. Résultats

**Réglage.** Aucun réglage n'apporte de gain robuste. Facteur d'actualisation,
dropout, régularisation L2, taux d'apprentissage et bruit d'augmentation font
moins bien que les valeurs par défaut, ou sont trop instables entre graines. Seul
un réseau plus petit (128 neurones par couche au lieu de 512) semblait mieux faire
sur 3 graines (Sharpe de validation 0,18 contre 0,02). Sur 5 graines, l'écart
disparaît : 128 neurones donne −0,04 ± 0,32 en validation et −0,41 ± 0,64 en test,
512 neurones 0,02 ± 0,14 et −0,18 ± 0,50.

**Long ou cash.** Interdire le short change tout (test, 128 neurones, moyenne ±
écart-type sur 5 graines) :

| Stratégie (test 2023-2024) | Sharpe | Rendement annualisé | Drawdown max |
|---|---|---|---|
| Buy & hold | 2,01 | +137 % | 26 % |
| TDQN long/short | −0,41 ± 0,64 | négatif | — |
| TDQN long/cash | 1,20 ± 0,91 | +40 % | 20 % |
| PPO long/cash | 1,34 ± 0,15 | +36 % | 20 % |

TDQN long/cash obtient un Sharpe d'environ 1,3 à 1,8 sur quatre graines, mais une
graine finit à −0,39. PPO (taux d'apprentissage 3·10⁻⁴) atteint un Sharpe
comparable avec une variance bien plus faible : ses 5 graines sont entre 1,12 et
1,50. Aucun agent ne bat buy & hold, mais les deux versions long/cash gagnent de
l'argent avec un drawdown plus faible.

## 5. Discussion et limites

Le premier problème est la **variance entre entraînements**, déjà notée par les
auteurs (§6.1, §6.6). Sur 3 graines, des écarts de Sharpe de 0,2 à 0,4 relèvent du
bruit, et plusieurs « améliorations » apparentes ont disparu à 5 graines. Un
réglage fin demanderait bien plus de graines que notre budget de calcul ne le
permet.

Le second est le **changement de régime** du marché, que les auteurs identifient
comme une limite principale (§6.6). La validation, baissière en 2022, récompense
les agents qui shortent ; le test, très haussier, les pénalise. Le Sharpe de
validation est donc un mauvais prédicteur du test, parfois même anti-corrélé :
TDQN long/cash a un Sharpe de validation négatif (−0,29) mais un résultat positif
en test. Une validation glissante sur plusieurs périodes serait plus sûre.

Le mode long/cash fonctionne parce qu'il borne le pire cas : sur un actif à forte
tendance haussière, l'agent ne peut plus perdre en pariant à la baisse, il peut
seulement manquer une partie de la hausse. Il ne peut donc pas battre buy & hold
dans un marché aussi haussier. Enfin, ces conclusions reposent sur une seule
période de test.

**Référence.** T. Théate, D. Ernst, *An Application of Deep Reinforcement Learning
to Algorithmic Trading*, arXiv:2004.06627v3.
