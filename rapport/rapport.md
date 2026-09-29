# TDQN appliqué au Bitcoin : reprise de Théate & Ernst

*Projet de recherche étudiant — [auteurs] — [date]*

## 1. Objectif

Théate et Ernst proposent TDQN, un agent d'apprentissage par renforcement profond
qui décide chaque jour comment se positionner sur un marché pour maximiser le
rapport entre gain et risque (§3.5). Ils l'évaluent sur 30 actions et indices
boursiers entre 2018 et 2019 (§5.1), et citent eux-mêmes le Bitcoin comme terrain
d'application possible (§3.1). Ce projet reprend leur méthode et l'adapte au BTC.

## 2. Contexte : trading et Bitcoin

Une stratégie de **trading algorithmique** décide automatiquement, à intervalles
réguliers (ici chaque jour), quelle position prendre sur un actif. Deux positions
de base existent. Être **long**, c'est acheter l'actif : on gagne s'il monte.
Être **short** (vente à découvert), c'est emprunter l'actif pour le vendre tout de
suite, puis le racheter plus tard pour le rendre : on gagne s'il baisse, mais la
perte n'est pas bornée s'il monte, et l'emprunt a généralement un coût. On peut
aussi rester **en cash**, sans exposition. Chaque achat ou vente paie des **frais
de transaction**, qui pénalisent les stratégies qui changent souvent de position.

Gagner de l'argent ne suffit pas à juger une stratégie : il faut tenir compte du
risque pris. L'indicateur principal du papier est le **ratio de Sharpe** : le
rendement moyen de la stratégie divisé par l'écart-type de ce rendement (sa
volatilité), ramené à l'année (§3.5). Un Sharpe de 1 signifie que le gain annuel
moyen vaut une fois la volatilité annuelle. On regarde aussi le **drawdown
maximal**, la plus forte perte subie entre un sommet et le creux suivant. La
référence naturelle est **buy & hold** : acheter au début et ne plus rien faire.

Le **Bitcoin** diffère des actions étudiées dans le papier sur trois points utiles
ici. Il s'échange en continu, tous les jours de l'année, sans fermeture le
week-end. Il est nettement plus volatil. Enfin, son historique de prix est plus
court, et une unité vaut une somme élevée, ce qui oblige à acheter des fractions.

## 3. La méthode TDQN

L'agent observe les prix et volumes des 30 derniers jours ainsi que sa position
actuelle (§3.4.1). Il choisit chaque jour entre deux actions, long ou short, la
quantité étant déduite du capital disponible et d'une contrainte qui garantit
qu'une position short reste remboursable (§3.4.2). Sa récompense est le rendement
journalier du portefeuille (§3.4.3). L'apprentissage repose sur l'algorithme DQN,
qui estime par un réseau de neurones la valeur de chaque action, avec plusieurs
améliorations connues pour stabiliser l'entraînement (§4.1, §4.3). Une astuce
propre au trading accélère l'exploration : à chaque pas, l'agent apprend aussi de
l'action qu'il n'a pas jouée, puisque son résultat se calcule directement à partir
des prix (§4.2). Pour compenser le peu de données, les prix sont lissés et des
séries artificielles légèrement modifiées sont générées (§4.3).

Sur Apple, TDQN obtient un Sharpe de 1,48 contre 1,24 pour buy & hold (Table 4) ;
sur Tesla, action très volatile, 0,26 contre 0,51 (Table 5). Les auteurs
soulignent une forte variance d'un entraînement à l'autre et un risque de
surapprentissage (§6).

## 4. Adaptations au Bitcoin

Nous avons d'abord adapté le **découpage des données** : l'historique du BTC étant
plus court, l'entraînement porte sur 2017-2021 et le test sur 2023-2024. L'année
2022 sert de **validation** : après chaque passe d'entraînement, l'agent est évalué
sur cette période et l'on conserve la version qui y obtient le meilleur Sharpe. Le
papier mentionne une telle validation (§5.1) sans la décrire ; elle évite de
choisir le modèle en regardant, même indirectement, la période de test.

Comme le Bitcoin s'échange tous les jours, les indicateurs sont **annualisés sur
365 jours** au lieu des 252 jours de bourse utilisés pour les actions (§3.5).

La **position short** pose deux questions sur crypto : son coût et sa
disponibilité. Nous avons donc prévu trois variantes : long ou short comme dans le
papier ; long ou short avec un coût d'emprunt quotidien sur les positions short ;
et long ou cash, sans vente à découvert. Pour la contrainte de remboursement des
shorts, le papier suppose une variation de prix journalière maximale fixe ; nous
l'estimons plutôt à partir de la plus forte hausse observée sur la période
d'entraînement, car la volatilité du BTC dépasse celle des actions.

L'agent peut acheter des **fractions de BTC**, alors que le papier impose un nombre
entier d'actions (§3.4.2), ce qui n'aurait pas de sens quand une unité représente
une grande part du capital. Le **bruit** utilisé pour générer des données
artificielles est désormais proportionnel au prix : dans la version d'origine,
son amplitude relative grandissait avec le niveau de prix et devenait démesurée
au prix du BTC. Enfin, nous avons corrigé une erreur dans le calcul du résultat
de l'action non jouée, qui faussait l'astuce d'exploration dans le cas d'un
passage de long à short.

## 5. Protocole expérimental

Les données sont les prix journaliers du BTC en dollars, disponibles depuis
septembre 2014. Le capital initial est de 100 000 $ et les frais de 0,1 % par
transaction, valeurs par défaut de l'implémentation des auteurs. TDQN est comparé aux stratégies de référence
du papier (§5.2) : buy & hold, vente à découvert permanente, suivi de tendance et
retour à la moyenne. Vu la variance de TDQN, plusieurs entraînements sont
nécessaires pour chaque variante.

## 6. Résultats

*À compléter après les expériences.*

| Stratégie | Sharpe | Rendement annualisé | Drawdown max |
|---|---|---|---|
| Buy & hold | | | |
| TDQN (long/short) | | | |
| TDQN (long/cash) | | | |

## 7. Discussion et limites

*À compléter.* Pistes : sensibilité aux frais et au type de position, instabilité
entre entraînements, changement de comportement du marché entre validation et test.

## Référence

T. Théate, D. Ernst, *An Application of Deep Reinforcement Learning to Algorithmic
Trading*, arXiv:2004.06627v3.
