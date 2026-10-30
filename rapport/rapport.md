# Apprentissage par renforcement profond pour trader le Bitcoin

*Projet de recherche étudiant — [auteurs] — [date]*

## 1. Objectif et contexte

Théate et Ernst [1] proposent TDQN, un agent d'apprentissage par renforcement profond qui
décide chaque jour comment se positionner sur un marché [1, §3]. Ils l'évaluent sur 30
actions et indices [1, §5.1] et citent le Bitcoin comme terrain possible [1, §3.1]. Nous
reprenons leur code [3], l'appliquons au Bitcoin (BTC) et le comparons à un second
algorithme, PPO [2].

Trois positions sont possibles sur un actif. Être **long**, c'est l'acheter et le
garder : on gagne s'il monte. Être **short** (vente à découvert), c'est emprunter
l'actif pour le vendre, puis le racheter plus tard pour le rendre : on gagne s'il
baisse, mais la perte n'est pas bornée s'il monte. Être **en cash**, c'est ne rien
détenir : on ne gagne ni ne perd quand le prix bouge. Chaque achat ou vente paie des
**frais** de 0,1 %.

Le critère qui guide tout le projet est le **ratio de Sharpe** : le rendement moyen
divisé par sa volatilité, ramené à l'année [1, §3.5]. Il mesure le gain obtenu par unité
de risque pris, ce qui le rend plus juste que le gain brut : une stratégie qui gagne
autant en étant moins secouée a un meilleur Sharpe. C'est lui qui sert à choisir les
modèles et à comparer les résultats. On regarde aussi le **drawdown maximal**, la plus
forte perte subie entre un sommet et le creux suivant.

La référence est **buy & hold** : acheter au début et ne plus rien faire. Sur le
Bitcoin, dont le prix a été multiplié de nombreuses fois en dix ans, c'est un adversaire
redoutable : sur une longue période, la détention patiente bat souvent un agent qui
entre et sort du marché, car chaque sortie risque de manquer une partie de la hausse.

## 2. Méthode

L'agent observe les prix et volumes des 30 derniers jours et sa position [1, §3.4.1]. Il
reçoit comme récompense le rendement journalier de son portefeuille [1, §3.4.3] et
apprend avec DQN [1, §4.1]. Il apprend aussi, à chaque pas, de l'action qu'il n'a pas
jouée, dont le résultat se calcule directement à partir des prix [1, §4.2].

**Données augmentées.** Le BTC n'offre qu'une série de prix journaliers, soit environ
1 500 jours pour l'entraînement : un réseau de neurones risque de l'apprendre par cœur
(surapprentissage). Comme dans le papier [1, §4.3], chaque passe d'entraînement utilise
donc deux versions de la série : une version lissée, qui retire les variations les
plus brèves, et une copie de celle-ci où chaque prix et volume journalier est
multiplié par un bruit aléatoire d'environ 0,5 %. L'agent voit ainsi des trajectoires
proches mais jamais identiques, et retient des tendances plutôt que des jours précis.

**Deux façons de se positionner.** Chaque jour, l'agent choisit entre deux actions
seulement. La première est toujours « être long » : il investit tout son cash en BTC,
ou le garde s'il en a déjà. La seconde dépend du mode testé.

- *Mode long/short*, celui du papier [1, §3.4.2] : la seconde action est « être short ».
  L'agent parie à la baisse en vendant du BTC emprunté.
- *Mode long/cash*, notre ajout : la seconde action est « être en cash ». L'agent
  vend tout son BTC et attend. Il ne fait **jamais** de short : il achète, garde,
  revend, puis peut racheter plus tard.

Nous avons mené les deux expériences. Le mode long/cash correspond avant tout à notre
propre manière d'investir : nous achetons, conservons et revendons, sans jamais parier à
la baisse. Il colle aussi au marché crypto, où la vente à découvert est coûteuse.

**Adaptations au BTC.** Le BTC s'échangeant tous les jours, les indicateurs sont
annualisés sur 365 jours au lieu de 252. L'agent peut acheter des fractions de BTC,
alors que le papier impose un nombre entier d'actions. En mode long/short, la
contrainte qui garantit qu'un short reste remboursable utilise la plus forte hausse
journalière observée, car le BTC dépasse souvent la valeur fixe prévue pour les
actions. Nous avons aussi corrigé une erreur du code d'origine dans le calcul de
l'action non jouée.

**PPO.** Là où TDQN estime la valeur de chaque action pour jouer la meilleure, PPO
apprend directement la probabilité de jouer chaque action, et limite à chaque mise à
jour l'écart avec la politique précédente, ce qui stabilise l'apprentissage. Il
reprend le réseau, les données et l'évaluation de TDQN.

## 3. Protocole

Les données sont les prix journaliers du BTC en dollars, avec un capital de 100 000 $.
L'entraînement couvre 2017 à 2020, la validation 2021-2022 et le test 2023-2024. Ces
périodes diffèrent fortement : sur la validation, buy & hold perd 44 % (Sharpe
−0,03) ; sur le test, il gagne 462 % (Sharpe 2,01).

Le papier mentionne une validation [1, §5.1] sans la décrire. Nous l'utilisons ainsi :
après chaque passe d'entraînement, l'agent est évalué sur la période de validation,
et l'on garde la version qui y obtient le **meilleur Sharpe**. Les hyperparamètres
sont réglés de la même façon, par phases, une famille à la fois, en retenant la
variante au meilleur Sharpe de validation moyen. Le test n'est regardé qu'à la fin.
Comme les résultats varient d'un entraînement à l'autre, chaque variante est
entraînée avec 3 graines aléatoires, et 5 pour les configurations finales.

## 4. Résultats

**Réglage de TDQN (mode long/short).** Nous avons fait varier le facteur
d'actualisation, le dropout, la régularisation, la taille du réseau, le taux
d'apprentissage et le bruit d'augmentation. Aucune variante ne bat durablement les
réglages d'origine. La graine aléatoire fixe l'initialisation du réseau et le hasard
de l'exploration : changer de graine donne un autre agent. Sur 3 graines, un réseau de 128 neurones semblait meilleur que celui
d'origine (512 neurones), avec un Sharpe de validation de 0,18 contre 0,02. Refait sur
5 graines, l'avantage disparaît (−0,04 contre 0,02) : il venait du hasard. En mode
long/short, TDQN perd de l'argent sur le test, car la validation (2022, année de krach)
retient les versions qui parient à la baisse, juste avant deux années de forte hausse.

**Passage en long/cash.** Supprimer le short change tout. Le tableau donne, pour
chaque stratégie, ses meilleurs paramètres et ses résultats sur le test 2023-2024
(moyenne ± écart-type sur 5 graines) :

| Stratégie | Meilleurs paramètres | Sharpe | Rendement annualisé | Drawdown max |
|---|---|---|---|---|
| Buy & hold | — | 2,01 | +137 % | 26 % |
| TDQN long/short | réglages d'origine (512 neurones) | −0,18 ± 0,50 | −13 % | 62 % |
| TDQN long/cash | 128 neurones | 1,20 ± 0,91 | +40 % | 20 % |
| PPO long/cash | 128 neurones, taux 3·10⁻⁴ | 1,34 ± 0,15 | +36 % | 20 % |

TDQN long/cash atteint un Sharpe de 1,3 à 1,8 sur quatre graines, mais une graine
finit à −0,39. PPO obtient un Sharpe comparable avec une régularité bien supérieure :
ses 5 graines sont toutes entre 1,12 et 1,50. Sans surprise vu la hausse du BTC sur
cette période, buy & hold reste devant ; mais les deux agents long/cash gagnent de
l'argent avec un drawdown plus faible, et PPO le fait de façon régulière en n'étant
investi que 44 % du temps.

## 5. Conclusion

Nous avons appliqué au Bitcoin l'agent TDQN de Théate et Ernst avec un minimum de
changements. Tel quel, en long/short, il ne fonctionne pas sur cet actif : la
validation sur le krach de 2022 favorise les paris à la baisse, qui coûtent cher
pendant la reprise. En
limitant l'agent à un choix entre être investi et rester en cash, nous obtenons une
stratégie rentable, avec un Sharpe de 1,2 à 1,3 et un risque plus faible que la
simple détention. PPO donne le résultat le plus fiable, avec un Sharpe stable d'un
entraînement à l'autre. Comme dans le papier, où TDQN ne devance buy & hold que de
peu en moyenne [1, Table 6], l'agent ne bat pas la détention passive sur un marché très
haussier, mais il capte une bonne part de la hausse en s'exposant beaucoup moins.

## Références

[1] T. Théate and D. Ernst, "An application of deep reinforcement learning to algorithmic
trading," *Expert Systems with Applications*, vol. 173, Art. no. 114632, 2021, doi:
10.1016/j.eswa.2021.114632 (arXiv:2004.06627).

[2] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy
optimization algorithms," arXiv:1707.06347, 2017.

[3] T. Théate, code source de [1], GitHub, 2020. [En ligne] : https://github.com/ThibautTheate/An-Application-of-Deep-Reinforcement-Learning-to-Algorithmic-Trading
