# Phase 4 : mettre de côté un jeu de test

Cette page explique comment j'ai séparé mes candidats en deux groupes, et surtout **pourquoi je l'ai fait à ce moment-là**. C'est la phase la plus courte du projet, mais c'est celle qui protège toutes les autres : si je me trompe ici, tous mes scores futurs seront trop beaux. Elle est écrite pour qu'on puisse suivre le raisonnement sans avoir fait le projet.

Comme dans la phase 3, **je ne remplis ni ne modifie aucune donnée.**

## L'idée en une image

Imagine un étudiant qui prépare un examen. Si on lui montre les questions de l'examen pendant qu'il révise, même par un détour, sa note finale ne prouve plus rien. Le jeu de test, c'est l'examen : il doit rester **jamais vu** par le modèle jusqu'à la toute fin.

Le découpage sert à ça. Je le fais **avant** de remplir les cases vides, parce que remplir c'est déjà « apprendre » quelque chose des données (une médiane, une valeur fréquente). Si cet apprentissage inclut les futurs candidats du test, le test a déjà influencé le modèle.

## Le vocabulaire utile

- **Jeu d'entraînement (train)** : les candidats sur lesquels le modèle apprend.
- **Jeu de test** : les candidats gardés de côté, utilisés une seule fois à la fin pour juger le modèle.
- **X** : les informations sur chaque candidat, ce que le modèle reçoit.
- **y** : la réponse à prédire, la colonne `target`.
- **Fuite de données** : quand une information du jeu de test influence l'entraînement. Les résultats paraissent alors meilleurs qu'ils ne le sont.
- **Stratifier** : garder la même proportion de chaque classe dans les deux jeux.
- **Reproductible** : quelqu'un d'autre obtient exactement les mêmes jeux, donc les mêmes résultats.

## Ce que j'ai fait, étape par étape

### Étape 1 : séparer ce que le modèle voit de ce qu'il doit deviner

```python
X = df.drop(columns=["target", "enrollee_id"])
y = df["target"].astype(int)
```

Je retire deux colonnes de `X` :

- **`target`**, parce que c'est la réponse. Si elle restait dans `X`, le modèle lirait la réponse au lieu de la deviner, et ses scores ne voudraient rien dire.
- **`enrollee_id`**, parce que c'est un numéro unique qui ne décrit pas le candidat. Un modèle puissant pourrait s'en servir pour mémoriser les lignes d'entraînement.

`drop` renvoie un **nouveau** tableau : `df` reste intact. `astype(int)` transforme `1.0` et `0.0` en vraies classes `1` et `0`.

Résultat : `X` fait (19 158, 12), c'est-à-dire 14 colonnes moins les deux retirées. `y` fait 19 158 lignes, avec 14 381 zéros et 4 777 uns (75,1 % et 24,9 %).

### Étape 2 : découper

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

Trois paramètres comptent :

- **`test_size=0.2`** met 20 % des candidats de côté pour le test.
- **`stratify=y`** garde la même proportion de candidats qui cherchent à changer dans chaque jeu. Sans lui, le hasard pourrait me donner un test avec, disons, 20 % de uns, et la comparaison serait faussée.
- **`random_state=42`** fixe le hasard : le découpage est reproductible. Le nombre 42 n'a rien de magique, n'importe quel nombre fixe fait l'affaire.

Ce que ça donne :

| | Candidats | Colonnes | Moyenne de `target` | Uns | Zéros |
|---|---|---|---|---|---|
| Entraînement | 15 326 | 12 | 0,2494 | 3 822 | 11 504 |
| Test | 3 832 | 12 | 0,2492 | 955 | 2 877 |
| Jeu complet | 19 158 | 14 | 0,2493 | 4 777 | 14 381 |

Une petite surprise sur les effectifs : 80 % de 19 158 donne 15 326,4. Mais un candidat ne se coupe pas en morceaux, donc scikit-learn arrondit le test **vers le haut** (3 831,6 devient 3 832) et laisse le reste à l'entraînement (15 326). Les deux font bien 19 158.

### Étape 3 : vérifier que le découpage est propre

Je ne fais pas confiance au découpage tant que je ne l'ai pas contrôlé.

```python
# 1. Aucun candidat dans les deux jeux à la fois
print(len(set(X_train.index) & set(X_test.index)))      # 0

# 2. Nombre de candidats qui changent dans chaque jeu
print(y_train.sum(), y_test.sum())                      # 3822 955

# 3. Parts de cases vides, train contre test
print(pd.DataFrame({
    "train": X_train.isna().mean().round(3),
    "test": X_test.isna().mean().round(3),
}))
```

**Contrôle 1.** `set(...) & set(...)` donne les candidats présents dans les deux jeux. Il doit être vide, donc 0. Un candidat dans les deux serait une fuite : le modèle aurait vu une ligne du test pendant l'entraînement et l'aurait mémorisée.

**Contrôle 2.** 3 822 + 955 = 4 777 : tous les candidats qui changent sont répartis, avec la même proportion (24,9 %) dans les deux jeux.

**Contrôle 3.** Les parts de cases vides par colonne, qui ne sont pas identiques mais proches :

| Colonne | Train | Test | Jeu complet |
|---|---|---|---|
| `gender` | 23,3 % | 24,4 % | 23,5 % |
| `enrolled_university` | 2,1 % | 1,8 % | 2,0 % |
| `education_level` | 2,4 % | 2,6 % | 2,4 % |
| `major_discipline` | 14,7 % | 14,5 % | 14,7 % |
| `experience` | 0,3 % | 0,4 % | 0,3 % |
| `company_size` | 30,8 % | 31,9 % | 31,0 % |
| `company_type` | 31,8 % | 32,9 % | 32,0 % |
| `last_new_job` | 2,2 % | 2,1 % | 2,2 % |

Les écarts vont de 0 à environ 1 point.

**Pourquoi ces parts ne sont pas identiques, et pourquoi ce n'est pas un problème.** `stratify=y` répartit les candidats selon la cible, pas selon les cases vides. Les cases vides se répartissent donc au hasard entre les deux jeux, et avec 3 832 candidats dans le test, un écart d'environ 1 point est plausible. Si un écart dépassait environ 5 points, il faudrait enquêter. Je n'ai fait aucun test statistique, donc je dis « plausible », pas « prouvé ».

Pour lire la sortie du contrôle 3 : `X_train.isna().mean()` donne la proportion de cases vides de chaque colonne. Un `0.308` veut dire 30,8 % de cases vides.

## Ce qui reste vide, et ce que ça veut dire

Après le découpage, `X_train` et `X_test` contiennent toujours leurs cases vides. On pourrait penser que le jeu de test garde ses cases vides pour toujours, « pour avoir le cas réel ». C'est vrai tant qu'on ne prépare rien, mais c'est incomplet : un modèle comme la régression logistique refuse les `NaN`, donc le test ne peut pas rester vide au moment de l'évaluation.

Ce qui se passe vraiment, dans le pipeline :

1. Le pipeline **apprend** les valeurs de remplissage sur le jeu d'entraînement seulement (une médiane, la valeur la plus fréquente, la catégorie « Inconnu »).
2. Il **applique** ensuite ces mêmes valeurs au jeu de test.

Le test ne choisit aucune valeur : il reçoit celles du train. C'est exactement la situation d'un nouveau candidat qui arrive avec des cases vides : le modèle doit les traiter avec ce qu'il a appris, pas en regardant d'autres candidats.

## Le piège classique : la fuite de données

Un exemple concret de ce projet. Si je calculais la médiane d'`experience` sur les 19 158 candidats avant de découper, cette médiane contiendrait de l'information venant des 3 832 candidats du futur test. Le modèle entraîné ensuite aurait, par ce détour, « entrevu » le test. Ses scores finaux seraient un peu trop optimistes, sans que personne ne puisse le voir.

Même logique pour un doublon : si la même ligne se retrouve dans le train et dans le test, le modèle l'a déjà vue. C'est pour ça que j'ai fait le contrôle 1.

## Pourquoi le test n'est utilisé qu'une seule fois

Si je regarde le score sur le test, que je change quelque chose dans mon modèle parce que le score ne me plaît pas, puis que je regarde à nouveau, le test est devenu une partie de mon entraînement : j'ai choisi le modèle en fonction de lui. Son score n'est plus une mesure honnête.

Pour choisir entre des modèles, j'utiliserai la **validation croisée sur le jeu d'entraînement**. Le test ne servira qu'une fois, à la fin, pour le modèle final.

## Le code complet de la phase

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["target", "enrollee_id"])
y = df["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(X_train.shape, X_test.shape)           # (15326, 12) (3832, 12)
print(y_train.mean(), y_test.mean())          # 0.2494 0.2492
```

## Les pièges à éviter

- **Penser que le jeu de test reste vide.** Il garde ses cases vides jusqu'à l'évaluation, mais le pipeline les remplit avec les valeurs apprises sur l'entraînement.
- **Parler d'un modèle qui « se perd » ou qui est « optimisé à ces lignes ».** Le modèle renvoie une erreur ou prédit mal ; et s'il a déjà vu des lignes du test, il les a mémorisées, ce qui rend le score trop optimiste.
- **Compter un effectif avec une virgule** : 15 326,4 n'existe pas, scikit-learn arrondit.
- **Laisser `target` dans `X`** : le modèle aurait de très bons scores sans rien apprendre.

## Les limites

- **Un seul découpage donne un seul résultat.** Le score sur le test dépend un peu du hasard du découpage. Pour choisir entre les modèles, j'utiliserai la validation croisée sur l'entraînement.
- **Je n'ai pas testé statistiquement** les écarts de cases vides entre train et test.
- **Le test est une estimation, pas une garantie** : les données sont publiques, pas celles d'une vraie entreprise.

## Prochaine étape

La phase 5 : construire le pipeline de préparation (conversions, imputation selon les décisions de la phase 3, encodage), entraîné sur `X_train` seulement.