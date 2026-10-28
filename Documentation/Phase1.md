# Phase 1 : comprendre les données

Avant de nettoyer, de remplir ou de modéliser quoi que ce soit, je voulais savoir ce que j'avais devant moi. Cette page raconte ce que j'ai regardé, ce que j'ai trouvé et ce que ça change pour la suite du projet. Elle sert aussi de guide : si je la relis dans un mois, je dois retrouver non seulement les chiffres, mais aussi le raisonnement.

Dans cette phase, **rien n'est modifié**. On regarde, on compte, on note.

## Ce que je voulais savoir

Quatre questions, dans cet ordre :

- Qu'est-ce que j'ai devant moi ? La taille, les colonnes, les types.
- Qu'est-ce que je dois prédire, et est-ce équilibré ?
- Qu'est-ce qui est cassé ? Des cases vides, des nombres écrits en texte, des doublons.
- Qu'est-ce que je ne dois pas donner au modèle ?

Sauter cette étape, c'est risquer de supprimer la moitié des données sans s'en rendre compte, ou de présenter un modèle à 75 % d'accuracy qui n'a rien appris.

## Le jeu de données en bref

| | |
|---|---|
| Source | Kaggle, « HR Analytics: Job Change of Data Scientists » |
| Fichier | `aug_train.csv`, dans `data/raw/`, jamais modifié |
| Taille | **19 158** candidats, **14** colonnes |
| Une ligne | un candidat et ses informations (ville, études, expérience, entreprise…) |
| Cible | `target` : 1 si le candidat cherche à changer d'emploi, 0 sinon |
| Répartition de la cible | 14 381 zéros (75,1 %) et 4 777 uns (24,9 %) |
| Doublons | aucun |

Un vocabulaire à fixer d'entrée, parce que je l'ai confondu au début : une **ligne** est un candidat, une **colonne** est une information, une **case** est le croisement des deux. Une ligne est *incomplète* si elle contient au moins une case vide ; elle n'est pas « vide ».

### Ce que contient chaque colonne

| Colonne | Type | Nature | Ce qu'elle représente | Valeurs différentes | Cases vides |
|---|---|---|---|---|---|
| `enrollee_id` | entier | identifiant | numéro unique du candidat | 19 158 | 0 % |
| `city` | texte | catégorielle | code de la ville | 123 | 0 % |
| `city_development_index` | décimal | numérique | indice de développement de la ville, entre 0 et 1 | 93 | 0 % |
| `gender` | texte | catégorielle | genre du candidat | 3 | 23,5 % |
| `relevent_experience` | texte | catégorielle | expérience pertinente pour le poste, oui ou non | 2 | 0 % |
| `enrolled_university` | texte | catégorielle | cursus universitaire suivi en ce moment, s'il y en a un | 3 | 2,0 % |
| `education_level` | texte | ordinale | niveau d'études | 5 | 2,4 % |
| `major_discipline` | texte | catégorielle | domaine d'études principal | 6 | 14,7 % |
| `experience` | texte | numérique stockée en texte | années d'expérience totale | 22 | 0,3 % |
| `company_size` | texte | ordinale | taille de l'entreprise actuelle, en tranches | 8 | 31,0 % |
| `company_type` | texte | catégorielle | type de l'entreprise actuelle | 6 | 32,0 % |
| `last_new_job` | texte | numérique stockée en texte | années entre l'avant-dernier emploi et l'emploi actuel | 6 | 2,2 % |
| `training_hours` | entier | numérique | heures de formation suivies | 241 | 0 % |
| `target` | décimal | cible (une classe) | 1 si le candidat cherche à changer d'emploi, 0 sinon | 2 | 0 % |

`relevent_experience` s'écrit avec cette faute dans le jeu de données. Je garde l'orthographe exacte, sinon pandas répond `KeyError`.

## La nature du problème

La cible `target` ne prend que deux valeurs : 1 ou 0. On ne prédit pas *combien* ni *quand* un candidat va changer d'emploi, mais **si** il cherche à le faire : une réponse parmi deux possibles. C'est une **classification binaire**, pas une régression. La régression, c'est prédire un nombre quelconque, comme un prix ou une température.

La cible est aussi **déséquilibrée** : environ trois candidats sur quatre ne cherchent pas à changer. Cette particularité pèse sur la façon dont on jugera les modèles, j'y reviens plus bas.

## Comment j'ai regardé les données

### Charger et regarder

```python
import pandas as pd

df = pd.read_csv("../data/raw/aug_train.csv")
df.shape        # (19158, 14)
df.head()       # les premières lignes
df.info()       # types et cases non vides par colonne
df.describe()   # statistiques des colonnes numériques
```

`shape` donne le nombre de lignes et de colonnes. `head()` montre déjà des cases vides (`NaN`) et des valeurs comme `>20` et `<1` dans `experience`. `info()` indique, pour chaque colonne, le nombre de cases non vides : si ce nombre est inférieur à 19 158, la colonne a des cases vides (13 018 pour `company_type`, donc 19 158 − 13 018 = 6 140 cases vides). Avec pandas 3, le texte s'affiche `str` à la place de `object` : c'est la même chose. Le `..` dans le chemin remonte d'un dossier, puisque le notebook est dans `notebooks/` et les données dans `data/raw/`.

### Les cases vides

```python
df.isna().sum()                       # nombre de cases vides par colonne
(df.isna().mean() * 100).round(1)     # pourcentage par colonne
```

Cette deuxième ligne se lit de l'intérieur vers l'extérieur. `df.isna()` renvoie un tableau de la même taille que `df`, avec `True` pour une case vide et `False` sinon. `.mean()` calcule la moyenne de chaque colonne : Python compte `True` comme 1 et `False` comme 0, donc la moyenne est la **proportion de cases vides** (0,32 pour `company_type`). `* 100` en fait un pourcentage, `.round(1)` garde une décimale.

### La cible, les doublons et l'identifiant

```python
df["target"].value_counts()                  # 14381 zéros, 4777 uns
df["target"].value_counts(normalize=True)    # 0.751 et 0.249
df.duplicated().sum()                        # 0
df["enrollee_id"].is_unique                  # True
df.nunique()                                 # valeurs différentes par colonne
```

## Ce que j'ai trouvé

### Des cases vides dans 8 colonnes sur 14

| Colonne | Cases vides | Part |
|---|---|---|
| `company_type` | 6 140 | 32,0 % |
| `company_size` | 5 938 | 31,0 % |
| `gender` | 4 508 | 23,5 % |
| `major_discipline` | 2 813 | 14,7 % |
| `education_level` | 460 | 2,4 % |
| `last_new_job` | 423 | 2,2 % |
| `enrolled_university` | 386 | 2,0 % |
| `experience` | 65 | 0,3 % |

Les six autres colonnes sont complètes.

### Pourquoi je ne supprime pas les lignes incomplètes

C'est le réflexe tentant : `dropna()` supprime toutes les lignes qui ont au moins une case vide. Ici, ça fait passer le jeu de **19 158 à 8 955 lignes**. On perd 10 203 lignes, soit **53,3 % des données** (environ 53 %). Deux raisons de ne pas le faire.

La première est la perte : il ne resterait que 46,7 % des candidats pour s'entraîner, et il faudrait encore les partager entre entraînement et test.

La deuxième est le biais. Les candidats qui laissent des champs vides ne sont peut-être pas comme les autres. Un modèle entraîné uniquement sur des profils complets rencontrerait, dans un cas réel, des candidats incomplets qu'il n'a jamais vus, et prédirait mal. Je traiterai donc les cases vides colonne par colonne (phase 3).

### Des nombres cachés dans du texte

`experience` et `last_new_job` sont stockées en texte, parce qu'elles contiennent des valeurs qui ne sont pas des nombres : `<1` et `>20` pour l'une, `never` et `>4` pour l'autre. `company_size` contient des tranches comme `50-99` (et `10/49`, avec une barre oblique, ce qui est incohérent avec les autres).

Pour un modèle, `>20` est du texte : il ne peut pas calculer dessus. Il faut convertir. Comme ces valeurs ont un ordre, je les convertis en gardant l'ordre : `<1` en 0, `>20` en 21, `never` en 0, `>4` en 5, et les autres valeurs gardent leur nombre.

Ces remplacements sont des **approximations**. Pour `>20`, je ne connais pas la vraie valeur (21 ans ? 35 ?), et `<1` regroupe de quelques jours à presque un an. Je prends une valeur raisonnable qui respecte l'ordre, et je le note dans les limites. La conversion se fera plus tard, dans le pipeline.

### Deux colonnes à ne pas donner au modèle

- **`enrollee_id`** est un numéro unique par candidat. Il ne décrit pas son profil et n'apporte aucune information utile. Pire : un modèle puissant pourrait s'en servir pour mémoriser les lignes d'entraînement, ce qui donne du surapprentissage (de bons scores à l'entraînement, de mauvais sur de nouveaux candidats).
- **`target`** est la réponse à prédire. Si elle restait dans les variables d'entrée, le modèle lirait la réponse au lieu de la deviner. Ce n'est pas lié à son déséquilibre : la retirer relève du bon sens.

### Pas de doublons

`df.duplicated().sum()` renvoie 0. Vérifier est utile : une ligne en double pèse plus lourd dans l'apprentissage, donc le modèle la favorise. Et si une copie tombait dans le jeu d'entraînement et l'autre dans le jeu de test, le modèle aurait déjà vu la ligne, ce qui gonflerait les résultats (c'est une forme de fuite de données).

## Les types de variables

| Type | Définition | Test | Exemples ici |
|---|---|---|---|
| **Numérique** | de vrais nombres, avec des écarts égaux | une moyenne a du sens | `training_hours`, `city_development_index`, `experience` après conversion |
| **Catégorielle** | des étiquettes sans ordre | une moyenne n'a pas de sens | `gender`, `city`, `company_type` |
| **Ordinale** | des catégories avec un ordre, mais des écarts non mesurables | on peut classer, pas calculer | `education_level`, `company_size` |
| **Identifiant** | un numéro qui désigne un candidat sans le décrire | n'apporte aucune information | `enrollee_id` |

Pour décider, je pose trois questions à chaque colonne. Quel est son type dans pandas ? Combien de valeurs différentes a-t-elle (`nunique`) ? Et surtout : **est-ce que la moyenne de la colonne entière aurait un sens ?** On calcule la moyenne sur les 19 158 candidats, pas sur un seul. « Expérience moyenne : X ans » a un sens. « La moyenne de Graduate et de Masters » n'en a pas.

Un chiffre n'est pas toujours un nombre : `enrollee_id` est fait de chiffres, mais c'est un identifiant. `target` est stockée en décimal (`1.0` et `0.0`), mais ne contient que deux classes : c'est une catégorie codée en chiffres. `city`, avec 123 valeurs, est une catégorie, et un encodage one-hot créerait 123 colonnes (à gérer en phase 5).

Pourquoi `experience` devient numérique après conversion : une fois `<1` et `>20` remplacés par 0 et 21, les valeurs sont des années avec des écarts égaux (de 5 à 6 ans, c'est un an, comme de 15 à 16). Avant la conversion, elle ressemble à une ordinale à cause du texte ; après, c'est une variable numérique, avec une approximation aux deux extrémités.

## Pourquoi l'accuracy ne suffit pas

L'accuracy est la part de bonnes réponses : bonnes réponses ÷ total. Un modèle qui répondrait **toujours « 0 »** a raison pour tous les candidats qui ne changent pas, soit 14 381 sur 19 158, donc **75,1 %**. Il n'a rien appris, et il ne trouve jamais un seul candidat qui cherche à changer. Pourtant, son accuracy a l'air correcte.

Sa matrice de confusion le montre :

| | La réalité : change | La réalité : ne change pas |
|---|---|---|
| **Le modèle dit : change** | 0 | 0 |
| **Le modèle dit : ne change pas** | 4 777 | 14 381 |

Il faut donc regarder ce qui se passe sur la classe qui nous intéresse (les candidats qui changent), avec trois mesures. Un « positif » est un candidat qui change. Les « bien trouvés » sont ceux que le modèle a prédits « change » **et** qui changent vraiment.

| Mesure | Ce qu'elle dit | Formule |
|---|---|---|
| **Précision** | quand le modèle dit « change », a-t-il raison ? | bien trouvés ÷ ce que le modèle a **prédit** |
| **Rappel** | parmi ceux qui changent vraiment, combien en trouve-t-il ? | bien trouvés ÷ **tous les vrais cas** |
| **F1** | une note qui résume les deux (basse si l'une est basse) | 2 × précision × rappel ÷ (précision + rappel) |

Le numérateur est toujours le même (les bien trouvés) ; seul le dénominateur change. Le modèle « toujours 0 » a un rappel de 0 ÷ 4 777 = **0 %**, et sa précision n'est pas calculable, puisqu'il ne dit jamais « change ».

### Un exemple pour fixer les idées

Sur 100 candidats, 25 cherchent vraiment à changer. Un modèle prédit « change » pour 20 d'entre eux.

| | La réalité : change | La réalité : ne change pas |
|---|---|---|
| **Le modèle dit : change** | 15 (bien trouvés) | 5 (fausses alertes) |
| **Le modèle dit : ne change pas** | 10 (ratés) | 70 |

- Précision : 15 ÷ 20 = **75 %**.
- Rappel : 15 ÷ 25 = **60 %**.
- F1 : 2 × 0,75 × 0,60 ÷ (0,75 + 0,60) = 0,90 ÷ 1,35 ≈ **66,7 %**.
- Accuracy : (15 + 70) ÷ 100 = **85 %**, qui a l'air excellente alors que 10 candidats sur 25 sont ratés.

### Si quelqu'un annonce 78 % d'accuracy

C'est à peine mieux que les 75,1 % d'un modèle qui répondrait toujours 0 : l'écart est de 2,9 points. Je demanderais la précision et le rappel sur la classe qui change, sa matrice de confusion, et s'il a évalué sur un jeu de test séparé.

## Les chiffres à retenir

| | |
|---|---|
| Lignes et colonnes | 19 158 et 14 |
| Cible | 75,1 % de 0 et 24,9 % de 1 (14 381 et 4 777) |
| Doublons | 0 |
| Après `dropna()` | 8 955 lignes, soit 53,3 % de perte |
| Colonnes avec cases vides | 8 sur 14 |
| Les plus vides | `company_type` 6 140 (32,0 %), `company_size` 5 938 (31,0 %), `gender` 4 508 (23,5 %) |
| Accuracy du modèle « toujours 0 » | 75,1 % |

## La mise en place

Le projet utilise uv. Les bibliothèques sont dans `pyproject.toml`, installées dans `.venv`.

```bash
uv sync
```

Dans VS Code, le noyau du notebook doit être celui de `.venv`. Pour le vérifier :

```python
import sys
print(sys.executable)   # doit finir par .venv\Scripts\python.exe
```

## Les limites

- Les conversions de `experience` et `last_new_job` sont des approximations aux bornes.
- Les descriptions des colonnes sont à recouper avec la page Kaggle du jeu de données.
- Les données sont publiques : ce ne sont pas celles d'une vraie entreprise.

## Prochaine étape

La phase 2 : regarder comment la cible varie selon les colonnes. Voir `Phase2.md`.