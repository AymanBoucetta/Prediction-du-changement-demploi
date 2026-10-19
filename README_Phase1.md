# Phase 1 : comprendre les données avant de les toucher

Avant de nettoyer ou de modéliser, j'ai d'abord cherché à répondre à une question simple : **qu'est-ce que j'ai devant moi ?** Cette page garde la trace de ce que j'ai vu, de ce que j'ai compris, et de ce que j'en retiens pour la suite.

Dans cette phase, je ne modifie rien. Je regarde, je compte, je note.

## Le jeu de données en un coup d'œil

| | |
|---|---|
| Source | Kaggle, « HR Analytics: Job Change of Data Scientists » |
| Fichier | `aug_train.csv`, rangé dans `data/raw/` et jamais modifié |
| Taille | 19 158 candidats (lignes), 14 colonnes |
| Une ligne | un candidat et ses informations (ville, études, expérience, entreprise…) |
| Cible | `target` : 1 si le candidat cherche à changer d'emploi, 0 sinon |
| Problème | classification binaire (deux réponses possibles) |
| Répartition de la cible | 75,1 % de 0 et 24,9 % de 1 |
| Doublons | aucun |
| Identifiant | `enrollee_id` est unique pour chaque ligne |

> « Quand » prédit une date. « Si » prédit un oui ou un non. Ici, on prédit **si** un candidat cherche à changer, donc c'est une classification.

## Le dictionnaire des colonnes

| Colonne | Type dans pandas | Nature | Ce qu'elle représente | Valeurs différentes | Cases vides |
|---|---|---|---|---|---|
| `enrollee_id` | entier | identifiant | numéro unique du candidat | 19 158 | 0 % |
| `city` | texte | catégorielle | code de la ville | 123 | 0 % |
| `city_development_index` | décimal | numérique | indice de développement de la ville, entre 0 et 1 | 93 | 0 % |
| `gender` | texte | catégorielle | genre du candidat | 3 | 23,5 % |
| `relevent_experience` | texte | catégorielle | expérience pertinente pour le poste, oui ou non | 2 | 0 % |
| `enrolled_university` | texte | catégorielle | type de cursus universitaire suivi en ce moment, s'il y en a un | 3 | 2,0 % |
| `education_level` | texte | ordinale | niveau d'études | 5 | 2,4 % |
| `major_discipline` | texte | catégorielle | domaine d'études principal | 6 | 14,7 % |
| `experience` | texte | numérique stockée en texte | années d'expérience totale | 22 | 0,3 % |
| `company_size` | texte | ordinale | taille de l'entreprise actuelle, en tranches | 8 | 31,0 % |
| `company_type` | texte | catégorielle | type de l'entreprise actuelle | 6 | 32,0 % |
| `last_new_job` | texte | numérique stockée en texte | années entre l'avant-dernier emploi et l'emploi actuel | 6 | 2,2 % |
| `training_hours` | entier | numérique | heures de formation suivies | 241 | 0 % |
| `target` | décimal | catégorie (la cible) | 1 si le candidat cherche à changer d'emploi, 0 sinon | 2 | 0 % |

Le nom `relevent_experience` s'écrit comme ça dans le jeu de données, avec la faute. Je le garde tel quel, sinon pandas répond `KeyError`.

**Comment je distingue les types de colonnes :** je me pose trois questions. Quel est son type dans pandas ? Combien de valeurs différentes a-t-elle ? Et surtout : est-ce que la moyenne de cette colonne aurait un sens ? Si oui, c'est numérique. Sinon, c'est une catégorie. Un chiffre n'est pas toujours un nombre : un identifiant ou une cible codée en 0 et 1 restent des étiquettes.

## Ce que j'ai trouvé

### 1. Des cases vides, dans 8 colonnes sur 14

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

Réflexe tentant : supprimer toutes les lignes qui ont une case vide avec `dropna()`. Résultat : on passe de **19 158 à 8 955 lignes**. On perd **53 % des données**. Et les candidats qui ne remplissent pas certains champs ne sont peut-être pas comme les autres. Donc on ne supprime pas : on traitera les cases vides colonne par colonne.

### 2. Des nombres cachés dans du texte

Certaines colonnes qui parlent de nombres sont stockées en texte, à cause de valeurs spéciales :

- `experience` contient `<1` et `>20` à côté de nombres exacts (5, 15…).
- `last_new_job` contient `never` et `>4`.
- `company_size` contient des tranches comme `50-99`.

Pour un modèle, `>20` n'est pas un nombre : c'est du texte. Il faudra convertir. Comme ces valeurs ont un ordre, ce sont des variables ordinales, et je garderai l'ordre en les convertissant.

Piste pour la phase 5, à confirmer : `>20` devient 21, `<1` devient 0, `never` devient 0, `>4` devient 5. Ce sont des approximations (je ne connais pas la vraie valeur), donc je les noterai dans les limites.

### 3. Des classes déséquilibrées

Environ 3 candidats sur 4 ne cherchent pas à changer. Un modèle qui répondrait toujours « 0 » aurait donc environ **75 % d'accuracy sans rien apprendre**, puisqu'il ne trouverait jamais un candidat qui change. L'accuracy seule ne suffira pas : je regarderai la précision, le rappel et le F1 sur la classe 1.

### 4. Une colonne à exclure

`enrollee_id` est un numéro unique par candidat. Il ne décrit pas le profil, donc il n'apporte aucune information utile. Pire : un modèle puissant pourrait s'en servir pour mémoriser les lignes d'entraînement (du surapprentissage). Je le retire avant d'entraîner.

### 5. Pas de doublons

`df.duplicated().sum()` renvoie 0. Rien à supprimer. Une ligne en double aurait pesé plus lourd dans l'apprentissage et, si une copie tombait dans le test, aurait gonflé les résultats.

## Ce que je retiens en cinq lignes

1. 19 158 candidats, 14 colonnes, et une cible oui/non : classification binaire.
2. 75,1 % de 0 pour 24,9 % de 1 : l'accuracy ne suffira pas.
3. 8 colonnes ont des cases vides, jusqu'à 32 % : `dropna()` ferait perdre 53 % des lignes.
4. `experience`, `last_new_job` et `company_size` sont des nombres cachés dans du texte.
5. `enrollee_id` est un numéro, pas une information : je le retire.

## La mise en place

Le projet utilise uv. Les bibliothèques sont dans `pyproject.toml`, installées dans `.venv`.

```bash
uv sync
```

Dans VS Code, le notebook doit utiliser le noyau de `.venv`. Pour vérifier :

```python
import sys
print(sys.executable)   # le chemin doit finir par .venv\Scripts\python.exe
```

Un détail : avec pandas 3, les colonnes de texte apparaissent avec le type `str` dans `df.info()` (au lieu de `object` avec les versions plus anciennes). C'est la même chose.

## Le code utilisé

```python
import pandas as pd

df = pd.read_csv("../data/raw/aug_train.csv")

df.shape                                     # (19158, 14)
df.head()                                    # les premières lignes
df.info()                                    # types et valeurs non vides par colonne
df.describe()                                # statistiques des colonnes numériques

df.isna().sum()                              # cases vides par colonne
(df.isna().mean() * 100).round(1)            # ... en pourcentage
df.dropna().shape                            # (8955, 14) : on perdrait 53 % des lignes

df["target"].value_counts(normalize=True)    # 75 % / 25 %
df.duplicated().sum()                        # 0
df["enrollee_id"].is_unique                  # True
df.nunique()                                 # valeurs différentes par colonne
```

Pour comprendre `(df.isna().mean() * 100).round(1)` : `isna()` donne `True` pour une case vide et `False` sinon. Python compte `True` comme 1 et `False` comme 0, donc la moyenne d'une colonne est la proportion de cases vides.
## Prochaine étape

La phase 2 : regarder comment la cible varie selon les colonnes. Voir `README_phase2.md`.