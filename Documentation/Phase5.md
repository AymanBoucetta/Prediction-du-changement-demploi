# Phase 5 : construire le pipeline de préparation

Jusqu'ici, j'avais compris les données (phase 1), exploré (phase 2), décidé quoi faire des cases vides (phase 3) et mis de côté un jeu de test (phase 4). Cette phase transforme toutes ces décisions en **un seul objet** qui prend les données brutes et renvoie des données prêtes pour un modèle. Cette page raconte comment il est construit, pourquoi chaque morceau est là, et ce qu'on peut en retenir.

Au bout du compte : 12 colonnes brutes entrent, 164 colonnes numériques et sans aucune case vide sortent.

## Pourquoi un pipeline

Un modèle de machine learning n'accepte que des nombres, sans case vide. Mais nos données contiennent du texte (`Pvt Ltd`, `>20`), des cases vides et des colonnes d'échelles très différentes. Il faut donc les préparer.

On pourrait le faire à la main, colonne par colonne, directement dans le tableau. Le problème : à chaque nouvelle donnée (le jeu de test, un nouveau candidat), il faudrait refaire les mêmes opérations avec les mêmes valeurs, sans erreur. Un pipeline range tout ça dans un objet qui sait **apprendre une fois**, puis **appliquer partout** de la même façon.

## Le vocabulaire utile

- **Pipeline** : une suite d'étapes appliquées dans l'ordre, rangées dans un seul objet.
- **`fit`** : l'étape apprend quelque chose sur les données (une médiane, une moyenne, la liste des catégories).
- **`transform`** : l'étape applique ce qu'elle a appris, sans rien apprendre de nouveau.
- **`fit_transform`** : les deux d'un coup. On l'utilise sur le jeu d'entraînement.
- **Imputer** (`SimpleImputer`) : remplit les cases vides. Il a trois stratégies utilisées ici : `median` (la médiane apprise), `most_frequent` (la valeur la plus fréquente apprise) et `constant` (une valeur fixe, comme « Inconnu », sans rien apprendre).
- **Encodage one-hot** (`OneHotEncoder`) : transforme une colonne de catégories en plusieurs colonnes de 0 et de 1, une par catégorie.
- **Mise à l'échelle** (`StandardScaler`) : recentre une colonne numérique sur une moyenne de 0 et un écart-type de 1.
- **`FunctionTransformer`** : fait entrer dans le pipeline une fonction qu'on a écrite soi-même (ici, les conversions).
- **`ColumnTransformer`** : envoie chaque groupe de colonnes vers sa propre branche, puis recolle les résultats.
- **`shape`** : la taille d'un tableau, sous la forme (nombre de lignes, nombre de colonnes). `(15326, 164)` veut dire 15 326 candidats et 164 colonnes.
- **`float64`** : le type « nombre décimal » de pandas. Une colonne de nombres qui contient des cases vides est toujours de ce type.

## Le principe : apprendre sur le train, appliquer au test

Chaque étape d'un pipeline a deux temps :

- **`fit`** : l'étape *apprend* quelque chose sur les données (une médiane, une moyenne, la liste des catégories).
- **`transform`** : l'étape *applique* ce qu'elle a appris.

On fait les deux sur le jeu d'entraînement (`fit_transform`), mais **seulement `transform` sur le jeu de test**.

Un exemple concret : le pipeline calcule, sur le train, une médiane de **9 ans** pour `experience` et de **1** pour `last_new_job`. Il remplit les cases vides du train avec ces valeurs, puis remplit les 95 cases vides du test (15 pour `experience`, 80 pour `last_new_job`) avec les **mêmes** 9 et 1. Le test n'a participé à aucun calcul : il a seulement reçu les valeurs.

Si les médianes étaient calculées avec les candidats du test, ce dernier ne serait plus un jeu jamais vu par le modèle. Ce serait une fuite de données, et les scores finaux seraient trop optimistes. Avec `transform` seul, le test reste dans la situation d'un nouveau candidat qui arrive avec une case vide : le modèle la traite avec ce qu'il a appris, sans recalculer.

## Les branches : chaque famille de colonnes a son traitement

| Famille | Colonnes | Traitement | Colonnes en sortie |
|---|---|---|---|
| Numériques à compléter | `experience`, `last_new_job` | conversion, puis médiane | 2 |
| Catégories avec « Inconnu » | `company_size`, `company_type`, `gender`, `education_level` | « Inconnu », puis one-hot | 26 |
| Catégorie « sans objet » | `major_discipline` | « Non applicable », puis one-hot | 7 |
| Catégorie à compléter simplement | `enrolled_university` | valeur la plus fréquente, puis one-hot | 3 |
| Numériques sans case vide | `city_development_index`, `training_hours` | mise à l'échelle | 2 |
| Catégorie sans case vide | `relevent_experience` | one-hot | 2 |
| Catégorie à nombreuses valeurs | `city` | one-hot | 122 |
| **Total** | **12 colonnes** | | **164** |

Les traitements viennent directement du tableau de décisions de la phase 3.

## Étape par étape

### 1. Convertir les nombres cachés dans du texte

```python
def convertir_experience(s):
    return pd.to_numeric(s.replace({"<1": "0", ">20": "21"}))

def convertir_last_new_job(s):
    return pd.to_numeric(s.replace({"never": "0", ">4": "5"}))
```

`replace` remplace les valeurs spéciales par du texte numérique, et `to_numeric` transforme toute la colonne en nombres. Les cases vides restent vides.

Sur le train, `experience` va de 0 à 21 avec 50 cases vides, et `last_new_job` de 0 à 5 avec 343 cases vides. Les deux colonnes sont de type `float64` : **pandas ne peut pas mettre une case vide dans une colonne d'entiers classique**, donc une colonne de nombres qui contient des cases vides est toujours décimale (9.0 et non 9). Les valeurs restent des années entières.

Cette étape est un simple dictionnaire fixe : elle n'apprend rien des données, donc on peut l'appliquer au train et au test sans fuite. Les remplacements (`>20` en 21, `<1` en 0, `>4` en 5, `never` en 0) sont des approximations : on ne connaît pas la vraie valeur.

### 2. Compléter par la médiane

```python
def convertir_numeriques(df_):
    df_ = df_.copy()
    df_["experience"] = convertir_experience(df_["experience"])
    df_["last_new_job"] = convertir_last_new_job(df_["last_new_job"])
    return df_

branche_num = Pipeline([
    ("conversion", FunctionTransformer(convertir_numeriques, feature_names_out="one-to-one")),
    ("imputation", SimpleImputer(strategy="median")),
])
```

La conversion passe avant l'imputation, pour une raison pratique : on ne peut pas calculer une médiane sur du texte. L'option `feature_names_out="one-to-one"` dit à scikit-learn que chaque colonne en entrée garde son nom en sortie : sans ça, on n'aurait pas pu récupérer les noms des colonnes à la fin.

Résultat : `(15326, 2)` pour le train, `(3832, 2)` pour le test, aucune case vide, et des médianes apprises de 9 et 1.

### 3. Les catégories avec « Inconnu »

```python
branche_inconnu = Pipeline([
    ("imputation", SimpleImputer(strategy="constant", fill_value="Inconnu")),
    ("encodage", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
```

`strategy="constant"` remplit toutes les cases vides avec la même valeur, « Inconnu » : rien à apprendre. L'encodage one-hot transforme ensuite chaque catégorie en une colonne.

On n'encode pas chaque catégorie par un nombre, pour une raison simple : si on numérotait `NGO` = 1, `Pvt Ltd` = 2, `Funded Startup` = 3, le modèle croirait que `Pvt Ltd` est « entre » `NGO` et `Funded Startup`, et que l'écart entre `NGO` et `Funded Startup` vaut deux fois celui entre `NGO` et `Pvt Ltd`. Ces types d'entreprise n'ont aucun ordre ni aucune distance. Avec le one-hot, chaque type a sa propre colonne, qui vaut 1 si le candidat est de ce type et 0 sinon :

| Candidat | `company_type_Pvt Ltd` | `company_type_NGO` | `company_type_Inconnu` |
|---|---|---|---|
| A | 1 | 0 | 0 |
| B | 0 | 1 | 0 |

Le nombre de colonnes obtenu : chaque colonne d'origine donne autant de colonnes que de catégories, « Inconnu » compris : `company_size` 9 (8 tranches + « Inconnu »), `company_type` 7 (6 + 1), `gender` 4 (3 + 1) et `education_level` 6 (5 + 1). Ça fait **26**.

`handle_unknown="ignore"` évite une erreur si une catégorie n'existe que dans le test : elle est encodée par des zéros.

### 4. « Non applicable » et valeur la plus fréquente

```python
branche_na = Pipeline([
    ("imputation", SimpleImputer(strategy="constant", fill_value="Non applicable")),
    ("encodage", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

branche_freq = Pipeline([
    ("imputation", SimpleImputer(strategy="most_frequent")),
    ("encodage", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
```

`major_discipline` donne **7 colonnes** (6 valeurs + « Non applicable »). `enrolled_university` donne **3 colonnes** et **aucune catégorie « Inconnu »** : l'imputation par la valeur la plus fréquente ne crée pas de catégorie, elle remplit les cases vides avec `no_enrollment` (valeur apprise sur le train).

### 5. Les colonnes qui n'ont pas de case vide

```python
branche_echelle = Pipeline([("echelle", StandardScaler())])
branche_bin = Pipeline([("encodage", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
branche_ville = Pipeline([("encodage", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
```

**La mise à l'échelle.** `city_development_index` varie entre 0 et 1, alors que `training_hours` monte à plusieurs centaines d'heures. Sans mise à l'échelle, un modèle comme la régression logistique donnerait mécaniquement plus de poids à la colonne aux grands nombres. `StandardScaler` apprend la moyenne et l'écart-type du **train**, puis les applique : chaque colonne se retrouve avec une moyenne proche de 0 et un écart-type de 1 (vérifié sur le train). Un Random Forest n'en a pas besoin, mais on compare plusieurs modèles.

**`relevent_experience`** a deux valeurs : 2 colonnes.

**`city`** donne **122 colonnes** dans le train, alors que le jeu complet a 123 villes : une ville, `city_140`, n'apparaît que dans le test (un seul candidat). C'est pour ça que `handle_unknown="ignore"` compte : ce candidat est encodé avec des zéros sur tout le bloc `city`, et le pipeline ne plante pas (vérifié : la somme du bloc vaut 0,0).

Un point à garder en tête : chaque ville a un seul indice de développement (`nunique().max()` vaut 1). Les colonnes `city` (122 colonnes) et `city_development_index` portent donc en partie la même information. Ce n'est pas un problème de fonctionnement, mais une redondance, que je regarderai à l'interprétation du modèle.

### 6. Assembler les branches

```python
from sklearn.compose import ColumnTransformer

preprocess = ColumnTransformer([
    ("num_mediane", branche_num, ["experience", "last_new_job"]),
    ("cat_inconnu", branche_inconnu, ["company_size", "company_type", "gender", "education_level"]),
    ("cat_na", branche_na, ["major_discipline"]),
    ("cat_freq", branche_freq, ["enrolled_university"]),
    ("echelle", branche_echelle, ["city_development_index", "training_hours"]),
    ("binaire", branche_bin, ["relevent_experience"]),
    ("ville", branche_ville, ["city"]),
])

Xtr_prep = preprocess.fit_transform(X_train)
Xte_prep = preprocess.transform(X_test)
```

Le `ColumnTransformer` envoie chaque groupe de colonnes à sa branche, puis colle les résultats côte à côte. Les 12 colonnes de `X` sont toutes assignées (une colonne oubliée serait abandonnée en silence).

Résultat : `(15326, 164)` et `(3832, 164)`, avec **0 case vide** dans les deux tableaux. Les deux ont le même nombre de colonnes, puisque le test a été transformé avec ce que le train a appris.

## Les vérifications

- **Les données d'origine n'ont pas changé.** `X_train` contient toujours ses 16 499 cases vides, soit environ 80 % des 20 733 du jeu complet. Le pipeline produit de nouveaux tableaux (`Xtr_prep`, `Xte_prep`) sans toucher à l'original.
- **Les médianes viennent du train** : `[9. 1.]`.
- **Les 164 colonnes ont un nom** (`num_mediane__experience`, `cat_inconnu__company_size_10/49`…), ce qui sera utile pour interpréter le modèle.
- **La ville absente du train** (`city_140`) est encodée par des zéros dans le test.

## Les chiffres à retenir

| | |
|---|---|
| Colonnes en entrée / en sortie | 12 / 164 |
| Lignes (train / test) | 15 326 / 3 832 |
| Cases vides après le pipeline | 0 dans les deux jeux |
| Médianes apprises | `experience` 9, `last_new_job` 1 |
| Cases vides complétées dans le test | 95 (15 + 80) pour les colonnes numériques |
| Colonnes issues du one-hot | `company_size` 9, `company_type` 7, `gender` 4, `education_level` 6, `major_discipline` 7, `enrolled_university` 3, `relevent_experience` 2, `city` 122 |
| Ville présente seulement dans le test | `city_140` (1 candidat) |

## Les choix et les limites

- **Le one-hot fait perdre l'ordre de `company_size`.** Les tranches sont des colonnes indépendantes : le modèle ne sait pas que `<10` est plus petit que `50-99`. C'est un choix de simplicité, à rediscuter si les résultats le justifient.
- **`city` crée 122 colonnes,** donc beaucoup de colonnes pour peu d'information, et elle recoupe `city_development_index`.
- **Les conversions des bornes** (`>20` en 21, `<1` en 0, `>4` en 5, `never` en 0) sont des approximations.
- **Les décisions de la phase 3** sur les catégories « Inconnu » et « Non applicable » restent des choix défendus par les données, pas des vérités.
- **Pas de test de la qualité du pipeline** : on a vérifié qu'il fonctionne, pas qu'il améliore les modèles. C'est le rôle de la phase suivante.

## Ce que le pipeline ne fait pas encore

Il prépare les données, mais il ne contient pas encore de modèle. Pour la validation croisée, il faudra l'intégrer dans un `Pipeline` complet avec le modèle : ainsi, les médianes, les échelles et les catégories seront recalculées à chaque pli sur les données d'entraînement de ce pli seulement, sans fuite de données.

## Prochaine étape

La phase 6 : mettre ce pipeline dans un `Pipeline` complet avec le modèle, pour que la validation croisée recalcule les médianes, les échelles et les catégories à chaque pli, sans fuite de données. On commence par un modèle de référence qui répond toujours « 0 », afin de fixer le score minimal à battre.