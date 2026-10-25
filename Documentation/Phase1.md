# Phase 1 — Comprendre les données

## Pourquoi commencer par cette phase ?

Avant de nettoyer les données ou de créer un modèle, je dois d’abord comprendre le jeu de données. Cette première étape me permet de vérifier sa taille, de repérer les colonnes importantes et de voir où se trouvent les problèmes.

Je cherche surtout à répondre à quatre questions :

- Quelles informations contient le fichier ?
- Quelle variable le modèle devra-t-il prédire ?
- Y a-t-il des valeurs manquantes, des doublons ou des valeurs mal formatées ?
- Quelles colonnes ne doivent pas être utilisées comme variables d’entrée ?

Cette vérification est importante. Si je la néglige, je peux supprimer trop de données ou obtenir de bons résultats avec un modèle qui n’a pas réellement appris à reconnaître les bons profils.

**Pendant cette phase, je ne modifie aucune donnée.** Je commence par observer, compter et noter les problèmes. Les corrections viendront plus tard.

## 1. Les notions à connaître

| Terme | Explication | Exemple dans le projet |
|---|---|---|
| Ligne | Informations concernant un candidat | 19 158 lignes |
| Colonne | Type d’information associé à chaque candidat | `city`, `gender`… |
| Case | Valeur située à l’intersection d’une ligne et d’une colonne | Niveau d’études d’un candidat |
| Valeur manquante | Information qui n’est pas renseignée (`NaN` dans pandas) | `company_type` est manquante pour 6 140 candidats |
| Variable cible | Ce que le modèle doit prédire | `target` |
| Variables d’entrée (`X`) | Informations fournies au modèle pour effectuer sa prédiction | Les colonnes utiles, sauf la cible et l’identifiant |
| Classification binaire | Prédiction entre deux catégories | Le candidat cherche à changer d’emploi ou non |
| Régression | Prédiction d’une valeur numérique | Un prix ou une température |
| Classes déséquilibrées | Une catégorie apparaît beaucoup plus souvent que l’autre | 75,1 % de `0` contre 24,9 % de `1` |
| Accuracy | Part des prédictions correctes | 90 prédictions correctes sur 100 donnent 90 % |
| Doublon | Ligne présente plusieurs fois à l’identique | Aucun doublon trouvé ici |
| Identifiant | Numéro qui distingue un candidat sans décrire son profil | `enrollee_id` |
| Surapprentissage | Situation où le modèle mémorise les données d’entraînement au lieu d’apprendre des règles générales | Bons résultats à l’entraînement, mais moins bons sur de nouveaux candidats |
| Imputation | Remplacement d’une valeur manquante par une valeur choisie | Prévue dans la phase 3 |

Il faut bien distinguer une ligne, une colonne et une case. Une ligne représente un candidat, tandis qu’une colonne décrit une information. Une ligne est dite incomplète dès qu’au moins une de ses cases est vide. Cela ne signifie pas que toute la ligne est vide.

## 2. Présentation du jeu de données

Le projet utilise le jeu de données **HR Analytics: Job Change of Data Scientists**, disponible sur Kaggle. Le fichier utilisé est `aug_train.csv`, placé dans `data/raw/`. Je conserve ce fichier d’origine sans le modifier.

| Élément | Résultat |
|---|---|
| Source | Kaggle — « HR Analytics: Job Change of Data Scientists » |
| Fichier | `aug_train.csv` |
| Nombre de candidats | 19 158 |
| Nombre de colonnes | 14 |
| Variable cible | `target` |
| Signification de `target` | `1` si le candidat cherche à changer d’emploi, `0` sinon |
| Répartition de la cible | 14 381 valeurs `0` (75,1 %) et 4 777 valeurs `1` (24,9 %) |
| Doublons | 0 |
| Identifiant | `enrollee_id`, unique pour chaque ligne |

### Description des colonnes

| Colonne | Type pandas | Nature | Signification | Valeurs différentes | Valeurs manquantes |
|---|---|---|---|---:|---:|
| `enrollee_id` | entier | Identifiant | Numéro unique du candidat | 19 158 | 0 % |
| `city` | texte | Catégorielle | Code de la ville | 123 | 0 % |
| `city_development_index` | décimal | Numérique | Indice de développement de la ville, entre 0 et 1 | 93 | 0 % |
| `gender` | texte | Catégorielle | Genre du candidat | 3 | 23,5 % |
| `relevent_experience` | texte | Catégorielle | Présence ou absence d’expérience pertinente | 2 | 0 % |
| `enrolled_university` | texte | Catégorielle | Situation du candidat concernant les études universitaires | 3 | 2,0 % |
| `education_level` | texte | Ordinale | Niveau d’études | 5 | 2,4 % |
| `major_discipline` | texte | Catégorielle | Domaine principal des études | 6 | 14,7 % |
| `experience` | texte | Numérique stockée en texte | Nombre d’années d’expérience | 22 | 0,3 % |
| `company_size` | texte | Ordinale | Taille de l’entreprise actuelle, par tranche | 8 | 31,0 % |
| `company_type` | texte | Catégorielle | Type de l’entreprise actuelle | 6 | 32,0 % |
| `last_new_job` | texte | Numérique stockée en texte | Temps écoulé entre deux emplois, selon le codage du jeu | 6 | 2,2 % |
| `training_hours` | entier | Numérique | Nombre d’heures de formation | 241 | 0 % |
| `target` | décimal | Cible catégorielle | `1` si le candidat cherche à changer d’emploi, `0` sinon | 2 | 0 % |

Le nom `relevent_experience` contient une faute dans le fichier d’origine. Je garde donc ce nom exact dans le code. Si je le remplace par une orthographe différente, pandas ne retrouvera pas la colonne et pourra renvoyer une erreur `KeyError`.

## 3. Exploration des données avec pandas

### 3.1 Charger le fichier

```python
import pandas as pd

df = pd.read_csv("../data/raw/aug_train.csv")
df.shape
df.head()
```

`shape` indique le nombre de lignes et de colonnes. Ici, le résultat attendu est `(19158, 14)`. Avec `head()`, je peux examiner les cinq premières lignes et repérer rapidement certaines particularités, notamment les valeurs manquantes et les valeurs comme `<1` ou `>20` dans `experience`.

Le chemin commence par `..` parce que le notebook se trouve dans `notebooks/`, alors que le fichier de données est rangé dans `data/raw/`.

### 3.2 Vérifier les types et les valeurs renseignées

```python
df.info()
```

Cette commande affiche le nom de chaque colonne, son type et le nombre de valeurs non manquantes. Si une colonne contient moins de 19 158 valeurs renseignées, cela signifie qu’il lui manque des informations.

Par exemple, `company_type` compte 13 018 valeurs renseignées. Le calcul `19 158 - 13 018` donne donc 6 140 valeurs manquantes. Selon la version de pandas, les colonnes textuelles peuvent apparaître avec le type `str` ou `object`.

### 3.3 Examiner les statistiques numériques

```python
df.describe()
```

Cette commande donne notamment le minimum, le maximum et la moyenne des colonnes numériques. Ces informations permettent de repérer d’éventuelles valeurs inhabituelles.

### 3.4 Compter les valeurs manquantes

```python
df.isna().sum()
(df.isna().mean() * 100).round(1)
```

La deuxième instruction calcule le pourcentage de valeurs manquantes pour chaque colonne. Voici comment elle fonctionne :

1. `df.isna()` renvoie `True` lorsqu’une case est vide et `False` lorsqu’elle est renseignée.
2. `.mean()` calcule la moyenne de chaque colonne. Comme Python traite `True` comme `1` et `False` comme `0`, cette moyenne correspond à la proportion de valeurs manquantes.
3. `* 100` transforme cette proportion en pourcentage.
4. `.round(1)` arrondit le résultat à une décimale.

Pour `company_type`, le pourcentage obtenu est d’environ 32,0 %.

### 3.5 Examiner la variable cible

```python
df["target"].value_counts()
df["target"].value_counts(normalize=True)
```

La première instruction compte les occurrences de chaque valeur. La deuxième renvoie leur proportion. On obtient 14 381 candidats dans la classe `0` et 4 777 dans la classe `1`, soit respectivement 75,1 % et 24,9 %.

### 3.6 Vérifier les doublons et l’identifiant

```python
df.duplicated().sum()
df["enrollee_id"].is_unique
```

La première instruction compte les lignes dupliquées. Elle renvoie `0`. La deuxième vérifie si chaque identifiant est unique ; le résultat attendu est `True`.

### 3.7 Compter les valeurs différentes

```python
df.nunique()
```

Cette commande compte le nombre de valeurs distinctes dans chaque colonne. Elle aide à comprendre la nature des variables : une colonne avec peu de catégories est souvent catégorielle, tandis qu’une colonne avec de nombreuses valeurs numériques peut être quantitative. Une colonne qui possède autant de valeurs distinctes que de lignes peut correspondre à un identifiant.

### 3.8 Observer l’effet de `dropna()`

```python
df.shape
df.dropna().shape
```

La première instruction renvoie `(19158, 14)`, tandis que la deuxième renvoie `(8955, 14)`. Cela montre combien de lignes resteraient si je supprimais toutes celles qui contiennent au moins une valeur manquante.

## 4. Résultats de l’exploration

### 4.1 Huit colonnes contiennent des valeurs manquantes

| Colonne | Valeurs manquantes | Pourcentage |
|---|---:|---:|
| `company_type` | 6 140 | 32,0 % |
| `company_size` | 5 938 | 31,0 % |
| `gender` | 4 508 | 23,5 % |
| `major_discipline` | 2 813 | 14,7 % |
| `education_level` | 460 | 2,4 % |
| `last_new_job` | 423 | 2,2 % |
| `enrolled_university` | 386 | 2,0 % |
| `experience` | 65 | 0,3 % |

Les six autres colonnes ne contiennent aucune valeur manquante.

### 4.2 Pourquoi je ne supprime pas toutes les lignes incomplètes

Si j’utilise `dropna()`, le jeu de données passe de 19 158 à 8 955 lignes. Je perdrais donc 10 203 lignes, soit environ 53,3 % des données initiales.

Cette solution pose deux problèmes. D’abord, il resterait moins de la moitié des candidats pour entraîner et évaluer le modèle. Ensuite, les candidats qui n’ont pas renseigné toutes les informations ne sont pas forcément représentatifs des autres. En supprimant systématiquement leurs lignes, je risque de créer un biais et d’obtenir un modèle moins adapté aux données réelles.

Je préfère donc étudier les valeurs manquantes avant de choisir une méthode de traitement.

### 4.3 Certaines valeurs numériques sont enregistrées comme du texte

Deux colonnes demandent une attention particulière :

| Colonne | Valeurs rencontrées | Conversion envisagée |
|---|---|---|
| `experience` | `<1`, `>20` | Remplacer `<1` par `0` et `>20` par `21` |
| `last_new_job` | `never`, `>4` | Remplacer `never` par `0` et `>4` par `5` |
| `company_size` | Tranches comme `10/49` et `50-99` | Conserver l’ordre des tranches et harmoniser leur représentation |

Ces conversions restent des approximations. Par exemple, `>20` indique seulement que l’expérience dépasse 20 ans : la valeur réelle pourrait être 21 ans ou davantage. De même, `<1` peut représenter une durée très courte ou presque un an.

Je devrai donc garder ces limites à l’esprit et effectuer les conversions plus tard, dans le pipeline de préparation des données.

### 4.4 La cible est déséquilibrée

Environ un candidat sur quatre cherche à changer d’emploi. Les autres représentent 75,1 % du jeu de données.

Ce déséquilibre compte au moment d’évaluer le modèle. En effet, un modèle qui prédit toujours `0` obtiendrait déjà 75,1 % d’accuracy, sans détecter un seul candidat qui souhaite changer d’emploi.

### 4.5 Deux colonnes ne doivent pas servir de variables d’entrée

- **`enrollee_id`** est un identifiant unique. Il ne décrit pas le profil du candidat et pourrait encourager le modèle à mémoriser certaines lignes.
- **`target`** correspond à la réponse que le modèle doit prédire. Si je l’inclus dans les variables d’entrée, le modèle aurait déjà accès à la réponse. Les résultats obtenus ne seraient donc pas valides.

Il faut distinguer ces deux raisons : l’identifiant n’apporte pas d’information utile sur le candidat, tandis que la cible doit être séparée des variables d’entrée pour éviter une fuite de données.

### 4.6 Aucun doublon n’a été trouvé

La commande `df.duplicated().sum()` renvoie `0`. Il n’y a donc aucune ligne strictement identique à une autre dans le jeu de données.

Les doublons peuvent poser problème pendant l’apprentissage. Une même observation risque de peser plusieurs fois dans le modèle. Si une copie se trouve dans les données d’entraînement et une autre dans les données de test, les résultats peuvent aussi sembler meilleurs qu’ils ne le sont réellement.

## 5. Comprendre les types de variables

| Type | Explication | Exemples dans le projet |
|---|---|---|
| Numérique | Les valeurs représentent des quantités et leurs écarts ont un sens | `training_hours`, `city_development_index`, `experience` après conversion |
| Catégorielle | Les valeurs représentent des catégories sans ordre naturel | `gender`, `city`, `company_type` |
| Ordinale | Les catégories peuvent être classées, mais les écarts entre elles ne sont pas forcément mesurables | `education_level`, `company_size` |
| Identifiant | La valeur sert à reconnaître une ligne, pas à décrire le candidat | `enrollee_id` |

Un bon moyen de distinguer une variable numérique d’une variable ordinale consiste à se demander si une moyenne a du sens. Calculer le nombre moyen d’heures de formation est pertinent. En revanche, calculer une moyenne entre les catégories `Graduate` et `Masters` ne l’est pas.

Après conversion, `experience` peut être traitée comme une variable numérique. Les valeurs représentent alors des années d’expérience, même si les conversions de `<1` et `>20` restent approximatives.

Quelques points méritent une attention particulière :

- `enrollee_id` est composé de chiffres, mais reste un identifiant.
- `target` est stockée sous forme décimale (`0.0` et `1.0`), mais représente deux catégories.
- `city` possède 123 valeurs différentes. Un encodage one-hot créerait donc 123 colonnes pour cette variable, ce qu’il faudra prendre en compte lors de la préparation des données.

## 6. Évaluer un modèle : accuracy, précision, rappel et F1

### 6.1 Pourquoi l’accuracy ne suffit pas

L’accuracy correspond au nombre de prédictions correctes divisé par le nombre total de prédictions.

Dans ce projet, un modèle qui prédit toujours `0` obtient :

\[
\text{Accuracy} = \frac{14\,381}{19\,158} \approx 75,1\%
\]

Pourtant, ce modèle ne détecte aucun candidat qui souhaite changer d’emploi. Son score paraît correct uniquement parce que la classe `0` est majoritaire.

Sa matrice de confusion serait la suivante :

| | Réalité : change | Réalité : ne change pas |
|---|---:|---:|
| Le modèle prédit « change » | 0 | 0 |
| Le modèle prédit « ne change pas » | 4 777 | 14 381 |

Le rappel de la classe `1` est donc de 0 %. La précision n’est pas définie dans ce cas, car le modèle ne prédit jamais la classe « change ».

### 6.2 Les trois mesures à connaître

Dans ce projet, un résultat positif correspond à un candidat qui cherche à changer d’emploi.

| Mesure | Question à laquelle elle répond | Calcul |
|---|---|---|
| Précision | Parmi les candidats signalés par le modèle, combien cherchent réellement à changer ? | Vrais positifs ÷ prédictions positives |
| Rappel | Parmi tous les candidats qui cherchent à changer, combien le modèle en détecte-t-il ? | Vrais positifs ÷ nombre réel de positifs |
| F1 | Quel équilibre obtient-on entre précision et rappel ? | \(2 \times \frac{\text{précision}\times\text{rappel}}{\text{précision}+\text{rappel}}\) |

Le numérateur des deux premières mesures est le même : le nombre de candidats correctement détectés. C’est le dénominateur qui change.

### 6.3 Exemple sur 100 candidats

Imaginons que 25 candidats souhaitent réellement changer d’emploi. Le modèle prédit un changement pour 20 candidats, et 15 de ces prédictions sont correctes.

| | Réalité : change | Réalité : ne change pas |
|---|---:|---:|
| Le modèle prédit « change » | 15 | 5 |
| Le modèle prédit « ne change pas » | 10 | 70 |

Les résultats sont les suivants :

- **Précision :** \(15/20 = 75\%\).
- **Rappel :** \(15/25 = 60\%\).
- **F1 :** environ 66,7 %.
- **Accuracy :** \((15+70)/100 = 85\%\).

L’accuracy atteint 85 %, mais le modèle ne détecte que 15 des 25 candidats concernés. Il en manque donc 10. C’est pour cette raison que je dois examiner plusieurs métriques plutôt que de me fier à un seul score.

### 6.4 Comment réagir à une accuracy de 78 %

Si quelqu’un annonce une accuracy de 78 %, je dois la comparer au résultat de référence de 75,1 %. L’écart n’est que de 2,9 points de pourcentage.

Ce score ne suffit pas pour conclure que le modèle est bon. Je demanderais aussi la précision, le rappel sur la classe `1`, la matrice de confusion et la confirmation que l’évaluation a été réalisée sur un jeu de test séparé.

## 7. Questions et réponses pour préparer l’entretien

### Q1. Combien de lignes et de colonnes contient le jeu de données ? Que représente une ligne ?

Le fichier contient 19 158 lignes et 14 colonnes. Chaque ligne correspond à un candidat et contient des informations comme sa ville, son niveau d’études ou son expérience. La cible est `target` : `1` si le candidat cherche à changer d’emploi, et `0` sinon. Le problème est donc une classification binaire.

### Q2. Comment la cible est-elle répartie ?

La classe `0` représente 75,1 % des candidats, soit 14 381 personnes. La classe `1` représente 24,9 %, soit 4 777 personnes. Les deux proportions totalisent 100 %.

### Q3. Quelles sont les trois colonnes qui contiennent le plus de valeurs manquantes ?

Il s’agit de `company_type`, avec 6 140 valeurs manquantes (32,0 %), de `company_size`, avec 5 938 (31,0 %), puis de `gender`, avec 4 508 (23,5 %).

### Q4. Y a-t-il des doublons ? Quelles colonnes faut-il retirer des variables d’entrée ?

Aucun doublon n’a été trouvé. Je retire `enrollee_id`, car cette colonne identifie le candidat sans décrire son profil. Je retire aussi `target`, puisqu’il s’agit de la réponse à prédire.

### Q5. Combien de lignes restent après `dropna()` ?

Il reste 8 955 lignes. Cela signifie que 10 203 lignes ont été supprimées, soit 53,3 % du jeu de données initial.

### Q6. Pourquoi s’agit-il d’une classification binaire plutôt que d’une régression ?

Le modèle doit déterminer si un candidat cherche à changer d’emploi. Il doit choisir entre deux catégories : `0` et `1`. Il ne doit pas prédire une quantité continue, comme un prix ou une température.

### Q7. Pourquoi l’accuracy seule ne suffit-elle pas ?

Un modèle qui prédit toujours `0` obtient déjà 75,1 % d’accuracy, mais ne détecte aucun candidat qui souhaite changer d’emploi. Je dois donc aussi regarder la précision, le rappel et le F1 pour la classe `1`.

### Q8. Le modèle signale 20 candidats. Quinze changent réellement d’emploi, sur un total de 25 candidats concernés. Quels sont les scores ?

La précision est de \(15/20 = 75\%\). Le rappel est de \(15/25 = 60\%\). Le F1 est d’environ 66,7 %. La précision mesure la fiabilité des alertes, tandis que le rappel indique la part des vrais cas détectés.

### Q9. Pourquoi éviter `dropna()` ?

Cette méthode ferait perdre 53,3 % des lignes : le jeu passerait de 19 158 à 8 955 candidats. En plus de cette perte importante, les candidats dont le profil est incomplet ne sont pas nécessairement représentatifs de l’ensemble. Le modèle pourrait donc apprendre sur un échantillon biaisé.

### Q10. Pourquoi retirer `enrollee_id` ?

Cette colonne sert uniquement à identifier les candidats. Elle ne donne pas d’information utile sur leur profil. La conserver pourrait aussi encourager le modèle à mémoriser certaines observations au lieu d’apprendre des relations pertinentes.

### Q11. Pourquoi `experience` et `last_new_job` sont-elles stockées comme du texte ?

Ces colonnes contiennent des valeurs telles que `<1`, `>20`, `never` ou `>4`. Ces expressions ne sont pas directement des nombres. Je prévois de convertir `<1` en `0`, `>20` en `21`, `never` en `0` et `>4` en `5`. Ces choix restent approximatifs, car les valeurs exactes ne sont pas connues pour les catégories ouvertes comme `>20`.

### Q12. Comment classer certaines colonnes ?

| Colonne | Type | Justification |
|---|---|---|
| `training_hours` | Numérique | Le nombre d’heures représente une quantité mesurable. |
| `gender` | Catégorielle | Les valeurs représentent des catégories sans ordre naturel. |
| `company_size` | Ordinale | Les catégories correspondent à des tranches de taille ordonnées. |
| `enrollee_id` | Identifiant | Le numéro distingue un candidat sans décrire son profil. |
| `experience` après conversion | Numérique | Les valeurs représentent des années, avec une approximation aux extrémités. |

### Q13. Pourquoi les doublons peuvent-ils poser problème ?

Une ligne répétée peut avoir davantage de poids pendant l’apprentissage. Si une copie se retrouve dans les données d’entraînement et l’autre dans le test, l’évaluation risque aussi de surestimer les performances. Dans ce jeu de données, aucun doublon n’a été détecté.

### Q14. Quel code permet de calculer les valeurs manquantes, la répartition de la cible et les doublons ?

```python
(df.isna().mean() * 100).round(1)
df["target"].value_counts(normalize=True)
df.duplicated().sum()
```

### Q15. Que fait `df.isna().mean()` ?

`isna()` renvoie `True` pour les cases manquantes et `False` pour les autres. Python traite ces valeurs comme `1` et `0` lors du calcul de la moyenne. Le résultat correspond donc à la proportion de cases manquantes. En multipliant cette valeur par 100, on obtient un pourcentage.

## 8. Erreurs à éviter

- Confondre les lignes, les colonnes et les cases. Dire que « 8 955 lignes sont vides » serait incorrect : 8 955 lignes restent après suppression des lignes incomplètes.
- Inverser le sens d’une explication, par exemple dire qu’une colonne décrit le candidat alors qu’elle ne le décrit pas.
- Confondre un effectif avec un pourcentage.
- Donner un chiffre de mémoire sans le vérifier. Il vaut mieux relancer le code en cas de doute.
- Présenter `enrollee_id` comme une simple variable numérique : c’est un identifiant.
- Classer `experience` comme ordinale après sa conversion : elle est alors traitée comme une variable numérique, avec des approximations aux extrémités.
- Signaler le déséquilibre de la cible sans expliquer sa conséquence : il peut rendre l’accuracy trompeuse.

## 9. Les chiffres à retenir

| Élément | Résultat |
|---|---|
| Nombre de lignes et de colonnes | 19 158 lignes, 14 colonnes |
| Répartition de `target` | 75,1 % de `0` et 24,9 % de `1` |
| Nombre de `0` et de `1` | 14 381 et 4 777 |
| Doublons | 0 |
| Lignes après `dropna()` | 8 955 |
| Perte liée à `dropna()` | 53,3 % |
| Colonnes avec des valeurs manquantes | 8 sur 14 |
| `company_type` | 6 140 valeurs manquantes, soit 32,0 % |
| `company_size` | 5 938 valeurs manquantes, soit 31,0 % |
| `gender` | 4 508 valeurs manquantes, soit 23,5 % |
| Accuracy d’un modèle qui prédit toujours `0` | 75,1 % |

## 10. Configuration du projet

Le projet utilise `uv` pour gérer les dépendances, qui sont déclarées dans `pyproject.toml` et installées dans l’environnement `.venv`.

Pour installer les dépendances, j’utilise :

```bash
uv sync
```

Dans VS Code, je dois également sélectionner l’interpréteur Python de `.venv`. Pour vérifier le chemin utilisé par le notebook :

```python
import sys
print(sys.executable)  # doit finir par .venv\Scripts\python.exe
```

## 11. Points restant à vérifier

- Comparer les descriptions des colonnes avec la page Kaggle, en particulier `last_new_job`, `enrolled_university` et `relevent_experience`.
- Ajouter le lien de la source Kaggle et les informations de licence dans le README principal.

## Étape suivante

La phase 2 consistera à étudier les variations de la cible selon les différentes colonnes. Pour la suite, je pourrai m’appuyer sur `phase2.md`.
