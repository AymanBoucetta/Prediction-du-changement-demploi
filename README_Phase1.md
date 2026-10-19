# Prédiction du changement d'emploi de candidats data

Projet de machine learning sur données tabulaires : à terme, prédire si un candidat cherche à changer d'emploi à partir de son profil.

> **État d'avancement :** phase 1 terminée (compréhension des données). Ce dépôt contient pour l'instant l'exploration initiale du jeu de données. Le traitement des valeurs manquantes, la modélisation et l'évaluation sont les prochaines étapes.

## Contexte

Le but du projet est de s'entraîner sur un cas réaliste de classification avec des données imparfaites : valeurs manquantes, colonnes de nombres stockées en texte et classes déséquilibrées. Cette première phase sert à bien comprendre les données avant de les modifier.

## Jeu de données

- **Source :** Kaggle, « HR Analytics: Job Change of Data Scientists »
- **Fichier utilisé :** `aug_train.csv`, placé dans `data/raw/` et jamais modifié.
- **Taille :** 19 158 candidats (lignes) et 14 colonnes.
- **Variable cible :** `target`, qui vaut 1 si le candidat cherche à changer d'emploi et 0 sinon. C'est un problème de **classification binaire**.
- **Répartition de la cible :** 75,1 % de 0 et 24,9 % de 1.
- **Doublons :** aucun.

### Dictionnaire des colonnes

| Colonne | Type | Nature | Description | Manquantes |
|---|---|---|---|---|
| `enrollee_id` | entier | identifiant | identifiant unique du candidat, non utilisé comme variable | 0 % |
| `city` | texte | catégorielle (123 valeurs) | code de la ville du candidat | 0 % |
| `city_development_index` | décimal | numérique | indice de développement de la ville, entre 0 et 1 | 0 % |
| `gender` | texte | catégorielle | genre du candidat | 23,5 % |
| `relevent_experience` | texte | catégorielle | a-t-il une expérience pertinente pour le poste, oui ou non | 0 % |
| `enrolled_university` | texte | catégorielle | type de cursus universitaire suivi en ce moment, s'il y en a un | 2,0 % |
| `education_level` | texte | ordinale | niveau d'études | 2,4 % |
| `major_discipline` | texte | catégorielle | domaine d'études principal | 14,7 % |
| `experience` | texte | numérique stockée en texte | années d'expérience totale (valeurs spéciales `<1` et `>20`) | 0,3 % |
| `company_size` | texte | ordinale | taille de l'entreprise actuelle, en tranches (par exemple `50-99`) | 31,0 % |
| `company_type` | texte | catégorielle | type de l'entreprise actuelle | 32,0 % |
| `last_new_job` | texte | numérique stockée en texte | années écoulées entre l'avant-dernier emploi et l'emploi actuel (valeurs `never` et `>4`) | 2,2 % |
| `training_hours` | entier | numérique | nombre d'heures de formation suivies | 0 % |
| `target` | décimal | cible (classe) | 1 si le candidat cherche à changer d'emploi, 0 sinon | 0 % |

Le nom `relevent_experience` est écrit tel qu'il apparaît dans le jeu de données.

## Ce que montre l'exploration initiale

### 1. Valeurs manquantes

8 colonnes sur 14 contiennent des valeurs manquantes :

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

Supprimer toutes les lignes qui ont au moins une case vide (`dropna()`) ferait passer le jeu de **19 158 à 8 955 lignes**, soit **53 % des données perdues**. Les candidats qui laissent des champs vides ne sont en outre peut-être pas représentatifs des autres. Les valeurs manquantes seront donc traitées colonne par colonne.

### 2. Colonnes mal typées

Plusieurs colonnes qui représentent des nombres sont stockées en texte à cause de valeurs spéciales :

- `experience` : `<1` et `>20` à côté de nombres exacts.
- `last_new_job` : `never` et `>4`.
- `company_size` : des tranches comme `50-99`.

Elles devront être converties en valeurs numériques ou ordinales, en documentant les choix faits pour les bornes.

### 3. Classes déséquilibrées

Environ 3 candidats sur 4 ne cherchent pas à changer d'emploi. Un modèle qui répondrait toujours « 0 » aurait donc environ 75 % d'accuracy sans rien apprendre. L'accuracy seule ne suffira pas pour juger les modèles : la précision, le rappel et le F1 sur la classe 1 seront utilisés.

### 4. Colonne à exclure

`enrollee_id` est un identifiant unique par candidat. Il ne décrit pas le profil et pourrait pousser un modèle à mémoriser les lignes d'entraînement (surapprentissage) : il sera retiré des variables d'entrée.

## Structure du dépôt

```
HR-Analytics/
├── README.md
├── pyproject.toml
├── uv.lock
├── data/
│   └── raw/                  # fichiers Kaggle (non versionnés)
└── notebooks/
    └── 01_exploration.ipynb  # exploration initiale
```

## Installation

Le projet utilise [uv](https://docs.astral.sh/uv/).

```bash
git clone <URL-du-dépôt>
cd HR-Analytics
uv sync
```

Télécharge le jeu de données depuis Kaggle et place `aug_train.csv` dans `data/raw/`. Ouvre ensuite `notebooks/01_exploration.ipynb` avec le noyau de l'environnement `.venv`.

## Prochaines étapes

1. Exploration : taux de la cible par catégorie et distributions des variables numériques.
2. Traitement des valeurs manquantes, colonne par colonne, avec la justification de chaque décision.
3. Découpage train/test stratifié avant toute imputation, puis pipeline scikit-learn.
4. Comparaison de modèles (modèle de référence, régression logistique, Random Forest).
5. Évaluation sur le jeu de test : précision, rappel, F1 et matrice de confusion.

## Outils

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn.

## Auteur

Ayman Boucetta, élève ingénieur IA/ML à l'ISMAGI (Témara, Maroc).