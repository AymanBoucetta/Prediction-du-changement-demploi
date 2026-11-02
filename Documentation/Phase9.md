# Phase 9 : garder une trace des expériences avec MLflow

Au fil des phases, j'ai entraîné plusieurs modèles : une référence, deux régressions logistiques, deux Random Forest. Leurs scores étaient dispersés entre des sorties de notebook. MLflow règle ce problème : il enregistre chaque essai, avec ses réglages et ses scores, au même endroit, pour pouvoir les comparer et les retrouver plus tard.

Une règle des phases précédentes continue de s'appliquer : **les scores enregistrés pour comparer les modèles sont ceux de la validation croisée, sur le jeu d'entraînement.** Le jeu de test sert uniquement à évaluer une fois le modèle officiel.

## Le vocabulaire utile

- **Expérience** : un dossier qui regroupe les essais d'un même projet. Ici, `job-change-prediction`.
- **Run** : un essai, avec ses **paramètres** (les réglages choisis) et ses **métriques** (les scores obtenus).
- **Tracking URI** : l'endroit où MLflow range ses données. J'utilise une petite base de données locale, le fichier `mlflow.db`.
- **Artefact** : un fichier lié à un run, comme le modèle entraîné lui-même.
- **Interface MLflow** : une page web locale qui affiche et compare les runs.

## La mise en place

MLflow est installé dans l'environnement du projet avec uv (`uv add mlflow`). Je travaille dans un notebook séparé, `9_mlflow.ipynb`, dont la première cellule recrée les données, le découpage et le modèle final, comme pour les phases précédentes. Puis :

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("job-change-prediction")
```

`mlflow.db` est créé à l'endroit d'où tourne le notebook.

## Enregistrer un run

```python
with mlflow.start_run(run_name="logreg_v1"):
    mlflow.log_param("modele", "LogisticRegression")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("villes", "toutes")
    s = cross_validate(modele_final, X_train, y_train, cv=cv, scoring=scoring)
    for m in scoring:
        mlflow.log_metric(f"cv_{m}", s[f"test_{m}"].mean())
```

`start_run` ouvre un run et tout ce qui est dans le bloc `with` y est enregistré. `log_param` garde un réglage, `log_metric` un score. Pour ne pas répéter ce code, j'en ai fait une petite fonction qui enregistre les moyennes et les écarts-types des quatre mesures, et je l'ai appliquée à cinq modèles.

### Les cinq runs de comparaison

| Run | Réglage | `cv_f1` | `cv_recall` |
|---|---|---|---|
| `reference` | répond toujours « 0 » | 0,000 | 0,000 |
| `rf_defaut` | Random Forest, 200 arbres | 0,584 | 0,623 |
| `logreg_v1` | régression logistique, toutes les villes | 0,611 | 0,741 |
| `logreg_v2` | régression logistique, villes rares regroupées | 0,610 | 0,743 |
| `rf_regularise` | Random Forest, profondeur 8, 20 candidats par feuille | 0,610 | 0,752 |

Ces chiffres retrouvent exactement ceux de la phase 6, ce qui confirme que l'enregistrement est fidèle. Le surapprentissage du Random Forest d'origine se lit directement dans le tableau.

### Le modèle officiel

Le modèle retenu (la v1, seul évalué sur le test) a son propre run, avec le modèle entraîné enregistré. Les métriques sont calculées à partir de `X_test` et `y_test`, puis enregistrées :

```python
y_pred = modele_final.predict(X_test)
y_proba = modele_final.predict_proba(X_test)[:, 1]
test_metrics = {
    "test_accuracy": accuracy_score(y_test, y_pred),
    "test_precision": precision_score(y_test, y_pred, zero_division=0),
    "test_recall": recall_score(y_test, y_pred, zero_division=0),
    "test_f1": f1_score(y_test, y_pred, zero_division=0),
    "test_roc_auc": roc_auc_score(y_test, y_proba),
    "test_pr_auc": average_precision_score(y_test, y_proba),
}

with mlflow.start_run(run_name="logreg_v1_final"):
    mlflow.log_params({"modele": "LogisticRegression", "class_weight": "balanced",
                       "villes": "toutes", "seuil": 0.5})
    mlflow.sklearn.log_model(
        modele_final, name="modele",
        skops_trusted_types=["__main__.convertir_numeriques", "numpy.dtype"],
    )
    mlflow.set_tag("statut", "officiel_evalue_sur_test")
    mlflow.log_metrics(test_metrics)
```

Le jeu de test n'est utilisé que pour cette évaluation finale, pas pour comparer ou choisir les modèles.

Les fichiers du modèle ne sont pas dans `mlflow.db`, qui ne contient que les paramètres, les scores et les noms : ils sont rangés dans un dossier d'artefacts, à côté de la base.

## Voir les résultats

```
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Puis `http://127.0.0.1:5000`. Dans l'interface, on passe sur l'onglet **Model training**, puis **Experiments**, puis `job-change-prediction`. On peut trier par score, afficher les colonnes qu'on veut et comparer deux runs. Depuis le notebook :

```python
mlflow.search_runs(experiment_names=["job-change-prediction"])[
    ["tags.mlflow.runName", "status", "metrics.cv_f1", "metrics.test_f1"]]
```

## Ce qui m'a posé problème

- **L'interface n'affichait qu'une expérience `Default`.** Sans l'option `--backend-store-uri`, elle lit sa propre base, pas la mienne. Il faut lui indiquer le fichier `mlflow.db`.
- **Un run enregistré sans score.** Un premier essai a échoué parce que la cellule de démarrage n'avait pas été exécutée (`cross_validate` n'était pas défini). Le run avait été créé avant l'erreur, donc il est resté dans la liste, sans métriques. Le notebook final l'exclut de la comparaison sans supprimer de données MLflow.
- **MLflow refusait de sauvegarder le modèle** (`UntrustedTypesFoundException`). Pour des raisons de sécurité, il n'accepte que des types connus. Ma fonction de conversion est définie dans le notebook, et un type de numpy est signalé aussi. Je les déclare comme sûrs avec `skops_trusted_types`.
- **Des runs en double.** Relancer une cellule d'enregistrement crée de nouveaux runs sans écraser les anciens : j'avais 11 runs au lieu de 6. La comparaison affiche le dernier run de chaque nom, sans supprimer les anciens.

## Les limites

- **Un modèle sauvegardé dépend de ma fonction `convertir_numeriques`.** Pour le recharger, la fonction doit exister avec le même nom au même endroit, et les mêmes types de confiance doivent être redéclarés. La phase suivante range le code dans des fichiers `.py` pour corriger ça.
- **Le modèle est défini dans le notebook.** Le suivi MLflow enregistre le pipeline entraîné, mais le code réutilisable sera déplacé dans des fichiers `.py` à la phase suivante.
- **Le suivi est local** : `mlflow.db` et le dossier d'artefacts restent sur mon ordinateur. Je les exclus du dépôt avec le `.gitignore`, et je garde une capture de l'interface comme preuve.
- **Les essais sont peu nombreux** (5 modèles). MLflow serait surtout utile avec beaucoup d'essais ou un réglage systématique des hyperparamètres.

## Prochaine étape

La phase 10 : ranger le code dans des fichiers `.py` (chargement des données, pipeline, entraînement, évaluation), pour qu'une seule commande reproduise le projet.