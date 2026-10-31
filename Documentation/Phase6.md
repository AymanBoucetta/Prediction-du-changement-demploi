# Phase 6 : entraîner et comparer des modèles

Les données sont prêtes (phase 5). Cette phase répond à une question simple : **quel modèle prédit le mieux si un candidat cherche à changer d'emploi ?** Je commence par un modèle volontairement naïf pour fixer un plancher, puis j'essaie une régression logistique et un Random Forest, et je les compare sans toucher au jeu de test.

Une règle pose le cadre de toute la phase : **le jeu de test reste intact.** Tous les choix (quel modèle, quels réglages) se font avec le jeu d'entraînement, par validation croisée. Le test ne servira qu'une seule fois, en phase 7.

## Le vocabulaire utile

- **Modèle de référence** (*baseline*) : un modèle sans intelligence, qui sert de plancher. Un vrai modèle qui ne fait pas mieux n'a rien appris.
- **Régression logistique** : un modèle qui calcule, pour chaque candidat, une probabilité entre 0 et 1 de chercher à changer. Au-dessus de 0,5, il répond `1`, sinon `0`. Ses coefficients se lisent directement.
- **`class_weight="balanced"`** : donne plus de poids aux candidats de la classe minoritaire (ceux qui changent), pour que le modèle ne se contente pas de répondre toujours `0`.
- **Random Forest** : un grand nombre d'arbres de décision, qui votent ensemble. Il capte des relations plus complexes, mais peut aussi mémoriser le jeu d'entraînement.
- **Surapprentissage** : le modèle mémorise le train au lieu d'apprendre des règles. Il a d'excellents scores sur le train et de moins bons sur de nouveaux candidats.
- **Hyperparamètre** : un réglage qu'on choisit avant l'entraînement (par exemple la profondeur maximale des arbres).
- **Validation croisée** : on découpe le train en plusieurs parts, on entraîne sur toutes sauf une et on juge sur celle qui reste, puis on recommence en changeant de part. Chaque candidat est jugé une fois sans que le modèle l'ait vu.
- **Pli** (*fold*) : l'une de ces parts. Ici, 5 plis.
- **Écart-type** (le `±` dans les résultats) : de combien le score varie d'un pli à l'autre. Si l'écart entre deux modèles est plus petit que ce `±`, la différence n'est pas fiable.
- **Matrice de confusion, précision, rappel, F1** : voir la phase 1.

## La mise en place

J'ai créé un nouveau notebook pour cette phase, `6_exploration.ipynb`. Les variables d'un notebook ne passent pas dans un autre : j'ai donc recopié en première cellule tout ce dont j'ai besoin (chargement des données, découpage avec `random_state=42`, conversions, pipeline de préparation). Je retrouve `(15326, 164) (3832, 164)` et aucune case vide, donc le même état qu'à la fin de la phase 5. Plus tard, ce code ira dans un fichier `.py`.

## Étape 1 : le modèle de référence

```python
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(Xtr_prep, y_train)
print(dummy.score(Xtr_prep, y_train))
print(dummy.predict(Xtr_prep)[:10])
```

Ce modèle ignore toutes les colonnes et répond toujours la classe la plus fréquente, donc `0`. Son accuracy est de **75 %** (exactement 11 504 ÷ 15 326 = 75,06 %, la part de candidats qui ne changent pas), et ses dix premières prédictions sont des `0`.

Son **rappel est de 0 %** : il ne trouve aucun candidat qui change, puisqu'il ne dit jamais « change ». Sa précision n'est pas calculable, et scikit-learn affiche 0 avec un avertissement. C'est le plancher à battre : 75 % d'accuracy sans rien comprendre.

## Étape 2 : la régression logistique

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
logreg.fit(Xtr_prep, y_train)
pred = logreg.predict(Xtr_prep)
```

Scores sur le train, face au modèle de référence :

| | Référence | Régression logistique |
|---|---|---|
| Accuracy | 75,1 % | 77,1 % |
| Précision | non calculable | 52,9 % |
| Rappel | 0 % | **75,2 %** |
| F1 | 0 % | 62,1 % |

L'accuracy ne gagne que 2 points, mais le rappel passe de 0 à 75,2 % : le modèle retrouve environ trois candidats sur quatre parmi ceux qui changent. C'est le piège de l'accuracy décrit en phase 1 : elle cache le vrai gain.

L'effet de `class_weight="balanced"` explique la forme du résultat. Comme les candidats qui changent pèsent davantage, le modèle prend le risque de dire « change » plus souvent. Il en trouve beaucoup (le rappel monte), mais fait aussi des fausses alertes (la précision n'est que de 52,9 %). On échange de la précision contre du rappel. L'accuracy, elle, reste proche de celle de la référence. Ces scores sont calculés sur le train : ils sont optimistes, ce n'est qu'un premier coup d'œil.

## Étape 3 : la matrice de confusion

```python
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train, pred))
```

| | Prédit `0` | Prédit `1` |
|---|---|---|
| **Réalité `0`** (11 504) | 8 948 bien rejetés | 2 556 fausses alertes |
| **Réalité `1`** (3 822) | 947 ratés | 2 875 bien trouvés |

Les quatre cases font bien 15 326. On retrouve les scores :

- **Rappel** : 2 875 ÷ 3 822 = 75,2 %. Le modèle trouve 2 875 candidats qui changent et en rate 947.
- **Précision** : 2 875 ÷ (2 875 + 2 556) = 52,9 %. Il a lancé 5 431 alertes, dont 2 556 sont fausses.

Les fausses alertes (2 556) sont un peu moins nombreuses que les candidats bien trouvés (2 875), d'où une précision juste au-dessus de 50 %. Mais elles touchent 22 % des candidats qui ne changent pas (2 556 sur 11 504). Le modèle a choisi de rater peu de candidats, au prix de nombreuses fausses alertes.

## Étape 4 : le Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                            random_state=42, n_jobs=-1)
rf.fit(Xtr_prep, y_train)
pred_rf = rf.predict(Xtr_prep)
```

Sur le train : **accuracy de 99,5 % et rappel de 100 %.** C'est bien au-dessus de la régression logistique (77,1 %), mais ça ne prouve pas que le modèle est meilleur. Il retrouve tous les candidats qui changent parce qu'il a **mémorisé** les 15 326 candidats d'entraînement : c'est du **surapprentissage**. Face à des candidats jamais vus, il fera beaucoup moins bien. Un score sur le train ne dit pas si un modèle est bon.

## Étape 5 : la validation croisée

Pour juger honnêtement, on utilise la validation croisée sur le train : 5 plis, stratifiés (la même proportion de candidats qui changent dans chaque pli), mélangés avec une graine fixe.

Un point technique décisif : le pipeline de préparation est placé **à l'intérieur** de chaque modèle. À chaque tour, il recalcule les médianes, les échelles et les catégories sur les 4 plis d'entraînement seulement, puis les applique au pli de jugement. Sinon, les valeurs de remplissage seraient calculées avec des candidats du pli de jugement, ce qui serait une fuite de données.

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1"]

for nom, modele in modeles.items():
    pipe = Pipeline([("prep", preprocess), ("modele", modele)])
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
```

Résultats (moyenne sur 5 plis) :

| | Accuracy | Précision | Rappel | F1 |
|---|---|---|---|---|
| Référence | 75,1 % | 0,0 % | 0,0 % | 0,0 % |
| Régression logistique | 76,5 % | 52 % | **74,1 %** | **61,1 %** |
| Random Forest | **77,9 %** | **55 %** | 62,3 % | 58,4 % |

- Le surapprentissage du Random Forest est confirmé : son rappel passe de 100 % sur le train à 62,3 % en validation croisée, et son accuracy de 99,5 % à 77,9 %.
- La régression logistique est stable : 75,2 % de rappel sur le train, 74,1 % en validation croisée. Elle ne mémorise pas.
- Le classement dépend de la mesure. L'accuracy met le Random Forest devant (77,9 % contre 76,5 %), mais le rappel et le F1 mettent la régression logistique en tête. L'accuracy est encore le critère le moins fiable ici.

Le modèle qui a le meilleur rappel est donc la régression logistique (74,1 % contre 62,3 %). Le Random Forest est un peu meilleur sur la précision (55 % contre 52 %).

## Étape 6 : un Random Forest qui mémorise moins

Le Random Forest est pénalisé par ses arbres trop profonds, qui épousent le train. J'ai limité deux hyperparamètres :

- `max_depth=8` : chaque arbre a au maximum 8 niveaux.
- `min_samples_leaf=20` : une feuille doit regrouper au moins 20 candidats, ce qui empêche de mémoriser des cas individuels.

```python
rf_reg = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                class_weight="balanced", random_state=42, n_jobs=-1)
```

| | Train | Validation croisée | Écart |
|---|---|---|---|
| Accuracy | 76,3 % | 76,0 % | 0,3 point |
| Rappel | 75,8 % | 75,2 % | 0,6 point |

L'écart de rappel passe de 37,7 points (100 % contre 62,3 %) à 0,6 point : le surapprentissage a disparu. Le prix : l'accuracy de validation baisse (77,9 % à 76,0 %), mais le rappel de validation monte (62,3 % à 75,2 %). Les scores sur le train sont maintenant honnêtes.

## Étape 7 : comparer les deux finalistes

Avec le Random Forest régularisé, deux modèles se tiennent de près. Je les compare sur les mêmes mesures, avec l'écart-type des 5 plis :

| Mesure | Régression logistique | Random Forest régularisé | Écart | Plus grand `±` |
|---|---|---|---|---|
| Accuracy | 76,5 % ± 0,7 | 76,0 % ± 0,9 | 0,5 point | 0,9 |
| Précision | 52,0 % ± 1,0 | 51,3 % ± 1,3 | 0,7 point | 1,3 |
| Rappel | 74,1 % ± 1,4 | 75,2 % ± 1,1 | 1,1 point | 1,4 |
| F1 | 61,1 % ± 0,8 | 61,0 % ± 1,1 | 0,1 point | 1,1 |

**Chaque écart est plus petit que l'écart-type.** Les plis varient plus que la différence entre les deux modèles : on ne peut pas affirmer que l'un est meilleur que l'autre. Ils sont à égalité.

## Choisir un modèle

**Quelle mesure compte le plus ?** Ça dépend de l'usage. Si rater un candidat qui cherche à changer coûte cher (une opportunité perdue), on privilégie le **rappel**. Si chaque fausse alerte coûte cher (du temps passé à contacter quelqu'un pour rien), on privilégie la **précision**. Sans usage précis, le **F1** équilibre les deux, et c'est celui que je prends comme repère, en gardant un œil sur le rappel.

**Quel modèle ?** Quand deux modèles sont à égalité, on garde le plus simple. Je retiens la **régression logistique** :

- elle est à égalité avec le Random Forest régularisé sur toutes les mesures (F1 de 61,1 % contre 61,0 %) ;
- elle est stable entre le train et la validation croisée ;
- elle est plus rapide, n'a pas d'hyperparamètres de profondeur à régler et ses coefficients se lisent directement, ce qui servira à l'interprétation en phase 8.

Ce choix est provisoire : il vaut pour ce critère. Si l'usage demandait un rappel maximal, le Random Forest régularisé (75,2 %) serait à peine devant, sans écart fiable.

## Les chiffres à retenir

| | |
|---|---|
| Référence (toujours « 0 ») | accuracy 75,1 %, rappel 0 % |
| Régression logistique, train | accuracy 77,1 %, précision 52,9 %, rappel 75,2 %, F1 62,1 % |
| Matrice de confusion (logreg, train) | 2 875 bien trouvés, 947 ratés, 2 556 fausses alertes, 8 948 bien rejetés |
| Random Forest, train | accuracy 99,5 %, rappel 100 % (surapprentissage) |
| Random Forest, validation croisée | accuracy 77,9 %, rappel 62,3 % |
| Régression logistique, validation croisée | accuracy 76,5 %, précision 52,0 %, rappel 74,1 %, F1 61,1 % |
| Random Forest régularisé, validation croisée | accuracy 76,0 %, précision 51,3 %, rappel 75,2 %, F1 61,0 % |

## Les limites

- **La validation croisée se fait sur le train.** Le score final et honnête viendra du test, en phase 7.
- **Je n'ai essayé qu'un seul réglage** du Random Forest (`max_depth=8`, `min_samples_leaf=20`). Un réglage plus fin pourrait donner un meilleur modèle.
- **Le seuil de décision est celui par défaut (0,5).** Le déplacer ferait monter le rappel ou la précision : c'est un autre moyen de choisir entre les deux.
- **Aucun test statistique** n'a été fait sur les écarts entre les modèles : je compare avec l'écart-type des plis.
- **`city` ajoute 122 colonnes** qui recoupent `city_development_index` : ça peut gêner la régression logistique et sera à regarder.
- **Les données sont publiques**, pas celles d'une vraie entreprise.

## Prochaine étape

La phase 7 : évaluer le modèle retenu **une seule fois** sur le jeu de test, avec la matrice de confusion, la précision, le rappel et le F1.