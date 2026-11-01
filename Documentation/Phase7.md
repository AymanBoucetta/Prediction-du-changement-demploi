# Phase 7 : évaluer le modèle sur le jeu de test

Jusqu'ici, tous mes choix se sont faits sur le jeu d'entraînement, par validation croisée. Cette phase fait quelque chose de différent : elle juge le modèle retenu sur des candidats qu'il n'a **jamais vus**, **une seule fois**. C'est la mesure la plus honnête de ce que le modèle vaut, et c'est elle qu'on cite à la fin.

Trois règles encadrent toute la phase :

- **Une seule évaluation.** Pas de « je relance pour voir ».
- **Rien ne change après avoir vu le résultat** : ni le modèle, ni le seuil de décision, ni les réglages. Sinon, le test devient un deuxième jeu de validation et son score n'est plus honnête.
- **Le modèle est fixé avant.** Ici : la régression logistique avec `class_weight="balanced"`, retenue en phase 6 parce qu'elle est à égalité avec le Random Forest régularisé et plus simple.

## La mise en place

J'ai créé un notebook séparé pour cette phase. Comme les variables ne passent pas d'un notebook à l'autre, sa première cellule recrée les données, le découpage (avec la même graine, donc les mêmes jeux), le pipeline de préparation et le modèle final, **entraîné sur le train seulement** :

```python
modele_final = Pipeline([
    ("prep", preprocess),
    ("modele", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
])
modele_final.fit(X_train, y_train)
```

La sortie `(15326, 12) (3832, 12)` confirme que les deux jeux sont ceux des phases précédentes. Cette cellule ne touche pas au test.

## L'évaluation

```python
pred_test = modele_final.predict(X_test)

confusion_matrix(y_test, pred_test)
accuracy_score(y_test, pred_test)
precision_score(y_test, pred_test)
recall_score(y_test, pred_test)
f1_score(y_test, pred_test)
```

### La matrice de confusion

| | Prédit `0` | Prédit `1` |
|---|---|---|
| **Réalité `0`** (2 877) | 2 214 bien rejetés | 663 fausses alertes |
| **Réalité `1`** (955) | 207 ratés | 748 bien trouvés |

Les quatre cases font bien 3 832, le nombre de candidats du test.

### Les scores

| Mesure | Test | Validation croisée (train) |
|---|---|---|
| Accuracy | 77,3 % | 76,5 % ± 0,7 |
| Précision | 53,0 % | 52,0 % ± 1,0 |
| Rappel | **78,3 %** | 74,1 % ± 1,4 |
| F1 | **63,2 %** | 61,1 % ± 0,8 |

Le score du test est proche de celui de la validation croisée, avec un rappel et un F1 un peu au-dessus. Ce n'est pas un problème : le test n'a servi à aucun choix, donc rien n'a fuité. Mais il faut éviter de dire que le modèle « vaut 78 % ». Le test ne contient que 955 candidats qui changent, donc le rappel mesuré dessus varie lui-même d'environ un point et demi. Le plus honnête est de dire que le rappel est **autour de 75 à 78 %**, en citant les deux mesures.

Il n'y a pas de surapprentissage : les scores du train, de la validation croisée et du test sont proches.

### Ce que ça veut dire, candidat par candidat

- Le modèle trouve **748 des 955** candidats qui changent, et en rate 207.
- Il lance 1 411 alertes (663 + 748), dont **663 sont fausses** : 23,0 % des 2 877 candidats qui ne changent pas sont signalés à tort.
- Parmi les candidats signalés, 53,0 % changent vraiment, contre 24,9 % au hasard : le modèle fait environ **2,1 fois mieux que le hasard**.
- Face au modèle de référence (toujours « 0 ») : accuracy de 77,3 % contre 75,1 %, mais surtout rappel de 78,3 % contre 0 %.

## ROC-AUC et PR-AUC

Ces deux mesures ne dépendent pas du seuil de 0,5 : elles jugent la qualité du **classement** des candidats par probabilité. Les calculer sur le test ne change rien au modèle, qui est déjà figé.

```python
proba_test = modele_final.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, proba_test)             # 0.81
average_precision_score(y_test, proba_test)   # 0.555
```

- **ROC-AUC = 0,81.** Si on prend au hasard un candidat qui change et un qui ne change pas, le modèle donne la probabilité la plus haute au premier dans environ 81 % des cas. Le hasard vaut 0,5 et un classement parfait 1.
- **PR-AUC = 0,555.** Son repère est la part de candidats qui changent, 24,9 % : un modèle au hasard obtient ce score. Le modèle fait donc environ 2,2 fois mieux.

Ces deux scores confirment que le modèle apprend un vrai signal, et qu'on peut jouer sur le seuil pour changer le compromis entre rappel et précision.

## Le seuil de décision

Le seuil de 0,5 est un choix par défaut : au-dessus, le modèle répond « change ». Le monter donne moins d'alertes, le baisser en donne plus. Je ne le règle **pas** avec le test (ce serait le transformer en jeu de validation). Je regarde l'effet sur le train avec des prédictions « hors pli » : chaque candidat est prédit par un modèle qui ne l'a pas vu.

```python
proba_cv = cross_val_predict(modele_final, X_train, y_train, cv=cv,
                             method="predict_proba")[:, 1]

for seuil in [0.3, 0.4, 0.5, 0.6, 0.7]:
    p = (proba_cv >= seuil).astype(int)
    # précision, rappel, F1 pour chaque seuil
```

| Seuil | Précision | Rappel | F1 |
|---|---|---|---|
| 0,3 | 36,6 % | 87,0 % | 51,5 % |
| 0,4 | 45,5 % | 79,4 % | 57,8 % |
| **0,5** | **52,0 %** | **74,1 %** | **61,1 %** |
| 0,6 | 56,7 % | 66,0 % | 61,0 % |
| 0,7 | 58,6 % | 50,3 % | 54,1 % |

Le mécanisme est simple : un seuil bas lance beaucoup d'alertes, donc on trouve presque tous les candidats qui changent (rappel élevé) mais avec beaucoup de fausses alertes (précision basse). Un seuil haut fait l'inverse. Le F1 est maximal autour de 0,5 à 0,6 (61,1 % et 61,0 %) : **le seuil par défaut est déjà le meilleur compromis pour le F1**, je ne le change donc pas.

## Faut-il craindre plus les ratés ou les fausses alertes ?

Ça dépend de l'usage, que les données ne tranchent pas :

- Si **rater** un candidat qui cherche à changer coûte cher (une opportunité perdue), on baisse le seuil pour monter le rappel : à 0,4, on trouve 79,4 % des candidats au prix d'une précision de 45,5 %.
- Si chaque **fausse alerte** coûte cher (du temps passé à contacter quelqu'un pour rien), on monte le seuil : à 0,6, la précision atteint 56,7 % pour un rappel de 66,0 %.
- Sans usage précis, le F1 équilibre les deux, et le seuil de 0,5 reste le bon choix.

Le modèle prédit des probabilités : c'est un outil pour **classer** des candidats et prioriser des contacts, pas pour trancher avec certitude.

## Les chiffres à retenir

| | |
|---|---|
| Matrice de confusion (test) | 2 214 bien rejetés, 663 fausses alertes, 207 ratés, 748 bien trouvés |
| Accuracy / précision / rappel / F1 (test) | 77,3 % / 53,0 % / 78,3 % / 63,2 % |
| ROC-AUC / PR-AUC (test) | 0,81 / 0,555 |
| Candidats qui changent trouvés | 748 sur 955 |
| Fausses alertes | 663 sur 2 877 candidats qui ne changent pas (23,0 %) |
| Meilleur seuil (F1) | 0,5 |
| Référence (toujours « 0 ») | accuracy 75,1 %, rappel 0 % |

## Les limites

- **Une seule évaluation sur 3 832 candidats**, dont 955 qui changent : les scores ont une marge d'environ un point et demi. Je cite donc des ordres de grandeur.
- **Je n'ai pas fait de test statistique** pour comparer le modèle aux autres.
- **Le seuil analysé est celui des prédictions hors pli** sur le train : c'est une indication, pas un réglage fin.
- **Je n'ai exploré qu'un petit nombre de modèles et de réglages.** Un autre modèle ou un meilleur réglage pourrait faire mieux.
- **Les données sont publiques** : ce ne sont pas celles d'une vraie entreprise, et le modèle n'a pas été testé sur d'autres populations (une autre ville, une autre période).
- **Le modèle repose en partie sur `city`** (122 colonnes) : je regarderai à la phase suivante si une ville pèse trop.

## Prochaine étape

La phase 8 : lire les coefficients de la régression logistique pour comprendre quelles caractéristiques pèsent le plus dans les prédictions, et vérifier que le modèle ne s'appuie pas trop sur une seule ville.