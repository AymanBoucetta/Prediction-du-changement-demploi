# Phase 8 : comprendre ce que le modèle a appris

Un modèle qui obtient de bons scores ne dit pas pourquoi. Cette phase ouvre la régression logistique pour voir **quelles caractéristiques pèsent dans ses prédictions**, vérifier qu'elles sont cohérentes avec ce que j'avais vu dans les données, et repérer ce qui ne l'est pas. C'est aussi la phase qui m'a fait trouver un défaut du modèle.

Une règle de la phase 7 continue de s'appliquer : **le jeu de test ne sert plus**. Tout ce qui suit se fait sur le jeu d'entraînement, en validation croisée. Le score du test (rappel de 78,3 %, F1 de 63,2 %) vient du modèle d'origine, noté **v1**, et ne change pas.

## Comment lire un coefficient

La régression logistique donne à chaque colonne un nombre, son **coefficient** :

- Un coefficient **positif** pousse le modèle vers « change », un coefficient **négatif** vers « ne change pas ».
- Plus il est grand en valeur absolue, plus la colonne pèse.
- `exp(coefficient)` donne le facteur par lequel les **chances** (*odds*) sont multipliées. Un coefficient de 1,12 multiplie les chances par environ 3,1.

Trois précautions pour ne pas mal lire :

- **« Toutes choses égales par ailleurs ».** Un coefficient mesure l'effet d'une colonne en tenant compte de toutes les autres. Il peut donc différer du taux brut vu en phase 2.
- **Il faut comparer les catégories d'une même colonne.** Avec l'encodage one-hot, chaque catégorie a son coefficient, et le signe seul ne dit rien : c'est l'**écart** entre deux catégories qui compte. Par exemple, `Graduate` (+0,59) et `Primary School` (−0,98) ont 1,58 d'écart, soit des chances environ 4,8 fois plus élevées.
- **Un lien n'est pas une cause.** Ces coefficients décrivent ce que le modèle utilise, pas pourquoi les gens changent d'emploi.

Pour les extraire :

```python
noms = modele_final.named_steps["prep"].get_feature_names_out()
coefs = modele_final.named_steps["modele"].coef_[0]
tab = pd.DataFrame({"colonne": noms, "coef": coefs}).sort_values("coef")
```

## Ce que montre la v1

Les 20 coefficients les plus extrêmes de la v1 contiennent **18 villes** sur 20. Les plus grands sont des villes, ce qui est surprenant au premier regard.

| Ville | Coefficient | Candidats dans le train |
|---|---|---|
| `city_7` | −1,67 (le plus bas) | 20 |
| `city_39` | −1,02 | 11 |
| `city_105` | −0,98 | 65 |
| `city_155` | +1,30 (le plus haut) | 10 |
| `city_21` | +1,12 | 2 128 |

**Les extrêmes viennent de toutes petites villes.** Avec 10 ou 20 candidats, un coefficient peut être très grand par pur hasard : le modèle s'est ajusté sur quelques cas. C'est une forme de surapprentissage sur des catégories rares. Le chiffre qui l'illustre : **96 villes sur 122 ont moins de 100 candidats**, et elles ne regroupent que **18,3 %** des candidats. Le modèle a donc 96 colonnes pour peu de données. Seule `city_21`, avec 2 128 candidats, a un coefficient assez fiable pour être lu, et il est cohérent avec ses 59,1 % de la phase 2.

## Les colonnes qui ne sont pas des villes

| Colonne | Coefficient | Lecture |
|---|---|---|
| `company_size_Inconnu` | **+0,94** | l'absence de réponse sur la taille de l'entreprise pousse vers « change » |
| `education_level_Graduate` | +0,59 | |
| `education_level_Phd`, `Masters` | +0,47, +0,45 | |
| `education_level_Primary School` | −0,98 | (groupe de 308 candidats : fragile) |
| `education_level_High School` | −0,42 | |
| `city_development_index` | −0,61 | un indice plus élevé va avec moins de changement |
| `company_type_Funded Startup` | −0,47 | |
| `relevent_experience` : `Has` / `No` | −0,24 / −0,05 | écart de 0,20 seulement |
| `training_hours` | −0,03 | quasi nul |

- **`company_size_Inconnu` vient directement de la phase 3** : garder une catégorie « Inconnu » plutôt qu'imputer par la valeur la plus fréquente a conservé un signal que le modèle utilise (40,6 % de changement chez les candidats sans taille d'entreprise, contre 17,9 % pour les autres).
- **`training_hours` est quasi nul**, ce qui confirme les histogrammes qui se superposaient et les moyennes de 63,1 et 66,1 heures.
- **`relevent_experience` pèse peu** (0,20 d'écart) alors que le taux brut montrait 33,8 % contre 21,5 %. Mon hypothèse : son effet est absorbé par des colonnes qui lui sont liées, comme `company_size_Inconnu` (56,5 % des candidats sans taille d'entreprise n'ont pas d'expérience pertinente). Je ne l'ai pas vérifié.
- **`Phd` est positif (+0,47)** alors que son taux brut était de 14,0 %. Un coefficient se lit toutes choses égales par ailleurs, et dans sa colonne, Phd est proche de Masters (+0,45) et très au-dessus de Primary School. Le groupe est petit et la lecture reste fragile.

## Corriger le défaut : la v2

Puisque le problème vient des villes rares, j'ai demandé à l'encodeur de les regrouper : toute ville qui a moins de 100 candidats tombe dans une même colonne « autres villes ».

```python
preprocess_v2 = clone(preprocess).set_params(
    ville=OneHotEncoder(handle_unknown="infrequent_if_exist",
                        min_frequency=100, sparse_output=False))
v2 = Pipeline([("prep", preprocess_v2),
               ("modele", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
```

La branche `ville` passe de 122 colonnes à environ 27, et le total de **164 à 69** colonnes. Les scores, en validation croisée sur le train :

| Mesure | v1 (toutes les villes) | v2 (villes rares regroupées) | Écart | `±` de la v1 |
|---|---|---|---|---|
| Accuracy | 76,5 % | 76,3 % | −0,2 | 0,7 |
| Précision | 52,0 % | 51,7 % | −0,3 | 1,0 |
| Rappel | 74,1 % | 74,3 % | +0,2 | 1,4 |
| F1 | 61,1 % | 61,0 % | −0,1 | 0,8 |

**Les scores sont à égalité** : chaque écart est plus petit que l'écart-type des 5 plis. Mais la v2 a plus de deux fois moins de colonnes, et son tableau de coefficients est plus fiable, puisque les villes à 10 ou 20 candidats ont disparu.

Les signaux qui ne viennent pas des petites villes **n'ont presque pas bougé** : `city_21` +0,97 (contre +1,12), `company_size_Inconnu` +0,93 (+0,94), `Primary School` −1,01 (−0,98), `city_development_index` −0,63 (−0,61). Ils ne dépendent donc pas de l'encodage de `city`, et on peut les lire avec confiance.

Dans la v2, les coefficients les plus extrêmes sont :

| Vers « change » | Coefficient | Vers « ne change pas » | Coefficient |
|---|---|---|---|
| `city_21` | +0,97 | `education_level_Primary School` | −1,01 |
| `company_size_Inconnu` | +0,93 | `city_138` | −0,75 |
| `city_160` | +0,64 | `city_development_index` | −0,63 |
| `education_level_Graduate` | +0,62 | `city_65` | −0,59 |
| `city_103` | +0,62 | `city_116` | −0,58 |

## Une surprise : `city_103` est positive

`city_103` est positive dans la v2 (+0,62), alors que son taux brut en phase 2 était de 21,3 %, sous la moyenne. L'explication plausible : un coefficient se lit à indice égal, et `city_103` a un indice élevé (environ 0,92) où le taux moyen des autres villes est plus bas. Par rapport à des villes comparables, elle est donc plutôt haute. Je n'ai pas vérifié cette explication.

## Le genre dans le modèle

`gender` est une colonne sensible, donc je regarde son poids :

| Catégorie | Coefficient |
|---|---|
| Male | −0,17 |
| Inconnu | −0,14 |
| Female | −0,09 |
| Other | +0,09 |

Les coefficients sont **petits** : l'écart maximal (entre `Other` et `Male`) est de 0,26, soit des chances multipliées par environ 1,3, contre environ 5 entre les niveaux d'études extrêmes. Le genre pèse donc peu, mais il pèse : `Male` a des chances plus basses que `Female` et `Other`, ce qui va dans le sens des taux de la phase 2 (22,8 % contre 26,3 % et 26,2 %). L'effet de l'absence de réponse sur le genre (30,8 % en phase 2) est largement absorbé par `company_size_Inconnu` et d'autres colonnes, d'où un coefficient proche de zéro pour « Inconnu ». Le groupe `Other` ne compte que 191 candidats : son coefficient est fragile.

Si ce modèle servait à de vraies décisions, il faudrait se poser la question de **retirer le genre** (pour ne pas pénaliser ou favoriser un groupe), et vérifier en validation croisée que les scores ne bougent pas.

## Ce que je retiens en cinq lignes

1. Le modèle utilise des signaux vus en phase 2 : l'absence de taille d'entreprise, l'indice de développement, `city_21`, le niveau d'études.
2. La v1 avait un défaut : ses coefficients les plus extrêmes viennent de villes à 10 ou 20 candidats, donc du bruit.
3. En regroupant les villes rares (v2), les scores restent à égalité et le modèle passe de 164 à 69 colonnes.
4. `training_hours` n'apporte presque rien, et `gender` pèse peu.
5. Un coefficient se lit toutes choses égales par ailleurs, en comparant les catégories d'une même colonne.

## Les limites

- **Ce sont des liens, pas des causes.** Je décris ce que le modèle utilise, pas pourquoi les candidats changent d'emploi.
- **Une seule ville (`city_21`) pèse beaucoup**, et le modèle risque de moins bien généraliser à d'autres villes ou à d'autres périodes.
- **Les colonnes liées se répartissent l'effet** (`city` et `city_development_index`, `relevent_experience` et `company_size_Inconnu`), donc un coefficient isolé est difficile à lire.
- **La v2 n'a pas été évaluée sur le test.** Le score officiel (rappel de 78,3 %, F1 de 63,2 %) est celui de la v1. En évaluer la v2 sur le même test serait un deuxième regard sur ce jeu, à éviter. La v2 est une amélioration de simplicité, validée seulement en validation croisée.
- **Les petits groupes** (`Primary School`, `Other`) donnent des coefficients fragiles.
- **Une explication reste non vérifiée** : le rôle de la colonne `relevent_experience` et de `Phd`.

## Prochaine étape

La phase 9 : le suivi des expériences avec MLflow (enregistrer les modèles, les paramètres et les scores). Puis le rangement du code dans des fichiers `.py` (phase 10), le README final et la publication sur GitHub.