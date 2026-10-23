# Phase 3 : décider quoi faire des cases vides

Dans la phase 2, j'ai vu qu'une case vide n'a pas le même sens d'une colonne à l'autre. Cette phase sert à **décider, colonne par colonne, comment traiter les 8 colonnes qui ont des valeurs manquantes**, et à écrire pourquoi.

Dans cette phase, je ne remplis rien. Le remplissage se fera dans le pipeline, **après** le découpage entre jeu d'entraînement et jeu de test, pour éviter la fuite de données. Ici, je pose les décisions et les preuves.

## La méthode

Pour chaque colonne, je réponds à cinq questions, dans cet ordre :

1. La colonne est-elle presque vide (plus de 60 %) ? Aucune des miennes ne l'est.
2. Est-ce un nombre caché dans du texte ? (`experience`, `last_new_job`, `company_size`). Dans ce cas, je convertis avant de remplir.
3. L'écart entre le taux de changement quand la case est vide et quand elle est remplie est-il grand ? (règle de départ, pas une loi : plus d'environ 10 points, c'est un signal fort ; moins de 5, il est faible.)
4. Le groupe de cases vides est-il gros ou petit ? (environ 500 candidats comme limite de confiance.)
5. La case vide a-t-elle un sens particulier, comme « sans objet » ?

Puis je choisis un traitement dans ce menu :

| Traitement | Quand |
|---|---|
| Supprimer la colonne | presque vide |
| Médiane | colonne numérique, sans signal dans l'absence |
| Valeur la plus fréquente | colonne catégorielle, sans signal dans l'absence |
| Catégorie « Inconnu » ou « Non applicable » | catégorielle, quand l'absence porte un signal ou un sens |
| Colonne « était vide » (0 ou 1) | pour imputer et garder le signal |

## Ce que j'ai vérifié avant de décider

### Les cases vides de `company_size` et `company_type` arrivent aux mêmes candidats

| `company_size` vide ? | `company_type` vide ? | Candidats |
|---|---|---|
| non | non | 12 440 |
| **oui** | **oui** | **5 360** |
| non | oui | 780 |
| oui | non | 578 |

Parmi les 5 938 candidats sans `company_size`, **90,3 %** n'ont pas non plus de `company_type`. Parmi les 6 140 sans `company_type`, **87,3 %** n'ont pas de `company_size`. Les deux colonnes manquent donc presque toujours pour la même raison.

Ce que ces candidats ont en commun (parmi les 5 938 sans `company_size`) :

| | Sans `company_size` | Tout le jeu |
|---|---|---|
| Sans expérience pertinente | 56,5 % | 28,0 % |
| `last_new_job` = `never` | 30,0 % | 12,8 % |

Ça va dans le sens d'une hypothèse : ces candidats n'ont probablement pas d'employeur actuel à décrire. C'est un indice, pas une preuve.

### Une case vide de `major_discipline` veut surtout dire « sans objet »

Sur les 2 813 candidats sans `major_discipline` :

- **2 017** ont le niveau High School, soit **tous** les candidats de ce niveau.
- **308** ont le niveau Primary School, soit **tous** les candidats de ce niveau.
- 460 n'ont pas de niveau d'études renseigné.
- 22 sont Graduate et 6 sont Masters : ce sont les seuls vrais « trous ».

Donc 2 325 des 2 813 cases vides (82,7 %) viennent de candidats qui n'ont pas fait d'études supérieures et n'ont pas de spécialité universitaire. Remplir ces cases par la valeur la plus fréquente (`STEM`) donnerait une spécialité scientifique à des candidats de niveau lycée ou primaire, ce qui n'a pas de sens.

### Les 460 candidats sans `education_level`

Leur `major_discipline` est vide à 100 %, mais leur `enrolled_university` n'est vide que pour 100 d'entre eux. Ils ont plus de cases vides que les autres ailleurs : `company_size` 66,5 %, `company_type` 65,4 %, `gender` 43,7 %. Mon hypothèse : ce sont des profils plus incomplets en général, pas des gens qui ont sauté un volet précis.

### Pas de valeur manquante déguisée

Aucune valeur de type « N/A », « Unknown » ou tiret dans les colonnes de texte. Certaines valeurs ressemblent à une absence mais sont de vraies réponses : `no_enrollment`, `No Major`, `never`, `Other`. `experience` contient 22 valeurs (1 à 20, `<1`, `>20`) et toutes les villes commencent par `city_`.

## Le tableau des décisions

| Colonne | Cases vides | Taux si vide / si rempli | Pourquoi ça manque | Décision | Justification |
|---|---|---|---|---|---|
| `company_size` | 31,0 % (5 938) | 40,6 % / 17,9 % | Hypothèse : pas d'entreprise actuelle (90,3 % n'ont pas non plus de `company_type`, 56,5 % sans expérience pertinente) | Catégorie « Inconnu », colonne gardée en catégories | Écart de 22,7 points sur un gros groupe : l'absence est un signal fort. La tranche la plus fréquente le noierait. |
| `company_type` | 32,0 % (6 140) | 38,8 % / 18,4 % | Même cause que `company_size` (87,3 % de recoupement). Pour les 780 cas où seul le type manque : oubli ou refus (hypothèse). | Catégorie « Inconnu » | Écart de 20,4 points sur 6 140 candidats. |
| `major_discipline` | 14,7 % (2 813) | 19,5 % / 25,9 % | Fait observé : 82,7 % des cases vides sont des candidats de niveau lycée ou primaire, donc « sans objet ». | Catégorie « Non applicable » | Imputer `STEM` inventerait une spécialité à des candidats qui n'en ont pas. |
| `education_level` | 2,4 % (460) | 22,6 % / 25,0 % | Profils plus incomplets en général (hypothèse). | **À trancher** : valeur la plus fréquente (`Graduate`) ou « Inconnu » | Écart de 2,4 points : peu de signal. Mais `Graduate` donnerait un diplômé sans spécialité pour ces 460 candidats (dont `major_discipline` sera « Non applicable »). Une catégorie « Inconnu » est plus cohérente. |
| `gender` | 23,5 % (4 508) | 30,8 % / 23,1 % | Hypothèse : information personnelle que certains candidats ne veulent pas indiquer. | Catégorie « Inconnu » | Écart de 7,7 points sur 4 508 candidats. Colonne sensible : je ne devine pas la valeur. |
| `last_new_job` | 2,2 % (423) | 36,4 % / 24,7 % | Hypothèse : historique non renseigné ou oubli (le vide est distinct de `never`). | Conversion (`never` en 0, `>4` en 5), puis médiane | Écart de 11,7 points mais petit groupe : j'impute simplement. |
| `enrolled_university` | 2,0 % (386) | 31,9 % / 24,8 % | Hypothèse : oubli ou refus, sans motif visible. | Valeur la plus fréquente (`no_enrollment`) | Petit groupe, écart de 7,1 points fragile. Colonne catégorielle, donc pas de médiane. |
| `experience` | 0,3 % (65) | 35,4 % / 24,9 % | Hypothèse : oubli ou erreur de saisie. | Conversion (`<1` en 0, `>20` en 21), puis médiane | Groupe minuscule : l'écart de 10,5 points n'est pas fiable. |

## L'ordre à respecter dans le pipeline

1. **Découper** train et test (phase 4).
2. **Convertir** les nombres cachés : `experience`, `last_new_job`, et les tranches de `company_size`. Les bornes sont des approximations (je ne connais pas la vraie valeur de `>20`), à noter dans les limites. Le format des tranches de `company_size` est incohérent (`10/49` avec une barre oblique, les autres avec un tiret).
3. **Imputer** : la médiane et la valeur la plus fréquente sont calculées sur le **jeu d'entraînement seulement**, puis appliquées au test.
4. **Encoder** les catégories pour le modèle.

## Ce que je retiens en cinq lignes

1. Les cases vides ne se traitent pas toutes pareil : l'écart entre le taux « si vide » et « si rempli » décide.
2. `company_size` et `company_type` manquent ensemble (87 à 90 % de recoupement) : une même cause probable, des candidats sans employeur actuel.
3. Pour `major_discipline`, le vide veut dire « sans objet » dans 82,7 % des cas : on ne devine pas une spécialité.
4. Sur les petits groupes (65 à 423 candidats), j'impute simplement, car l'écart est peu fiable.
5. Aucune décision n'est appliquée ici : tout se fera dans le pipeline, après le découpage.

## Les limites

- **Mes raisons sont des hypothèses.** Les données montrent des liens (par exemple 56,5 % des candidats sans `company_size` n'ont pas d'expérience pertinente), mais pas la cause.
- **Le signal des cases vides recoupe d'autres colonnes** : `relevent_experience` porte déjà un signal fort. Je verrai à la modélisation si les cases vides apportent quelque chose en plus.
- **Les seuils (10 points, 500 candidats) sont des règles de départ**, pas des tests statistiques.
- **Les conversions des bornes** (`>20` en 21, `<1` en 0, `>4` en 5, `never` en 0) sont des approximations.
- **`gender` est sensible** : traiter les cases vides comme « Inconnu » évite de deviner, mais le biais possible du modèle reste à discuter.

## Le code utilisé

```python
# Deux colonnes manquent-elles ensemble ?
df[["company_size", "company_type"]].isna().value_counts()

# D'où viennent les cases vides de major_discipline ?
df[df["major_discipline"].isna()]["education_level"].value_counts(dropna=False)

# Que ont en commun les candidats sans company_size ?
df[df["company_size"].isna()]["relevent_experience"].value_counts(dropna=False)
df[df["company_size"].isna()]["last_new_job"].value_counts(dropna=False)

# Les candidats sans education_level ont-ils d'autres cases vides ?
df[df["education_level"].isna()].isna().sum()

# Valeurs déguisées ?
for col in df.select_dtypes(exclude="number").columns:
    print(col)
    print(df[col].value_counts(dropna=False).head(10))

# Essai sur une copie, jamais sur df
df2 = df.copy()
df2["company_size"] = df2["company_size"].fillna("Inconnu")
df2["company_size_vide"] = df["company_size"].isna().astype(int)
```

## Questions d'entretien sur cette phase

**Pourquoi ne pas remplir toutes les cases vides par la valeur la plus fréquente ?**
Parce qu'une case vide ne veut pas dire la même chose partout. Pour `company_size`, 40,6 % des candidats qui n'ont pas répondu cherchent à changer, contre 17,9 % de ceux qui ont répondu : remplir effacerait ce signal.

**Comment décides-tu du traitement d'une colonne ?**
Je regarde l'écart entre le taux si vide et le taux si rempli, la taille du groupe, le type de la colonne (nombre ou catégorie) et le sens possible de l'absence.

**Qu'as-tu trouvé de plus intéressant sur les valeurs manquantes ?**
Que les cases vides de `major_discipline` sont surtout des candidats de niveau lycée ou primaire (82,7 %) : elles veulent dire « sans objet », pas « oubli ».

**Pourquoi imputer après le découpage train/test ?**
Pour que la médiane ou la valeur la plus fréquente soit calculée sans voir le jeu de test, sinon c'est de la fuite de données et les résultats seraient trop optimistes.

**Comment traites-tu une colonne sensible comme `gender` ?**
Je ne devine pas la valeur : je crée une catégorie « Inconnu » et je note le risque de biais dans les limites.

## Prochaine étape

La phase 4 : découper les données en jeu d'entraînement et jeu de test (80 % et 20 %, stratifié sur la cible), avant toute modification. Les décisions de ce tableau serviront ensuite à construire le pipeline (phase 5).