# Phase 2 : explorer les données — guide complet

Ce document sert à te **guider** pendant la phase, à **documenter** ce que j'ai trouvé pour quelqu'un qui consulte le projet, et à te **préparer à l'entretien**. Il répond à toutes les questions de la phase 2 du questionnaire (questions 16 à 30, section 8).

Règle de la phase : on compare des groupes de candidats. **Aucune donnée n'est modifiée.**

## Sommaire

1. Objectif et règle du jeu
2. Le vocabulaire à maîtriser
3. Les outils : le code expliqué
4. Résultats, colonne par colonne
5. Les histogrammes : comment les lire
6. Quand une case vide est un signal
7. Corrélation et cause : pourquoi la prudence
8. Questions et réponses (Q16 à Q30)
9. Les erreurs que j'ai déjà faites
10. La fiche de chiffres
11. Les limites de cette phase

---

## 1. Objectif et règle du jeu

Je cherche à voir **quelles caractéristiques semblent liées au fait de chercher un nouvel emploi**, et à repérer où les cases vides portent de l'information.

Dans le jeu de données, **24,9 % des candidats cherchent à changer d'emploi**. Ce chiffre est la **référence** : tout groupe se compare à lui.

La règle : pour chaque groupe, je regarde toujours **deux choses**, le **taux** et l'**effectif**.

- Un taux sans référence ne dit rien : « 33,8 % » est élevé parce que la moyenne est à 24,9 %.
- Un taux sans effectif ne vaut rien : 60 % sur 20 candidats n'a pas la même valeur que 60 % sur 5 000.

> Taux sans effectif : une anecdote. Taux avec effectif et référence : une observation.

---

## 2. Le vocabulaire à maîtriser

| Mot | Définition simple | Exemple dans ce projet |
|---|---|---|
| **Taux** | la part de candidats d'un groupe qui cherchent à changer (la moyenne de `target` dans ce groupe) | 33,8 % chez les candidats sans expérience pertinente |
| **Référence** | le taux sur tout le jeu de données | 24,9 % |
| **Effectif** | le nombre de candidats d'un groupe | 5 366 |
| **Écart en points** | la différence entre un taux et la référence, en points (pas en %) | 33,8 − 24,9 = 8,9 points |
| **Groupe fragile** | un groupe trop petit pour que son taux soit fiable | Phd : 414 candidats |
| **Signal** | une information utile pour prédire | un écart fort sur un gros groupe |
| **Tranche** | un intervalle de valeurs qu'on regroupe | indice entre 0,6 et 0,65 |
| **Distribution** | comment les valeurs d'une colonne se répartissent | beaucoup de petites valeurs, peu de grandes |
| **Pic** | une zone où beaucoup de candidats ont la même valeur | autour de 0,62 pour l'indice de développement |
| **Étalée vers la droite** | beaucoup de petites valeurs, et une longue traîne vers les grandes | `training_hours` |
| **Corrélation / lien** | deux choses qui varient ensemble | taux élevé et ville `city_21` |
| **Cause** | une chose qui en provoque une autre | ce que les données ne démontrent pas |

**Points ou pourcents ?** L'écart entre deux taux se dit en **points**. Passer de 24,9 % à 33,8 % est un écart de 8,9 points, pas de 8,9 %.

---

## 3. Les outils : le code expliqué

### 3.1 La référence

```python
df["target"].mean()                 # 0.2493
df["target"].sum() / len(df)        # la même chose : 4777 / 19158
```

`target` ne contient que des 0 et des 1. La somme est le nombre de 1 (4 777). En divisant par le nombre de lignes (19 158), on obtient la **part** de 1 : 0,2493, soit 24,9 %. Le chiffre 19 158, lui, est un effectif, pas une référence.

### 3.2 Le taux et l'effectif par groupe

```python
(df.groupby("education_level", dropna=False)["target"]
   .agg(["mean", "count"])
   .sort_values("mean", ascending=False))
```

Ligne par ligne :
- `groupby("education_level")` crée un groupe par niveau d'études.
- `["target"]` regarde la colonne `target` dans chaque groupe.
- `.agg(["mean", "count"])` calcule, pour chaque groupe, la **moyenne** (le taux) et le **nombre de lignes** (l'effectif).
- `.sort_values("mean", ascending=False)` trie du taux le plus haut au plus bas.
- **`dropna=False`** garde les cases vides comme **un groupe à part** (la ligne `NaN`). Sans lui, pandas ignorerait ces candidats, et on ne verrait pas que, par exemple, les candidats sans `company_size` sont à 40,6 %.

Pour plusieurs colonnes :

```python
for col in ["relevent_experience", "enrolled_university", "company_size", "gender"]:
    print(col)
    print(df.groupby(col, dropna=False)["target"].agg(["mean", "count"]))
    print()
```

### 3.3 Découper une colonne numérique en tranches

```python
tranches = pd.cut(df["city_development_index"], bins=[0, 0.6, 0.65, 0.8, 0.9, 1.0])
df.groupby(tranches, observed=True)["target"].agg(["mean", "count"])
```

- `pd.cut` découpe une colonne **numérique** en tranches selon les bornes `bins`.
- La notation `(0.6, 0.65]` veut dire : 0,6 **exclu**, 0,65 **inclus**.
- On le fait parce que `city_development_index` a 93 valeurs différentes : chacune a trop peu de candidats pour calculer un taux. En regroupant en tranches, on obtient des groupes assez gros pour comparer.

### 3.4 Comparer deux distributions

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(data=df, x="training_hours", hue="target",
             stat="density", common_norm=False)
plt.show()
```

- `x=` est la colonne étudiée.
- `hue="target"` trace **une courbe par groupe** (cherche à changer, ou non). Il faut mettre `target` ici, pas la colonne étudiée.
- `stat="density"` met chaque courbe à l'échelle, pour comparer des **formes** et non des nombres de candidats.
- `common_norm=False` normalise **chaque groupe séparément**. Sans lui, le groupe des 14 381 candidats qui ne changent pas écraserait celui des 4 777 qui changent.

### 3.5 Le taux quand une colonne est vide, contre quand elle est remplie

```python
rows = []
for col in df.columns[df.isna().any()]:
    vide = df[col].isna()
    rows.append({
        "colonne": col,
        "n_vides": vide.sum(),
        "taux_si_vide": df.loc[vide, "target"].mean(),
        "taux_si_rempli": df.loc[~vide, "target"].mean(),
    })
pd.DataFrame(rows).round(3)
```

- `vide` vaut `True` pour les candidats dont la colonne est vide.
- `df.loc[vide, "target"]` garde la cible de ces candidats seulement.
- `~vide` veut dire « le contraire de vide » : ceux qui ont répondu.

---

## 4. Résultats, colonne par colonne

Les écarts sont calculés avec les taux arrondis (précision de ±0,1 point). Référence : **24,9 %**.

### 4.1 Expérience pertinente (`relevent_experience`)

| Groupe | Taux | Écart | Effectif |
|---|---|---|---|
| Sans expérience pertinente | 33,8 % | +8,9 points | 5 366 |
| Avec expérience pertinente | 21,5 % | −3,4 points | 13 792 |

Un écart net sur de gros groupes (5 366 + 13 792 = 19 158). Les candidats sans expérience pertinente cherchent plus souvent à changer.

### 4.2 Études en cours (`enrolled_university`)

| Groupe | Taux | Écart | Effectif |
|---|---|---|---|
| Cours à temps plein | 38,1 % | +13,2 points | 3 757 |
| Case vide | 31,9 % | +7,0 points | 386 |
| Cours à temps partiel | 25,2 % | +0,3 point | 1 198 |
| Pas inscrit | 21,1 % | −3,8 points | 13 817 |

Le temps plein a le **taux le plus élevé** de toutes les catégories que j'ai regardées, sur un groupe de bonne taille. Les « pas inscrits » pèsent 72 % du jeu de données : leur taux est donc fiable et proche de la moyenne.

### 4.3 Niveau d'études (`education_level`)

| Groupe | Taux | Écart | Effectif |
|---|---|---|---|
| Graduate | 28,0 % | +3,1 points | 11 598 |
| Case vide | 22,6 % | −2,3 points | 460 |
| Masters | 21,4 % | −3,5 points | 4 361 |
| High School | 19,5 % | −5,4 points | 2 017 |
| Phd | 14,0 % | −10,9 points | 414 |
| Primary School | 13,3 % | −11,6 points | 308 |

Il n'y a pas de tendance simple du type « plus on est diplômé, plus on change » : les Graduate sont les plus hauts et les deux plus bas (Phd et Primary School) sont aux extrémités. Ces deux groupes sont petits (722 candidats à eux deux), donc moins fiables. Les candidats sans niveau renseigné (22,6 %) se distinguent peu des autres.

### 4.4 Genre (`gender`)

| Groupe | Taux | Écart | Effectif |
|---|---|---|---|
| Case vide | 30,8 % | +5,9 points | 4 508 |
| Female | 26,3 % | +1,4 point | 1 238 |
| Other | 26,2 % | +1,3 point | 191 |
| Male | 22,8 % | −2,1 points | 13 221 |

Entre Female, Male et Other, l'écart est faible. `Other` ne compte que 191 candidats : son taux est peu fiable. Colonne sensible (voir Q27).

### 4.5 Taille de l'entreprise (`company_size`)

| Tranche | Taux | Effectif |
|---|---|---|
| `<10` | 17,1 % | 1 308 |
| `10/49` | 23,4 % | 1 471 |
| `50-99` | 17,7 % | 3 083 |
| `100-500` | 16,1 % | 2 571 |
| `500-999` | 17,3 % | 877 |
| `1000-4999` | 15,1 % | 1 328 |
| `5000-9999` | 18,1 % | 563 |
| `10000+` | 19,1 % | 2 019 |
| **Case vide** | **40,6 %** | **5 938** |

Toutes les tranches renseignées sont sous la moyenne (entre 15 % et 23 %). Les 5 938 candidats sans réponse sont à 40,6 %, soit 15,7 points au-dessus de la référence. Les tranches sont triées par ordre alphabétique dans la sortie, pas par taille.

### 4.6 Indice de développement de la ville (`city_development_index`)

| Tranche | Taux | Écart | Effectif |
|---|---|---|---|
| (0, 0,6] | 57,9 % | +33,0 points | 725 |
| (0,6, 0,65] | 58,3 % | +33,4 points | 2 754 |
| (0,65, 0,8] | 26,0 % | +1,1 point | 2 487 |
| (0,8, 0,9] | 13,6 % | −11,3 points | 3 549 |
| (0,9, 1,0] | 16,8 % | −8,1 points | 9 643 |

La tendance générale est « indice bas, taux élevé », mais elle n'est pas parfaitement régulière : la dernière tranche est un peu au-dessus de l'avant-dernière. Les deux tranches basses sont à plus du double de la référence.

### 4.7 Les villes derrière les deux pics

L'histogramme de l'indice a **deux pics**, chacun dû à **une seule ville** :

| Pic | Ville | Effectif | Part du jeu | Taux | Écart |
|---|---|---|---|---|---|
| autour de 0,62 | `city_21` | 2 702 | 14,1 % | 59,1 % | +34,2 points |
| autour de 0,92 | `city_103` | 4 355 | 22,7 % | 21,3 % | −3,6 points |

Comment je l'ai trouvé : l'histogramme montre où est le pic, puis un filtre donne la ville.

```python
df[df["city_development_index"].between(0.62, 0.63)]["city"].value_counts().head()
# city_21 : 2702, city_129 : 3
```

Sur 2 705 candidats dont l'indice est entre 0,62 et 0,63, **2 702 viennent de `city_21`** (99,9 %). Et `city_21` compte 2 702 candidats en tout : tous ses candidats ont donc le même indice. De même, 98,1 % des candidats de la tranche (0,6, 0,65] viennent de `city_21` (2 702 sur 2 754).

Les cinq plus grosses villes : `city_103` (4 355), `city_21` (2 702), `city_16` (1 533), `city_114` (1 336), `city_160` (845). Elles regroupent 10 771 candidats, soit 56 % du jeu. `city_21` et `city_103` à elles deux : 7 057 candidats, environ 37 %.

**Nuance :** le lien « indice bas, taux élevé » ne vient pas *uniquement* de `city_21`. La tranche (0, 0,6] (725 candidats, 57,9 %) ne contient pas `city_21` et a pourtant un taux aussi élevé. Mais le pic de 0,62, lui, vient presque entièrement de cette ville.

### 4.8 Heures de formation (`training_hours`)

Les deux courbes (ceux qui cherchent à changer, ceux qui ne cherchent pas) **se superposent presque**. La distribution est **étalée vers la droite** : un pic autour de 20 à 30 heures, puis une longue traîne jusqu'à environ 340 heures. Cette colonne **distingue peu** les deux groupes.

À vérifier : `df.groupby("target")["training_hours"].mean()` pour confirmer que les deux moyennes sont proches.

---

## 5. Les histogrammes : comment les lire

Pour chaque histogramme avec `hue="target"`, trois questions dans l'ordre :

1. **Les deux courbes se ressemblent-elles ou sont-elles décalées ?**
2. **Si elles sont décalées, de quel côté se trouve le groupe qui cherche à changer ?** (valeurs plus basses ou plus hautes)
3. **L'écart est-il grand ou petit ?**

| Ce que je vois | Ce que je dis |
|---|---|
| Courbes superposées (`training_hours`) | la colonne distingue peu les deux groupes |
| Courbes décalées (`city_development_index`) | la colonne aide à distinguer les groupes |
| Un pic haut | beaucoup de candidats ont la même valeur : chercher pourquoi |
| Courbe qui descend vers la droite | distribution étalée vers la droite |

**Sur l'histogramme de l'indice :** autour de 0,62, la courbe de ceux qui cherchent à changer monte à environ 25, contre environ 6 pour les autres. Autour de 0,92, c'est l'inverse : environ 30 pour ceux qui ne changent pas, 21 pour les autres. Les deux courbes sont **nettement décalées**.

Les courbes en densité montrent **comment chaque groupe se répartit**, pas directement le taux dans chaque zone. Pour obtenir le taux par zone, on utilise `pd.cut` et `groupby`.

---

## 6. Quand une case vide est un signal

J'ai comparé, pour chaque colonne qui a des cases vides, le taux quand la case est vide et quand elle est remplie.

| Colonne | Cases vides | Taux si vide | Taux si rempli | Écart |
|---|---|---|---|---|
| `company_size` | 5 938 | 40,6 % | 17,9 % | +22,7 points |
| `company_type` | 6 140 | 38,8 % | 18,4 % | +20,4 points |
| `last_new_job` | 423 | 36,4 % | 24,7 % | +11,7 points |
| `experience` | 65 | 35,4 % | 24,9 % | +10,5 points |
| `gender` | 4 508 | 30,8 % | 23,1 % | +7,7 points |
| `enrolled_university` | 386 | 31,9 % | 24,8 % | +7,1 points |
| `major_discipline` | 2 813 | 19,5 % | 25,9 % | −6,4 points |
| `education_level` | 460 | 22,6 % | 25,0 % | −2,4 points |

- **Gros écart sur un gros groupe** (`company_size`, `company_type`) : ne pas répondre est un signal fort. Une imputation par la valeur la plus fréquente l'effacerait.
- **Écart faible** (`education_level`) : une imputation simple suffira probablement.
- **Écart qui va dans l'autre sens** (`major_discipline`) : une case vide ne veut pas toujours dire « plus de changement ».
- **Petits groupes** (`experience` : 65, `enrolled_university` : 386, `last_new_job` : 423) : les écarts sont moins fiables.

C'est le point de départ de la phase 3.

---

## 7. Corrélation et cause : pourquoi la prudence

Une **corrélation** (ou un lien) veut dire que deux choses varient ensemble : par exemple, être de la ville `city_21` va avec un taux de 59,1 %. Une **cause** voudrait dire que l'une provoque l'autre.

Les données montrent le lien, pas la raison. Pour `city_21`, plusieurs explications sont possibles : un secteur d'activité local, la façon dont les données ont été collectées, un type de formation particulier. Aucune n'est démontrée par ces chiffres.

- À dire : « les candidats de `city_21` ont plus tendance à chercher un nouvel emploi ».
- À ne pas dire : « vivre à `city_21` pousse à changer d'emploi ».

Je n'ai fait aucun test statistique : je parle d'écarts « nets » ou « marqués », pas « prouvés ».

---

## 8. Questions et réponses (phase 2 du questionnaire)

**Q16. Quelle est la référence de l'exploration, et comment l'as-tu calculée ?**
24,9 %, la part de candidats qui cherchent à changer d'emploi : 4 777 ÷ 19 158 = 0,2493. En code : `df["target"].mean()`.

**Q17. Taux et effectif des candidats sans expérience pertinente et de ceux qui en ont.**
Sans : 33,8 % (5 366 candidats), soit 8,9 points au-dessus de la référence. Avec : 21,5 % (13 792), soit 3,4 points en dessous. Les deux effectifs font 19 158.

**Q18. Taux et effectif des candidats à temps plein et de ceux qui ne sont pas inscrits.**
Temps plein : 38,1 % (3 757), soit 13,2 points au-dessus. Pas inscrits : 21,1 % (13 817), soit 3,8 points en dessous.

**Q19. Quelle ville explique le pic de l'indice, avec quel effectif et quel taux ?**
Deux pics, chacun dû à une ville. Autour de 0,62 : `city_21`, 2 702 candidats, 59,1 % (34,2 points au-dessus de la référence). Autour de 0,92 : `city_103`, 4 355 candidats, 21,3 % (3,6 points en dessous).

**Q20. Taux de changement quand `company_size` est vide, et quand elle est remplie.**
Vide : 40,6 % (5 938 candidats), soit 15,7 points au-dessus de la référence. Remplie : 17,9 % (13 220), soit 7,0 points en dessous. L'absence de réponse est un signal fort.

**Q21. Pourquoi comparer chaque groupe à la référence, et pourquoi regarder l'effectif ?**
Sans la référence, un taux ne dit rien : 33,8 % n'est « élevé » que parce que la moyenne est à 24,9 %. Sans l'effectif, on ne sait pas si le taux est fiable : 38,1 % sur 3 757 candidats (temps plein) est crédible, alors que 14,0 % sur 414 candidats (Phd) l'est beaucoup moins.

**Q22. Que montre l'indice découpé en tranches, et pourquoi le lien vient-il surtout d'une seule ville ?**
On le découpe parce que c'est une variable numérique continue (93 valeurs différentes) : chaque valeur a trop peu de candidats pour calculer un taux, alors qu'une tranche donne un groupe assez gros. Les tranches montrent « indice bas, taux élevé » : environ 58 % pour les deux tranches basses, contre 13,6 % et 16,8 % pour les hautes. Le pic de 0,62 vient presque entièrement de `city_21` (2 702 des 2 754 candidats de la tranche). Mais la tranche (0, 0,6] (725 candidats, 57,9 %) montre que le lien ne vient pas uniquement de cette ville.

**Q23. Que montrent les histogrammes de `training_hours` ?**
Les deux courbes se superposent presque : la colonne distingue peu ceux qui cherchent à changer de ceux qui ne cherchent pas. La distribution est étalée vers la droite : beaucoup de candidats ont peu d'heures (pic autour de 20 à 30 heures), quelques-uns en ont beaucoup (jusqu'à environ 340). Elle apportera probablement peu d'information au modèle, à vérifier plus tard.

**Q24. Pourquoi une corrélation n'est-elle pas une cause ? Exemple de ce projet.**
Une corrélation est un lien observé entre deux choses ; une cause demande de démontrer que l'une provoque l'autre. Exemple : les candidats de `city_21` sont à 59,1 %, mais les données ne disent pas pourquoi (secteur d'activité, collecte des données…). Je dis « ont plus tendance à », pas « poussent à ».

**Q25. Pourquoi comparer le taux quand une colonne est vide et quand elle est remplie ? Qu'est-ce que ça change ?**
Pour savoir si ne pas répondre porte de l'information. Si l'écart est grand sur un gros groupe (`company_size` : 40,6 % contre 17,9 %), je garde l'information, par exemple avec une catégorie « Inconnu ». Si l'écart est faible (`education_level` : 22,6 % contre 25,0 %), une imputation simple suffit.

**Q26. Pourquoi les petits groupes sont-ils fragiles ? Exemple.**
Parce que quelques candidats peuvent suffire à déplacer le taux. Par exemple, Phd (414 candidats), Primary School (308) et `Other` dans `gender` (191) sont de petits groupes : leurs taux sont intéressants mais moins fiables que ceux de groupes de plusieurs milliers. Pareil pour les 65 cases vides de `experience`.

**Q27. Pourquoi `gender` demande-t-elle une précaution ?**
C'est une information personnelle et sensible. Elle a 23,5 % de cases vides (4 508 candidats), et les candidats qui n'ont pas répondu sont à 30,8 %. Je ne devine pas la valeur : je prévois une catégorie « Inconnu ». Je décris des liens observés, jamais des causes, et je note le risque de biais d'un modèle dans les limites du projet.

**Q28. Que fait `df.groupby("education_level", dropna=False)["target"].agg(["mean", "count"])` ? Que change `dropna=False` ?**
Elle forme un groupe par niveau d'études, puis calcule pour chacun la moyenne de `target` (le taux de changement) et le nombre de candidats (l'effectif). `dropna=False` garde les cases vides comme un groupe à part ; sans lui, ces candidats seraient ignorés.

**Q29. À quoi sert `pd.cut` ? Dans quel cas l'as-tu utilisé ?**
Elle découpe une colonne numérique en tranches. Je l'ai utilisée sur `city_development_index` avec les bornes 0, 0,6, 0,65, 0,8, 0,9 et 1,0, pour obtenir des groupes assez gros et comparer leur taux. La notation `(0.6, 0.65]` exclut 0,6 et inclut 0,65.

**Q30. Dans un histogramme avec `hue="target"`, à quoi servent `stat="density"` et `common_norm=False` ?**
`stat="density"` met chaque courbe à l'échelle pour comparer les formes. `common_norm=False` normalise chaque groupe séparément : sans lui, le groupe des 14 381 candidats qui ne changent pas écraserait celui des 4 777 qui changent.

---

## 9. Les erreurs que j'ai déjà faites (à éviter)

- **Confondre la référence et un effectif :** la référence est 24,9 %, pas 19 158.
- **Additionner deux taux** pour « combiner » deux groupes (j'avais trouvé 37 % pour Phd et Primary School). Il faut calculer le taux sur les candidats concernés (environ 13,7 %).
- **Lire une courbe à l'envers :** `training_hours` n'est pas « ascendante », elle descend vers la droite.
- **Décrire la forme sans comparer les deux groupes.**
- **Tronquer au lieu d'arrondir :** 21,4 % au lieu de 21,5 %, 38 % au lieu de 38,1 %, 40,5 % au lieu de 40,6 %.
- **Se tromper dans une soustraction :** 40,6 − 24,9 = 15,7, pas 15,9. Écrire le calcul avant le résultat.
- **Oublier de comparer à la référence** quand on donne un taux.
- **Confondre les deux pics** : `city_21` pour le pic de 0,62, `city_103` pour celui de 0,92.
- **Donner un chiffre de mémoire** (45,4 %, 522, 15,3 %…) sans relancer le code.
- **Dire « l'indice est découpé car il est catégoriel » :** il est numérique ; on le découpe pour avoir des groupes assez gros.

---

## 10. La fiche de chiffres de la phase 2

| | |
|---|---|
| Référence | 24,9 % (4 777 ÷ 19 158) |
| Sans expérience pertinente | 33,8 % (5 366) |
| Avec expérience pertinente | 21,5 % (13 792) |
| Cours à temps plein | 38,1 % (3 757) |
| Pas inscrits | 21,1 % (13 817) |
| `city_21` | 59,1 % (2 702) |
| `city_103` | 21,3 % (4 355) |
| `company_size` vide / remplie | 40,6 % (5 938) / 17,9 % (13 220) |
| `company_type` vide / remplie | 38,8 % (6 140) / 18,4 % |
| Phd / Primary School | 14,0 % (414) / 13,3 % (308) |

---

## 11. Les limites de cette phase

- **Un lien n'est pas une cause.**
- **Aucun test statistique** : je parle d'écarts nets, pas prouvés.
- **Les petits groupes sont fragiles** : Phd, Primary School, `Other`, et les cases vides de `experience`, `enrolled_university` et `last_new_job`.
- **Une ville pèse beaucoup** : à la modélisation, je regarderai si le modèle s'appuie surtout sur `city_21`.
- **Les écarts sont calculés avec des taux arrondis** (précision de ±0,1 point).
- **Les données sont publiques**, pas celles d'une entreprise réelle.
- **`training_hours`** : je n'ai pas vérifié les moyennes des deux groupes, seulement les histogrammes.

## Prochaine étape

La phase 3 : décider, colonne par colonne, quoi faire des cases vides. Voir `phase3.md`.