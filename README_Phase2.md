# Phase 2 : ce que disent les données

Avant de nettoyer ou de modéliser quoi que ce soit, j'ai regardé comment la cible (`target`) varie selon les colonnes. Cette page garde une trace de ce que j'ai trouvé, de comment je l'ai lu, et de ce que j'en retiens pour la suite.

Aucune donnée n'est modifiée dans cette phase : on regarde, on compte, on compare.

## La règle du jeu

Dans le jeu de données, **24,9 % des candidats cherchent à changer d'emploi**. Ce chiffre sert de référence pour tout le reste.

Un taux seul ne dit rien. « 33,8 % » ne veut rien dire tant qu'on ne sait pas que la moyenne est à 24,9 %. Et un taux ne vaut rien non plus sans la taille du groupe : 14 % sur 414 candidats est bien moins solide que 14 % sur 4 000.

Donc pour chaque groupe, je regarde toujours deux choses : **le taux** et **l'effectif**.

> Taux sans effectif = anecdote. Taux avec effectif = observation.

## Ce que j'ai trouvé

### Niveau d'études (`education_level`)

| Groupe | Taux | Effectif |
|---|---|---|
| Graduate | 28,0 % | 11 598 |
| Case vide | 22,6 % | 460 |
| Masters | 21,4 % | 4 361 |
| High School | 19,5 % | 2 017 |
| Phd | 14,0 % | 414 |
| Primary School | 13,3 % | 308 |

Pas de tendance simple du genre « plus on est diplômé, plus on change » : les Graduate sont les plus hauts, et les deux plus bas (Phd et Primary School) sont aux deux extrémités. Ces deux derniers groupes sont petits (722 candidats à eux deux), donc je reste prudent.

**À retenir :** les candidats sans niveau d'études renseigné (22,6 %) ne se distinguent presque pas des autres.

### Expérience pertinente (`relevent_experience`)

| Groupe | Taux | Effectif |
|---|---|---|
| Sans expérience pertinente | 33,8 % | 5 366 |
| Avec expérience pertinente | 21,5 % | 13 792 |

Un écart net sur de gros groupes. Les candidats sans expérience pertinente cherchent plus souvent à changer.

### Études en cours (`enrolled_university`)

| Groupe | Taux | Effectif |
|---|---|---|
| Cours à temps plein | 38,1 % | 3 757 |
| Case vide | 31,9 % | 386 |
| Cours à temps partiel | 25,2 % | 1 198 |
| Pas inscrit | 21,1 % | 13 817 |

C'est le taux le plus élevé de toutes les catégories que j'ai regardées, sur un groupe de bonne taille.

### Genre (`gender`)

| Groupe | Taux | Effectif |
|---|---|---|
| Case vide | 30,8 % | 4 508 |
| Female | 26,3 % | 1 238 |
| Other | 26,2 % | 191 |
| Male | 22,8 % | 13 221 |

Entre Female, Male et Other, l'écart est faible (quelques points). `Other` ne compte que 191 candidats, donc son taux est peu fiable. C'est une colonne sensible : je décris un lien observé, sans en tirer de cause.

### Taille de l'entreprise (`company_size`)

Toutes les tranches renseignées sont sous la moyenne (entre 15 % et 23 %). Mais les **5 938 candidats qui n'ont rien indiqué** sont à **40,6 %**. On y revient plus bas.

### Indice de développement de la ville (`city_development_index`)

| Tranche | Taux | Effectif |
|---|---|---|
| (0, 0,6] | 57,9 % | 725 |
| (0,6, 0,65] | 58,3 % | 2 754 |
| (0,65, 0,8] | 26,0 % | 2 487 |
| (0,8, 0,9] | 13,6 % | 3 549 |
| (0,9, 1,0] | 16,8 % | 9 643 |

Les deux tranches basses sont à plus du double de la moyenne. La tendance générale est « indice bas, taux élevé », mais elle n'est pas parfaitement régulière (la dernière tranche est un peu au-dessus de l'avant-dernière).

Sur l'histogramme, deux pics ressortent, vers 0,62 et vers 0,92. Le pic de 0,62 vient presque entièrement d'**une seule ville, `city_21`** : 2 702 candidats (14 % du jeu de données) avec un taux de **59,1 %**. Le lien entre indice bas et taux élevé vient donc en grande partie de cette ville.

Les cinq plus grosses villes (`city_103`, `city_21`, `city_16`, `city_114`, `city_160`) regroupent 10 771 candidats, soit 56 % du jeu.

### Heures de formation (`training_hours`)

Les deux courbes (ceux qui cherchent à changer, ceux qui ne cherchent pas) se superposent presque. La distribution est étalée vers la droite : beaucoup de candidats avec peu d'heures, quelques-uns avec beaucoup (jusqu'à environ 340). Cette colonne distingue peu les deux groupes.

## Le point qui change tout : une case vide, ça veut dire quelque chose

J'ai comparé, pour chaque colonne qui a des cases vides, le taux quand la case est vide et le taux quand elle est remplie.

| Colonne | Cases vides | Taux si vide | Taux si rempli |
|---|---|---|---|
| `company_size` | 5 938 | 40,6 % | 17,9 % |
| `company_type` | 6 140 | 38,8 % | 18,4 % |
| `last_new_job` | 423 | 36,4 % | 24,7 % |
| `experience` | 65 | 35,4 % | 24,9 % |
| `gender` | 4 508 | 30,8 % | 23,1 % |
| `enrolled_university` | 386 | 31,9 % | 24,8 % |
| `major_discipline` | 2 813 | 19,5 % | 25,9 % |
| `education_level` | 460 | 22,6 % | 25,0 % |

Ce que ça montre :

- **`company_size` et `company_type`** : de gros écarts sur de gros groupes. Ne pas répondre est ici un signal fort. Les candidats qui n'ont pas renseigné leur entreprise cherchent environ deux fois plus souvent à changer.
- **`major_discipline`** : l'écart va dans l'autre sens (taux plus bas quand c'est vide). Une case vide ne veut donc pas toujours dire « plus de changement ».
- **`education_level`** : presque aucun écart. Là, une imputation simple suffira probablement.
- **`experience`, `enrolled_university`, `last_new_job`** : les groupes vides sont petits (65 à 423 candidats), donc leurs écarts sont moins solides.

**À retenir :** si je remplissais toutes les cases vides par la valeur la plus fréquente, j'effacerais justement le signal qui compte. Pour la phase 3, l'idée est de ne pas traiter toutes les colonnes de la même façon, et d'envisager une catégorie « Inconnu » pour celles où l'absence de réponse est informative. C'est une piste, pas une décision : elle sera arrêtée colonne par colonne.

## Ce que je retiens en cinq lignes

1. La référence est de 24,9 %, et chaque groupe se compare à elle, avec son effectif.
2. Sans expérience pertinente (33,8 %) ou en cours à temps plein (38,1 %), on cherche plus souvent à changer.
3. Un indice de développement bas va avec un taux élevé, mais surtout à cause d'une ville, `city_21` (59,1 %).
4. `training_hours` distingue très peu les deux groupes.
5. Pour `company_size` et `company_type`, une case vide est un signal (autour de 40 %), pas du bruit.

## Les limites de cette phase

- **Un lien n'est pas une cause.** « Les candidats de `city_21` ont plus tendance à chercher un emploi » est une observation. « Vivre à `city_21` pousse à changer » n'en est pas une.
- **Je n'ai fait aucun test statistique.** Je compare des taux par groupe, sans calculer si l'écart pourrait être dû au hasard. Je parle donc d'écarts « nets » ou « marqués », pas « prouvés ».
- **Les petits groupes sont fragiles** (`Phd`, `Primary School`, `Other`, et les cases vides de `experience`, `enrolled_university`, `last_new_job`).
- **Une ville pèse beaucoup.** À la modélisation, je regarderai si le modèle s'appuie surtout sur `city_21`. Si oui, ce sera une limite à écrire dans le README.
- **Les données sont publiques**, pas celles d'une vraie entreprise.

## Ce qu'il reste à vérifier

- Est-ce que `city` et `city_development_index` portent la même information (une ville = un seul indice) ?
  `df.groupby("city")["city_development_index"].nunique().max()` : si le résultat est 1, oui.
- Est-ce que les cases vides de `company_size` et de `company_type` tombent sur les mêmes lignes ?
  `df[["company_size", "company_type"]].isna().value_counts()`

## Le code utilisé

```python
# Référence
df["target"].mean()   # 0.249

# Taux et effectif par catégorie
(df.groupby("education_level", dropna=False)["target"]
   .agg(["mean", "count"])
   .sort_values("mean", ascending=False))

# Taux par tranche d'indice
tranches = pd.cut(df["city_development_index"], bins=[0, 0.6, 0.65, 0.8, 0.9, 1.0])
df.groupby(tranches, observed=True)["target"].agg(["mean", "count"])

# Histogrammes comparant les deux groupes
sns.histplot(data=df, x="training_hours", hue="target",
             stat="density", common_norm=False)

# Cases vides : le taux change-t-il quand la colonne est vide ?
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

Deux détails utiles : `dropna=False` garde les cases vides comme un groupe à part, et `common_norm=False` permet de comparer la forme de deux courbes même si les groupes n'ont pas la même taille.
## Prochaine étape

La phase 3 : décider, colonne par colonne, quoi faire des valeurs manquantes. Le tableau « une case vide, ça veut dire quelque chose » est le point de départ.