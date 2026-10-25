# Phase 2 : explorer les données

Une fois les données comprises, j'ai voulu savoir ce qui, chez un candidat, va avec le fait de chercher un nouvel emploi. Cette page raconte comment j'ai comparé des groupes, ce que j'ai trouvé, et surtout comment je lis ces résultats sans m'emballer. Elle sert aussi de guide : on doit y retrouver le raisonnement, pas seulement les chiffres.

Dans cette phase, **aucune donnée n'est modifiée**. On compare des groupes de candidats.

## La règle du jeu

Dans le jeu de données, **24,9 % des candidats cherchent à changer d'emploi**. Ce chiffre est ma **référence** : tout groupe se compare à lui. Je le calcule en divisant le nombre de candidats qui changent par le nombre total : 4 777 ÷ 19 158 = 0,2493.

Pour chaque groupe, je regarde toujours deux choses :

- Le **taux** : la part de candidats du groupe qui cherchent à changer. Sans référence, un taux ne dit rien : « 33,8 % » est élevé parce que la moyenne est à 24,9 %.
- L'**effectif** : le nombre de candidats du groupe. Sans lui, on ne sait pas si le taux est fiable : 38,1 % sur 3 757 candidats est crédible, 14,0 % sur 414 candidats l'est beaucoup moins.

Le calcul d'écart se fait en **points**, pas en pourcents : passer de 24,9 % à 33,8 % est un écart de 8,9 points. Et attention à ne pas confondre la référence (24,9 %, une part) avec l'effectif total (19 158, un nombre de candidats).

## Les outils

### La référence

```python
df["target"].mean()                 # 0.2493
df["target"].sum() / len(df)        # la même chose : 4777 / 19158
```

`target` ne contient que des 0 et des 1. La somme est le nombre de 1, et en divisant par le nombre de lignes, on obtient la part de 1.

### Le taux et l'effectif par groupe

```python
(df.groupby("education_level", dropna=False)["target"]
   .agg(["mean", "count"])
   .sort_values("mean", ascending=False))
```

`groupby("education_level")` crée un groupe par niveau d'études. `["target"]` regarde la cible dans chaque groupe. `.agg(["mean", "count"])` calcule la moyenne (le taux) et le nombre de lignes (l'effectif). `dropna=False` garde les cases vides comme un groupe à part : sans lui, pandas ignorerait ces candidats, et on ne verrait pas, par exemple, que ceux qui n'ont pas indiqué la taille de leur entreprise sont à 40,6 %.

### Découper une colonne numérique en tranches

```python
tranches = pd.cut(df["city_development_index"], bins=[0, 0.6, 0.65, 0.8, 0.9, 1.0])
df.groupby(tranches, observed=True)["target"].agg(["mean", "count"])
```

`pd.cut` découpe une colonne numérique en tranches. La notation `(0.6, 0.65]` veut dire : 0,6 exclu, 0,65 inclus. J'en ai besoin parce que l'indice de développement est une variable numérique continue (93 valeurs différentes) : chaque valeur a trop peu de candidats pour calculer un taux, alors qu'une tranche donne un groupe assez gros pour comparer.

### Comparer deux distributions

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(data=df, x="training_hours", hue="target",
             stat="density", common_norm=False)
plt.show()
```

`x` est la colonne étudiée. `hue="target"` trace une courbe par groupe (cherche à changer, ou non) : il faut y mettre `target`, pas la colonne étudiée. `stat="density"` met chaque courbe à l'échelle pour comparer les **formes** et non les nombres de candidats. `common_norm=False` normalise chaque groupe séparément : sans lui, les 14 381 candidats qui ne changent pas écraseraient les 4 777 qui changent.

## Ce que j'ai trouvé

Les écarts sont calculés avec les taux arrondis (précision de ±0,1 point). Référence : 24,9 %.

### Expérience pertinente

| Groupe | Taux | Écart | Effectif |
|---|---|---|---|
| Sans expérience pertinente | 33,8 % | +8,9 points | 5 366 |
| Avec expérience pertinente | 21,5 % | −3,4 points | 13 792 |

Un écart net sur de gros groupes (les deux effectifs font 19 158). Ceux qui n'ont pas d'expérience pertinente cherchent plus souvent à changer.

### Études en cours

| Groupe | Taux | Écart | Effectif |
|---|---|---|---|
| Cours à temps plein | 38,1 % | +13,2 points | 3 757 |
| Case vide | 31,9 % | +7,0 points | 386 |
| Cours à temps partiel | 25,2 % | +0,3 point | 1 198 |
| Pas inscrit | 21,1 % | −3,8 points | 13 817 |

Le temps plein a le **taux le plus élevé** de toutes les catégories que j'ai regardées, sur un groupe de bonne taille. Les « pas inscrits » pèsent environ 72 % du jeu de données : leur taux est fiable et forcément proche de la moyenne.

### Niveau d'études

| Groupe | Taux | Écart | Effectif |
|---|---|---|---|
| Graduate | 28,0 % | +3,1 points | 11 598 |
| Case vide | 22,6 % | −2,3 points | 460 |
| Masters | 21,4 % | −3,5 points | 4 361 |
| High School | 19,5 % | −5,4 points | 2 017 |
| Phd | 14,0 % | −10,9 points | 414 |
| Primary School | 13,3 % | −11,6 points | 308 |

Pas de tendance simple du type « plus on est diplômé, plus on change » : les Graduate sont les plus hauts, et les deux plus bas, Phd et Primary School, sont aux extrémités. Ces deux groupes sont petits (722 candidats à eux deux), donc moins fiables. Les candidats sans niveau renseigné (22,6 %) se distinguent peu des autres.

### Genre

| Groupe | Taux | Écart | Effectif |
|---|---|---|---|
| Case vide | 30,8 % | +5,9 points | 4 508 |
| Female | 26,3 % | +1,4 point | 1 238 |
| Other | 26,2 % | +1,3 point | 191 |
| Male | 22,8 % | −2,1 points | 13 221 |

Entre Female, Male et Other, l'écart est faible, et `Other` ne compte que 191 candidats : son taux est peu fiable. C'est une colonne sensible, je reviens plus bas sur la précaution qu'elle demande.

### Taille de l'entreprise

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

Toutes les tranches renseignées sont sous la moyenne, entre 15 % et 23 %. Mais les 5 938 candidats qui n'ont rien indiqué sont à **40,6 %**, soit 15,7 points au-dessus de la référence. Ceux qui ont répondu sont à 17,9 % (13 220 candidats), 7,0 points en dessous. Dans cette colonne, ne pas répondre est un signal fort. Les tranches s'affichent dans l'ordre alphabétique, pas par taille.

### Indice de développement de la ville

| Tranche | Taux | Écart | Effectif |
|---|---|---|---|
| (0, 0,6] | 57,9 % | +33,0 points | 725 |
| (0,6, 0,65] | 58,3 % | +33,4 points | 2 754 |
| (0,65, 0,8] | 26,0 % | +1,1 point | 2 487 |
| (0,8, 0,9] | 13,6 % | −11,3 points | 3 549 |
| (0,9, 1,0] | 16,8 % | −8,1 points | 9 643 |

La tendance générale est « indice bas, taux élevé » : les deux tranches basses sont à plus du double de la référence. Mais elle n'est pas parfaitement régulière, puisque la dernière tranche est un peu au-dessus de l'avant-dernière.

### Les deux pics de l'indice : deux villes

L'histogramme de l'indice a **deux pics**, chacun dû à une seule ville. Pour le trouver, j'ai d'abord repéré sur le graphique où se trouvait le pic, puis filtré les candidats de cette zone et compté leurs villes :

```python
df[df["city_development_index"].between(0.62, 0.63)]["city"].value_counts().head()
# city_21 : 2702, city_129 : 3
```

| Pic | Ville | Effectif | Part du jeu | Taux | Écart |
|---|---|---|---|---|---|
| autour de 0,62 | `city_21` | 2 702 | 14,1 % | 59,1 % | +34,2 points |
| autour de 0,92 | `city_103` | 4 355 | 22,7 % | 21,3 % | −3,6 points |

Sur les 2 705 candidats dont l'indice est entre 0,62 et 0,63, **2 702 viennent de `city_21`** (99,9 %). Et `city_21` compte 2 702 candidats en tout : tous ont donc le même indice. De même, 98,1 % de la tranche (0,6, 0,65] vient de cette ville (2 702 sur 2 754). Le même raisonnement vaut pour `city_103` autour de 0,92 : ses 4 355 candidats ont tous le même indice.

Les cinq plus grosses villes sont `city_103` (4 355), `city_21` (2 702), `city_16` (1 533), `city_114` (1 336) et `city_160` (845), soit 10 771 candidats, 56 % du jeu. `city_21` et `city_103` représentent ensemble 7 057 candidats, environ 37 %.

Une nuance : le lien « indice bas, taux élevé » ne vient pas *uniquement* de `city_21`. La tranche (0, 0,6] contient 725 candidats à 57,9 % sans compter cette ville. Mais le **pic** de 0,62, lui, vient presque entièrement d'elle. Ce sont des liens observés : les données ne disent pas pourquoi `city_21` est si haute.

### Les heures de formation

Les deux courbes (ceux qui cherchent à changer et ceux qui ne cherchent pas) **se superposent presque**. La distribution est **étalée vers la droite** : beaucoup de candidats ont peu d'heures (un pic autour de 20 à 30 heures), et une longue traîne s'étire jusqu'à environ 340 heures. Cette colonne distingue peu les deux groupes ; elle apportera sans doute peu d'information au modèle. Je le vérifierai à la modélisation, par exemple en comparant les moyennes des deux groupes avec `df.groupby("target")["training_hours"].mean()`.

## Comment lire un histogramme avec deux groupes

Trois questions, dans l'ordre :

1. Les deux courbes se ressemblent-elles ou sont-elles décalées ?
2. Si elles sont décalées, de quel côté se trouve le groupe qui cherche à changer ?
3. L'écart est-il grand ou petit ?

| Ce que je vois | Ce que j'en conclus |
|---|---|
| Courbes superposées (`training_hours`) | la colonne distingue peu les deux groupes |
| Courbes décalées (`city_development_index`) | la colonne aide à les distinguer |
| Un pic haut | beaucoup de candidats ont la même valeur : chercher pourquoi |
| Courbe qui descend vers la droite | distribution étalée vers la droite |

Sur l'histogramme de l'indice, autour de 0,62, la courbe de ceux qui cherchent à changer monte à environ 25, contre environ 6 pour les autres. Autour de 0,92, c'est l'inverse : environ 30 pour ceux qui ne changent pas, 21 pour les autres. Les deux courbes sont nettement décalées. Ces courbes montrent comment chaque groupe se répartit, pas directement le taux dans chaque zone : pour avoir le taux par zone, on utilise `pd.cut` et `groupby`.

## Quand une case vide est un signal

J'ai comparé, pour chaque colonne qui a des cases vides, le taux de changement quand la case est vide et quand elle est remplie :

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

`vide` vaut `True` pour les candidats dont la colonne est vide, et `~vide` désigne le contraire : ceux qui ont répondu.

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

Cette comparaison répond à une question concrète : **est-ce que ne pas répondre porte de l'information ?** Si l'écart est grand sur un gros groupe (`company_size`, `company_type`), l'absence de réponse est un signal que je veux garder, par exemple avec une catégorie « Inconnu ». Si l'écart est faible (`education_level`), une imputation simple suffira. Et l'écart peut aller dans l'autre sens (`major_discipline`) : une case vide ne veut pas toujours dire « plus de changement ». Sur les petits groupes (65 à 423 cases vides), les écarts sont moins fiables. C'est le point de départ de la phase 3.

## Pourquoi je reste prudent

### Les petits groupes sont fragiles

Quand un groupe est petit, quelques candidats suffisent à déplacer son taux. Phd (414 candidats), Primary School (308), `Other` dans le genre (191) et les 65 cases vides d'`experience` donnent des taux intéressants mais bien moins solides que ceux de groupes de plusieurs milliers.

### Un lien n'est pas une cause

Une **corrélation**, c'est deux choses qui varient ensemble : par exemple, être de la ville `city_21` va avec un taux de 59,1 %. Une **cause** voudrait dire que l'une provoque l'autre, et les données ne le montrent pas. Pour `city_21`, plusieurs explications sont possibles : un secteur d'activité local, la façon dont les données ont été collectées, un type de formation particulier. Aucune n'est démontrée.

Je dis donc « les candidats de `city_21` ont plus tendance à chercher un nouvel emploi », jamais « vivre à `city_21` pousse à changer ». Et comme je n'ai fait aucun test statistique, je parle d'écarts « nets » ou « marqués », pas « prouvés ».

### Le genre demande une précaution particulière

C'est une information personnelle et sensible. Elle a 23,5 % de cases vides (4 508 candidats), et ceux qui n'ont pas répondu sont à 30,8 %. Je ne devine pas la valeur d'un candidat : je prévois une catégorie « Inconnu ». Je décris des liens observés, jamais des causes, et je noterai dans les limites le risque de biais d'un modèle sur cette colonne.

### Quelques pièges de lecture

- Additionner deux taux pour « combiner » deux groupes : il faut recalculer le taux sur les candidats concernés (Phd et Primary School ensemble : environ 13,7 %, pas 27,3 %).
- Lire une courbe à l'envers : `training_hours` descend vers la droite, elle n'est pas ascendante.
- Décrire la forme d'une courbe sans comparer les deux groupes.
- Tronquer au lieu d'arrondir (21,5 % et non 21,4 %), et faire la soustraction à la main avant de la citer (40,6 − 24,9 = 15,7).
- Confondre les deux pics : `city_21` pour 0,62, `city_103` pour 0,92.

## Les chiffres à retenir

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

## Les limites

- Un lien n'est pas une cause, et je n'ai fait aucun test statistique.
- Les petits groupes sont fragiles.
- Une ville pèse beaucoup : à la modélisation, je regarderai si le modèle s'appuie surtout sur `city_21`.
- Les écarts sont calculés avec des taux arrondis (précision de ±0,1 point).
- Je n'ai pas vérifié les moyennes de `training_hours` des deux groupes, seulement les histogrammes.
- Les données sont publiques, pas celles d'une vraie entreprise.

## Prochaine étape

La phase 3 : décider, colonne par colonne, quoi faire des cases vides. Voir `phase3.md`.