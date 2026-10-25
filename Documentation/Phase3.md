# Phase 3 : décider quoi faire des cases vides

Cette page raconte comment j'ai décidé quoi faire des valeurs manquantes, et pourquoi. Elle sert aussi de guide pour moi : si je relis ce document dans un mois ou la veille d'un entretien, je dois retrouver le raisonnement, pas seulement les décisions. Elle est écrite pour qu'on puisse comprendre chaque décision sans avoir suivi le projet depuis le début.

Une chose à garder en tête avant de commencer : **dans cette phase, je ne remplis rien.** Je décide et je documente. Le remplissage se fera plus tard, dans le pipeline, une fois les données découpées en train et test. J'explique pourquoi plus bas.

## Ce que je voulais savoir

Après la phase 2, j'avais un constat gênant : les cases vides ne se ressemblent pas. Quand `company_size` est vide, 40,6 % des candidats cherchent à changer d'emploi. Quand `education_level` est vide, on est à 22,6 %, presque la moyenne. Si je remplis tout pareil, par exemple avec la valeur la plus fréquente, je risque d'effacer justement l'information qui compte.

Donc la question de cette phase est simple : **pour chacune des 8 colonnes qui ont des cases vides, qu'est-ce que je fais, et pourquoi ?**

## Le vocabulaire utile

- **Imputer**, c'est remplir une case vide avec une valeur choisie, au lieu de la laisser vide ou de supprimer la ligne. L'inconvénient : la valeur est une supposition, pas une vraie réponse.
- **La médiane**, c'est la valeur du milieu quand on range toutes les valeurs dans l'ordre. Elle n'est pas tirée par quelques valeurs extrêmes, contrairement à la moyenne.
- **La valeur la plus fréquente** (le *mode*), c'est la catégorie qui revient le plus souvent.
- **Une catégorie « Inconnu »**, c'est garder l'absence de réponse comme une information en créant une catégorie à part.
- **Une colonne « était vide »**, c'est une colonne de 0 et de 1 ajoutée à côté, qui dit si la case d'origine était vide. On peut imputer **et** garder le signal.
- **Un fait et une hypothèse.** Un fait, c'est ce que les données montrent (par exemple : 5 360 candidats n'ont ni `company_size` ni `company_type`). Une hypothèse, c'est une explication possible que je ne peux pas prouver (par exemple : ils n'ont pas d'employeur actuel). Je les écris toujours séparément.
- **Une fuite de données**, c'est quand des informations du jeu de test influencent l'entraînement. J'y reviens plus bas.

## Ma méthode : cinq questions pour chaque colonne

1. **La colonne est-elle presque vide** (plus de 60 %) ? Aucune des miennes ne l'est, donc je ne supprime aucune colonne.
2. **Est-ce un nombre caché dans du texte ?** C'est le cas de `experience` et `last_new_job`. Il faut convertir avant de remplir.
3. **L'écart est-il grand** entre le taux de changement quand la case est vide et quand elle est remplie ? Ma règle de départ : au-dessus d'environ 10 points, c'est un signal fort ; en dessous de 5, il est faible. Ce sont des repères que je me suis donnés, pas des lois.
4. **Le groupe de cases vides est-il gros ou petit ?** Au-dessus d'environ 500 candidats, l'écart est assez fiable. En dessous, je m'en méfie.
5. **La case vide veut-elle dire quelque chose de précis ?** Par exemple « sans objet ».

Puis je choisis dans ce menu :

| Traitement | Quand y penser |
|---|---|
| Supprimer la colonne | elle est presque vide |
| Médiane | colonne numérique, quand l'absence ne dit rien |
| Valeur la plus fréquente | colonne catégorielle, quand l'absence ne dit rien |
| Catégorie « Inconnu » ou « Non applicable » | colonne catégorielle, quand l'absence dit quelque chose |
| Colonne « était vide » (0 ou 1) | quand je veux imputer et garder le signal |

## Ce que j'ai vérifié avant de décider

### Les deux colonnes sur l'entreprise manquent presque toujours ensemble

J'ai croisé les cases vides de `company_size` et de `company_type` :

| `company_size` vide ? | `company_type` vide ? | Candidats |
|---|---|---|
| non | non | 12 440 |
| **oui** | **oui** | **5 360** |
| non | oui | 780 |
| oui | non | 578 |

Le code :

```python
df[["company_size", "company_type"]].isna().value_counts()
```

`True` veut dire « la case est vide » et `False` « la case est remplie ». Chaque ligne du résultat est une combinaison : les deux remplies, les deux vides, ou une seule vide. Les totaux se recoupent : 5 360 + 578 = 5 938 candidats sans `company_size`, et 5 360 + 780 = 6 140 sans `company_type`.

Ça donne deux pourcentages utiles : **90,3 %** des candidats sans `company_size` n'ont pas non plus de `company_type` (5 360 sur 5 938), et **87,3 %** de ceux sans `company_type` n'ont pas de `company_size` (5 360 sur 6 140). Les deux colonnes manquent donc presque toujours pour la même raison, et il ne faut pas les compter comme deux signaux indépendants.

Qui sont ces candidats ? J'ai regardé ce qu'ils ont en commun :

| | Sans `company_size` (5 938) | Tout le jeu |
|---|---|---|
| Sans expérience pertinente | 56,5 % (3 357) | 28,0 % |
| `last_new_job` = `never` | 30,0 % (1 782) | 12,8 % |

Ça va dans le sens d'une hypothèse : ils n'ont probablement pas d'employeur actuel à décrire. C'est un indice, pas une preuve.

### Une case vide de `major_discipline` veut surtout dire « sans objet »

C'est la trouvaille de la phase. Sur les 2 813 candidats sans `major_discipline` :

- 2 017 ont le niveau High School, et ce sont **tous** les candidats de ce niveau ;
- 308 ont le niveau Primary School, et ce sont **tous** les candidats de ce niveau ;
- 460 n'ont pas de niveau d'études ;
- seulement 22 Graduate et 6 Masters ont un « vrai » trou.

```python
df[df["major_discipline"].isna()]["education_level"].value_counts(dropna=False)
```

Donc 2 325 des 2 813 cases vides (82,7 %) viennent de candidats qui n'ont pas fait d'études supérieures. Ils n'ont pas de spécialité universitaire à indiquer. Si j'imputais la valeur la plus fréquente (`STEM`), je dirais à 2 325 candidats de niveau lycée ou primaire qu'ils ont une spécialité scientifique. Ça n'a aucun sens.

### Les 460 candidats sans `education_level`

Mon hypothèse de départ était qu'ils avaient « sauté toute la partie études ». Les données l'ont **corrigée** :

- leur `major_discipline` est vide à 100 % (460 sur 460) ;
- mais leur `enrolled_university` n'est vide que pour 100 d'entre eux : 360 l'ont renseignée ;
- ils ont plus de cases vides ailleurs : `company_size` 66,5 % (306), `company_type` 65,4 % (301), `gender` 43,7 % (201).

Mon hypothèse corrigée : ce sont des **profils plus incomplets en général**, pas des gens qui ont sauté un volet précis.

### Pas de valeurs manquantes déguisées

Je me suis demandé si des cases vides se cachaient sous des mots comme « Unknown », « N/A » ou un tiret. Rien de tel. `experience` contient 22 valeurs (de 1 à 20, plus `<1` et `>20`), et toutes les villes commencent par `city_`. Certaines valeurs *ressemblent* à une absence mais sont de vraies réponses : `no_enrollment`, `No Major`, `never`, `Other`. Je ne les touche pas.

## Mes décisions, colonne par colonne

### `company_size` (5 938 cases vides, 31,0 %) : catégorie « Inconnu »

Taux : 40,6 % si vide, contre 17,9 % si remplie, soit 22,7 points d'écart sur un gros groupe. L'absence de réponse est un signal fort. Si j'imputais la tranche la plus fréquente (`50-99`), je mélangerais ces candidats à un groupe qui est à 17,7 %, et j'effacerais ce signal. La colonne reste donc en catégories, avec « Inconnu » parmi elles. (L'autre option, convertir les tranches en nombres, impose d'ajouter une colonne « était vide » : on ne peut pas mettre « Inconnu » dans une colonne de nombres.)

### `company_type` (6 140 cases vides, 32,0 %) : catégorie « Inconnu »

Taux : 38,8 % si vide, contre 18,4 % si remplie, soit 20,4 points. Elle manque presque toujours avec `company_size` (87,3 %), donc probablement pour la même raison. Pour les 780 candidats où seul le type manque, mon hypothèse est un oubli ou un refus de répondre.

### `major_discipline` (2 813 cases vides, 14,7 %) : catégorie « Non applicable »

Taux : 19,5 % si vide, contre 25,9 % si remplie. Ici l'écart va **dans l'autre sens** (6,4 points en dessous), ce qui montre que l'absence ne veut pas toujours dire « plus de changement ». Le vrai sens est « sans objet » (82,7 % des cas viennent de candidats sans études supérieures), donc je garde une catégorie à part au lieu de deviner une spécialité.

### `gender` (4 508 cases vides, 23,5 %) : catégorie « Inconnu »

Taux : 30,8 % si vide, contre 23,1 % si remplie, soit 7,7 points sur un groupe de 4 508 candidats. C'est une colonne sensible : je ne devine pas une valeur personnelle à la place d'un candidat. Mon hypothèse est que certains candidats ne veulent pas l'indiquer. Je note le risque de biais dans les limites.

### `education_level` (460 cases vides, 2,4 %) : catégorie « Inconnu »

Taux : 22,6 % si vide, contre 25,0 % si remplie, soit seulement 2,4 points d'écart : l'absence de réponse porte peu de signal en elle-même. L'hésitation portait donc sur autre chose : remplir par la valeur la plus fréquente (`Graduate`) ou garder une catégorie à part. J'ai choisi « Inconnu », pour trois raisons tirées des données :

- **La cohérence avec `major_discipline`.** Les 460 candidats sans niveau d'études n'ont tous aucune spécialité. Chez les vrais Graduate, ce cas est quasi inexistant : 22 sur 11 598 (0,2 %). Les transformer en Graduate fabriquerait 460 diplômés sans spécialité, un profil que les données ne contiennent presque jamais.
- **Leur taux n'est pas celui des Graduate.** Ils sont à 22,6 %, contre 28,0 % pour les Graduate, et plus près des Masters (21,4 %). Les mélanger aux Graduate déformerait le taux de ce groupe.
- **Ce sont des profils plus incomplets en général** : 66,5 % n'ont pas de `company_size`, 43,7 % pas de `gender`. Ce ne sont pas des candidats ordinaires à qui on pourrait attribuer le niveau le plus courant.

Le groupe est petit (2,4 %), donc l'impact reste faible, mais la catégorie « Inconnu » est la plus honnête.

### `last_new_job` (423 cases vides, 2,2 %) : conversion, puis médiane

Taux : 36,4 % si vide, contre 24,7 % si remplie, soit 11,7 points. L'écart est notable, mais le groupe est petit, donc le signal est fragile. J'impute simplement. Comme la colonne est du texte (`never`, `>4`), je **convertis d'abord** en nombres (`never` en 0, `>4` en 5), puis j'impute par la médiane.

### `enrolled_university` (386 cases vides, 2,0 %) : valeur la plus fréquente

Taux : 31,9 % si vide, contre 24,8 % si remplie, soit environ 7 points sur un petit groupe : fragile. C'est une colonne catégorielle, donc pas de médiane. J'impute par la valeur la plus fréquente, `no_enrollment` (13 817 candidats).

### `experience` (65 cases vides, 0,3 %) : conversion, puis médiane

Taux : 35,4 % si vide, contre 24,9 % si remplie. L'écart de 10,5 points ne veut presque rien dire sur un groupe de 65 candidats. Après conversion (`<1` en 0, `>20` en 21), la colonne est numérique et j'impute par la médiane.

## Vue d'ensemble

| Colonne | Cases vides | Taux si vide / si rempli | Décision |
|---|---|---|---|
| `company_size` | 31,0 % (5 938) | 40,6 % / 17,9 % | catégorie « Inconnu » |
| `company_type` | 32,0 % (6 140) | 38,8 % / 18,4 % | catégorie « Inconnu » |
| `major_discipline` | 14,7 % (2 813) | 19,5 % / 25,9 % | catégorie « Non applicable » |
| `education_level` | 2,4 % (460) | 22,6 % / 25,0 % | catégorie « Inconnu » |
| `gender` | 23,5 % (4 508) | 30,8 % / 23,1 % | catégorie « Inconnu » |
| `last_new_job` | 2,2 % (423) | 36,4 % / 24,7 % | conversion, puis médiane |
| `enrolled_university` | 2,0 % (386) | 31,9 % / 24,8 % | valeur la plus fréquente |
| `experience` | 0,3 % (65) | 35,4 % / 24,9 % | conversion, puis médiane |

## Les conversions à faire avant d'imputer

| Colonne | Conversion |
|---|---|
| `experience` | `<1` en 0, `>20` en 21, les autres valeurs gardent leur nombre |
| `last_new_job` | `never` en 0, `>4` en 5, les autres valeurs gardent leur nombre |
| `company_size` | les tranches en catégories ordonnées (le format `10/49`, avec une barre oblique, est incohérent avec les autres, qui ont un tiret) |

Ce sont des approximations. Pour `>20`, je ne connais pas la vraie valeur ; pour `<1`, ça va de quelques jours à presque un an. Je prends une valeur raisonnable qui garde l'ordre, et je le dis dans les limites.

## Pourquoi convertir avant d'imputer

Pour `experience` et `last_new_job`, la conversion vient **avant** l'imputation, pour une raison pratique : on ne peut pas calculer une médiane sur du texte. Tant que la colonne contient `never` ou `>4`, pandas ne sait pas ranger les valeurs. Une fois les valeurs transformées en nombres qui gardent l'ordre (`never` en 0, `>4` en 5), la médiane a un sens.

## Médiane ou moyenne ?

Pour une colonne numérique, je choisis la médiane plutôt que la moyenne quand les valeurs sont étalées ou qu'il y a des valeurs extrêmes. Quelques très grandes valeurs tirent la moyenne vers le haut ; la médiane, qui est la valeur du milieu, ne bouge presque pas. C'est le cas typique des colonnes étalées vers la droite, comme `training_hours`. Pour une colonne catégorielle, la médiane n'existe pas : on prend la valeur la plus fréquente, ou on crée une catégorie.

## Pourquoi une imputation simple pour les petits groupes

Pour les groupes de 65 à 423 candidats (`experience`, `enrolled_university`, `last_new_job`), je ne cherche pas de traitement sophistiqué. Avec si peu de candidats, quelques cas suffisent à déplacer le taux : l'écart que je mesure est peu fiable, et un traitement plus fin ne s'appuierait que sur du bruit. Une imputation simple est plus honnête.

## Un exemple de fait et un exemple d'hypothèse

La différence compte, parce que je ne l'applique pas à la même chose. Un **fait** : 5 360 des 5 938 candidats sans `company_size` n'ont pas non plus de `company_type`. Une **hypothèse** : ils n'ont pas d'employeur actuel à décrire. Les données montrent la première, elles ne prouvent pas la seconde. De même, 82,7 % des cases vides de `major_discipline` viennent de candidats sans études supérieures (fait) ; l'explication « sans objet » est une hypothèse très plausible, mais une hypothèse.

## Pourquoi je ne remplis rien maintenant

Si je calculais une médiane sur **toutes** les lignes, y compris celles qui iront dans le jeu de test, le test aurait déjà influencé mes données d'entraînement. C'est une **fuite de données** : les résultats finaux paraîtraient meilleurs qu'ils ne sont.

L'ordre à respecter :

1. **Découper** train et test (phase 4).
2. **Convertir** les nombres cachés.
3. **Imputer**, avec des valeurs apprises sur le train seulement, puis appliquées au test.
4. **Encoder** les catégories pour le modèle.

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

# Taux si vide et si rempli, pour une colonne
vide = df["company_size"].isna()
df.loc[vide, "target"].mean()      # 0.406
df.loc[~vide, "target"].mean()     # 0.179

# Valeurs déguisées ?
for col in df.select_dtypes(exclude="number").columns:
    print(col)
    print(df[col].value_counts(dropna=False).head(10))

# Essais d'imputation sur une COPIE, jamais sur df
df2 = df.copy()
df2["company_size"] = df2["company_size"].fillna("Inconnu")
df2["company_size_vide"] = df["company_size"].isna().astype(int)
```

`~vide` veut dire « le contraire de vide » : les candidats qui ont répondu.

## Les pièges à éviter

- **Imputer par la valeur la plus fréquente quand la case vide veut dire « sans objet »** : c'est ce qui aurait donné une spécialité `STEM` à des candidats de niveau lycée.
- **Écrire une décision et une justification qui se contredisent** (par exemple « Inconnu » avec « l'absence ne porte pas de signal »).
- **Dire « petit groupe » sans donner l'effectif et l'écart.**
- **Ignorer le « pourquoi ça manque »** : c'est lui qui donne les bonnes décisions.
- **Confondre lignes complètes et lignes vides** après `dropna()` : les 8 955 lignes qui restent sont complètes.

## Les limites

- **Mes raisons sont des hypothèses.** Les données montrent des liens, pas des causes.
- **Le signal des cases vides recoupe d'autres colonnes** : `relevent_experience` porte déjà un fort signal (33,8 % contre 21,5 %). Je verrai à la modélisation si les cases vides apportent quelque chose en plus.
- **Mes seuils (10 points, 500 candidats) sont des repères, pas des tests statistiques.**
- **Les conversions des bornes** (`>20` en 21, `<1` en 0, `>4` en 5, `never` en 0) sont des approximations.
- **`gender` est sensible** : « Inconnu » évite de deviner, mais le risque de biais du modèle reste à discuter.

## Ce qui reste ouvert

- Le comportement des cases vides une fois le modèle entraîné : `relevent_experience` porte déjà un signal fort, donc je verrai si les cases vides apportent quelque chose en plus.

## Prochaine étape

La phase 4 : découper les données en jeu d'entraînement et jeu de test, avant toute modification. Voir `phase4.md`.