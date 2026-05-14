# DATASET.md — Moniteur de Tendances en Santé Mentale (MTSM)
Travail Réalisé par:
**-> CHOUHE Jihane**
**-> EL BARKOUKI Alae**
**-> EL HAMDAOUI Jihane**


*Dans le cadre du module GI2-Machine Learning*

## Vue d'ensemble

| Propriété | Valeur |
|---|---|
| **Nombre total de lignes** | 12 000 |
| **Colonnes** | 12 |
| **Lignes de détresse** | 1 800 (15,0 %) |
| **Lignes non-détresse** | 10 200 (85,0 %) |
| **Date de collecte** | 2026-05-14 |
| **Source** | Reddit via l'API publique Arctic Shift |
| **Type de tâche** | Classification binaire (`is_distress`) |
| **Graine aléatoire (mélange)** | 42 |

---

## Sources de données

Les publications ont été collectées depuis 8 sous-reddits, réparties en deux catégories qui déterminent le label binaire :

| Catégorie | Sous-reddit | Publications collectées | `is_distress` |
|---|---|---|---|
| Détresse | r/depression | 450 | 1 |
| Détresse | r/SuicideWatch | 450 | 1 |
| Détresse | r/anxiety | 450 | 1 |
| Détresse | r/mentalhealth | 450 | 1 |
| Neutre | r/CasualConversation | 2 550 | 0 |
| Neutre | r/AskReddit | 2 550 | 0 |
| Neutre | r/fitness | 2 550 | 0 |
| Neutre | r/productivity | 2 550 | 0 |

**Justification de l'étiquetage.** Les sous-reddits dédiés à l'expression de détresse psychologique (dépression, idées suicidaires, anxiété, troubles de santé mentale) sont étiquetés `is_distress = 1`. Les sous-reddits généralistes ou thématiques sans vocation clinique sont étiquetés `is_distress = 0`. Aucun contenu de publication individuelle n'a été lu pour attribuer les labels ; le sous-reddit lui-même sert de proxy pour la vérité terrain.

---

## Schéma

### Colonnes de caractéristiques

| # | Colonne | Type | Plage / Valeurs | Description |
|---|---|---|---|---|
| 1 | `hour_of_day` | int | 0 – 23 | Heure UTC extraite de `created_utc`. `-1` si l'horodatage est indisponible. |
| 2 | `title_length_chars` | int | 1 – 300 | Nombre de caractères dans le titre de la publication. |
| 3 | `body_length_chars` | int | 0 – 9 000 | Nombre de caractères dans le corps de la publication (`selftext`). 0 pour les publications sans texte ou titre uniquement. Les corps supprimés/retirés sont traités comme 0. |
| 4 | `title_ends_with_question` | int (binaire) | {0, 1} | 1 si le titre se termine par `?`, sinon 0. |
| 5 | `is_nsfw` | int (binaire) | {0, 1} | 1 si la publication est marquée `over_18` par Reddit. |
| 6 | `has_url_in_body` | int (binaire) | {0, 1} | 1 si le corps contient `http://` ou `https://`. |
| 7 | `author_is_new` | int | {-1} | Espace réservé — âge du compte auteur non collecté. Toujours `-1`. |
| 8 | `author_comment_karma` | int | {-1} | Espace réservé — karma auteur non collecté. Toujours `-1`. |
| 9 | `author_post_karma` | int | {-1} | Espace réservé — karma auteur non collecté. Toujours `-1`. |
| 10 | `flair_text` | string | Texte libre / "None" | Flair de la publication défini par l'auteur ou les modérateurs. `"None"` si absent. |
| 11 | `subreddit` | string | 8 valeurs | Nom du sous-reddit source (sans préfixe `r/`). |



### Colonne cible

| Colonne | Type | Valeurs | Classe positive |
|---|---|---|---|
| `is_distress` | int (binaire) | {0, 1} | 1 → publication issue d'un sous-reddit de détresse |

---

## Statistiques Descriptives

### Caractéristiques numériques

| Statistique | `hour_of_day` | `title_length_chars` | `body_length_chars` |
|---|---|---|---|
| moyenne | 17,4 | 54,7 | 201,3 |
| écart-type | 4,1 | 30,8 | 497,6 |
| min | 0 | 1 | 0 |
| 25 % | 15 | 30 | 0 |
| 50 % | 18 | 49 | 0 |
| 75 % | 21 | 73 | 456 |
| max | 23 | 299 | 8 917 |

### Caractéristiques binaires

| Caractéristique | Taux global | Taux détresse | Taux neutre |
|---|---|---|---|
| `title_ends_with_question` | 53,5 % | 23,8 % | 59,3 % |
| `is_nsfw` | 4,6 % | 14,2 % | 2,7 % |
| `has_url_in_body` | 0,4 % | 0,2 % | 0,4 % |

### Par classe

| Métrique | `is_distress = 0` | `is_distress = 1` |
|---|---|---|
| Effectif | 10 200 | 1 800 |
| Moy. `title_length_chars` | 57,1 | 42,8 |
| Moy. `body_length_chars` | 108,7 | 724,6 |
| Moy. `hour_of_day` | 17,2 | 18,8 |

La différence la plus frappante entre les classes est la **longueur du corps** : les publications de détresse sont en moyenne ~6,7× plus longues que les publications neutres, ce qui reflète la nature narrative des témoignages personnels en santé mentale.

### Déséquilibre de classes

```
is_distress
0    10 200   (85,0 %)
1     1 800   (15,0 %)
```

Le ratio 85/15 est intentionnel (voir justification de l'étiquetage). Il s'inscrit dans la cible de 5–25 % de classe minoritaire définie dans `cadrage.md`. Lors de l'entraînement, envisager :
- **Perte pondérée** (ex. `class_weight="balanced"` dans scikit-learn)
- **Rééchantillonnage** (SMOTE sur le fold d'entraînement, jamais sur validation/test)
- Évaluation avec le **score F1** ou l'**AUROC** plutôt que la précision seule

---

## Distribution des Classes par Sous-reddit

| Sous-reddit | Publications | `is_distress` | % de la classe |
|---|---|---|---|
| r/depression | 450 | 1 | 25,0 % de la détresse |
| r/SuicideWatch | 450 | 1 | 25,0 % de la détresse |
| r/anxiety | 450 | 1 | 25,0 % de la détresse |
| r/mentalhealth | 450 | 1 | 25,0 % de la détresse |
| r/CasualConversation | 2 550 | 0 | 25,0 % du neutre |
| r/AskReddit | 2 550 | 0 | 25,0 % du neutre |
| r/fitness | 2 550 | 0 | 25,0 % du neutre |
| r/productivity | 2 550 | 0 | 25,0 % du neutre |

Chaque classe est parfaitement équilibrée **au sein** de sa catégorie (représentation égale par sous-reddit), ce qui aide à éviter que le modèle n'apprenne les styles d'écriture propres à chaque sous-reddit comme proxy de la détresse.

---

## Analyse des Flairs

Le texte de flair n'est **significatif que pour les sous-reddits de détresse** — les sous-reddits neutres utilisent les flairs de manière clairsemée ou avec des étiquettes thématiques sans rapport avec la santé mentale. Les flairs les plus fréquents dans la classe de détresse sont :

| Flair | Nombre approx. | Sous-reddit(s) |
|---|---|---|
| `None` | ~650 | tous distresse |
| `Advice Needed` | ~130 | r/anxiety |
| `Venting` | ~120 | r/mentalhealth, r/anxiety |
| `Health` | ~110 | r/anxiety |
| `Need Support` | ~95 | r/mentalhealth |
| `Content Warning: Suicidal Thoughts / Self Harm` | ~90 | r/mentalhealth |
| `Medication` | ~85 | r/anxiety |
| `Needs A Hug/Support` | ~60 | r/anxiety |
| `Question` | ~55 | r/mentalhealth |
| `Discussion` | ~40 | r/anxiety |

Le flair est une variable catégorielle à **haute cardinalité** (~80 valeurs uniques). Si utilisé comme caractéristique, appliquer un encodage cible ou regrouper les flairs rares dans un seau `Autre`. Il est également possible de le supprimer — le flair est partiellement redondant avec la colonne subreddit, et le subreddit lui-même ne doit pas être utilisé comme caractéristique du modèle en production (le modèle doit se généraliser au-delà des sous-reddits connus).

---

## Limitations et Mises en Garde Connues

| Problème | Impact | Recommandation |
|---|---|---|
| **Fuite d'information via le sous-reddit** | La colonne `subreddit` prédit parfaitement `is_distress` — la supprimer avant l'entraînement | Pré-traitement obligatoire |
| **Métadonnées auteur indisponibles** | `author_is_new`, `author_comment_karma`, `author_post_karma` sont constantes | Supprimer ces 3 colonnes |
| **Pas de texte brut** | Seules les caractéristiques structurelles/métadonnées sont disponibles ; le signal NLP est absent | Travaux futurs : collecter le texte brut des titres/corps |
| **Biais de titre AskReddit** | Presque tous les titres d'AskReddit se terminent par `?` → gonfle `title_ends_with_question` pour la classe neutre | Le modèle peut trop s'appuyer sur cette caractéristique ; vérifier l'importance par permutation |
| **Portée temporelle** | Données collectées en une seule journée (2026-05-14) ; les tendances saisonnières ou hebdomadaires ne sont pas représentées | Étendre la fenêtre de collecte dans les itérations futures |
| **Majorité body_length = 0** | ~52 % des lignes ont un corps vide | Les modèles à base d'arbres gèrent bien cela ; les modèles linéaires peuvent nécessiter une imputation ou un indicateur binaire `has_body` |
| **Taux NSFW en détresse** | 14,2 % des publications de détresse sont marquées NSFW contre 2,7 % pour le neutre | Signal utile, mais les politiques NSFW varient selon les sous-reddits |

---

## Fichiers

| Fichier | Description |
|---|---|
| `data/dataset.csv` | Jeu de données complet mélangé (12 000 lignes × 12 colonnes) |
| `data/sample.csv` | 100 premières lignes — vérification rapide |
| `data/checkpoint_<subreddit>.json` | Dictionnaires de caractéristiques brutes par sous-reddit (utilisés pour la collecte avec reprise) |
| `data/collection.log` | Journal de collecte détaillé avec les statuts des appels API |
| `data_collection.py` | Script de collecte (livrable P1) |

---

## Pipeline de Pré-traitement Recommandé (pour P3)

```python
import pandas as pd

df = pd.read_csv("data/dataset.csv")

# 1. Supprimer les colonnes de fuite / constantes
drop_cols = ["subreddit", "flair_text",
             "author_is_new", "author_comment_karma", "author_post_karma"]
df.drop(columns=drop_cols, inplace=True)

# 2. Optionnel : ajouter un indicateur has_body
df["has_body"] = (df["body_length_chars"] > 0).astype(int)

# Caractéristiques et cible
X = df.drop(columns=["is_distress"])
y = df["is_distress"]

# 3. Division entraînement/validation/test (stratifiée)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```

---

*Dernière mise à jour : 2026-05-15.*