# Fiche de Cadrage — Projet de Fin de Module Machine Learning

**Module :** Machine Learning — 2ème année Cycle d'Ingénieurs – GI
**Encadrant :** Pr. Y. EL YOUNOUSSI
**Année académique :** 2025-2026
**Équipe :**
- CHOUHE Jihane
- EL BARKOUKI Alae
- EL HAMDAOUI Jihane

---

## 1. Sujet et Domaine Métier

**Titre du projet :** Moniteur de Tendances en Santé Mentale (MTSM) — Détection automatique de détresse psychologique dans les publications Reddit

**Domaine métier :** Santé mentale numérique / Veille psychologique sur les réseaux sociaux

**Problématique métier :**
Les plateformes de discussion en ligne comme Reddit accueillent des millions de publications quotidiennes, parmi lesquelles de nombreux utilisateurs expriment des signes de détresse psychologique (dépression, anxiété, idéations suicidaires). Identifier automatiquement ces signaux permettrait à des organisations de santé mentale, des modérateurs ou des systèmes d'alerte précoce d'intervenir plus efficacement et plus rapidement.

**Question métier principale :**
> Peut-on détecter automatiquement, à partir des métadonnées structurelles d'une publication Reddit (longueur, heure, flair, NSFW, etc.), si cette publication provient d'un contexte de détresse psychologique ?

---

## 2. Source de Données

**API utilisée :** Arctic Shift Public API (accès aux données historiques Reddit)
**URL :** `https://arctic-shift.photon-reddit.com/api`
**Endpoints interrogés :**
- `/posts/search` — Recherche de publications par subreddit
- Paramètres : `subreddit`, `limit`, `after` (pagination par curseur)

**Date de collecte :** 2026-05-14
**Licence :** Données publiques Reddit (contenu public soumis aux CGU Reddit)

**Subreddits sources :**

| Catégorie | Subreddit | Justification |
|---|---|---|
| Détresse (classe 1) | r/depression | Publications sur la dépression |
| Détresse (classe 1) | r/SuicideWatch | Expressions d'idéations suicidaires |
| Détresse (classe 1) | r/anxiety | Publications sur l'anxiété |
| Détresse (classe 1) | r/mentalhealth | Publications générales sur la santé mentale |
| Neutre (classe 0) | r/CasualConversation | Discussions légères et quotidiennes |
| Neutre (classe 0) | r/AskReddit | Questions ouvertes, aucune vocation clinique |
| Neutre (classe 0) | r/fitness | Sport et bien-être physique |
| Neutre (classe 0) | r/productivity | Productivité et organisation personnelle |

---

## 3. Objectifs Métiers Quantifiés

| # | Objectif Métier |
|---|---|
| OM1 | Détecter au moins 80 % des publications exprimant une détresse psychologique réelle (minimiser les cas non détectés) |
| OM2 | Limiter les fausses alertes à un niveau acceptable — maintenir une précision supérieure ou égale à 50 % sur la classe positive pour éviter de surcharger les équipes de modération |
| OM3 | Produire un modèle capable de généraliser au-delà des subreddits connus (le subreddit source ne doit pas être utilisé comme feature) |
| OM4 | Obtenir une performance globale mesurée par le F1-score supérieur ou égal à 0,70 sur la classe minoritaire (détresse) |

---

## 4. Traduction Métier vers Objectifs ML

| Objectif Métier | Objectif ML | Métrique principale | Seuil cible |
|---|---|---|---|
| OM1 — Détecter 80 % des cas de détresse | Maximiser le rappel sur la classe positive (`is_distress = 1`) | Recall | >= 0,80 |
| OM2 — Limiter les fausses alertes | Maintenir une précision correcte sur la classe positive | Precision | >= 0,50 |
| OM3 — Généralisation | Exclure `subreddit` et `flair_text` des features ; évaluer sur un hold-out stratifié | Écart train/val (overfitting gap) | < 5 pts de F1 |
| OM4 — Performance globale | Optimiser le F1-score sur la classe minoritaire | F1-score (classe 1) | >= 0,70 |

**Métrique principale retenue : F1-score sur la classe `is_distress = 1`**

**Métriques secondaires :** PR-AUC (Précision-Rappel), Recall

**Métriques exclues comme principale :**
- Accuracy seule : trop optimiste sur des données 85/15, un modèle prédisant toujours la classe 0 obtiendrait 85 % d'accuracy sans aucune utilité réelle.
- ROC-AUC seule : avantageuse même pour des modèles médiocres sur données déséquilibrées, ne reflète pas fidèlement la performance sur la classe minoritaire.

---

## 5. Analyse du Coût Métier Asymétrique

**Question fondamentale : faux positif ou faux négatif — lequel est le plus coûteux ?**

| Type d'erreur | Description concrète | Coût estimé |
|---|---|---|
| Faux négatif (FN) | Une publication de détresse réelle n'est pas détectée — l'utilisateur ne reçoit aucune aide, risque de crise non prévenue | Très élevé — risque humain direct, conséquences potentiellement irréversibles |
| Faux positif (FP) | Une publication neutre est classée comme détresse — un modérateur examine inutilement un post sans risque | Faible — coût en temps de modération uniquement |

**Conclusion asymétrique :**

Le faux négatif est bien plus coûteux qu'un faux positif dans notre contexte. Rater un signal de détresse réel peut avoir des conséquences humaines graves et irréversibles, tandis qu'une fausse alerte engendre seulement un effort de vérification manuelle supplémentaire de la part d'un modérateur.

**Décision stratégique :** on privilégie le recall comme contrainte dure (>= 0,80), tout en cherchant à maintenir une précision acceptable (>= 0,50) pour ne pas rendre le système inutilisable en pratique.

**Impact sur le seuil de décision :** le seuil de classification sera abaissé en dessous de 0,50 si nécessaire pour maximiser le recall, au détriment d'une légère baisse de précision.

---

## 6. Description du Dataset Constitué

| Critère | Valeur | Conformité |
|---|---|---|
| Type de tâche | Classification supervisée binaire | Conforme |
| Taille totale | 12 000 lignes | Conforme (>= 10 000) |
| Nombre de features utilisables | 8 features après exclusion des colonnes constantes | Conforme (>= 8) |
| Classe minoritaire | 1 800 lignes — 15,0 % | Conforme (entre 5 % et 25 %) |
| Types de variables | Numériques + Binaires + Catégorielles | Conforme |

**Variable cible :** `is_distress` dans {0, 1}
- 1 : publication issue d'un subreddit de détresse psychologique
- 0 : publication issue d'un subreddit neutre

**Distribution des classes :**

```
is_distress = 0 : 10 200 lignes (85,0 %)
is_distress = 1 :  1 800 lignes (15,0 %)
```

**Features retenues après feature engineering :**

| Feature | Type | Description |
|---|---|---|
| `hour_of_day` | Numérique | Heure UTC de publication (0–23) |
| `title_length_chars` | Numérique | Nombre de caractères dans le titre |
| `body_length_chars` | Numérique | Nombre de caractères dans le corps |
| `title_ends_with_question` | Binaire | 1 si le titre se termine par ? |
| `is_nsfw` | Binaire | 1 si la publication est marquée 18+ |
| `has_url_in_body` | Binaire | 1 si le corps contient une URL |
| `has_body` | Binaire | 1 si le corps est non vide (feature engineerée) |

**Colonnes exclues avant entraînement :** `subreddit` (fuite d'information), `flair_text` (haute cardinalité, redondance), `author_is_new`, `author_comment_karma`, `author_post_karma` (constantes à -1, non collectées).

---

## 7. Pipeline de Pré-traitement Prévu (Phase 3)

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/dataset.csv")

# 1. Suppression des colonnes à fuite ou constantes
drop_cols = ["subreddit", "flair_text",
             "author_is_new", "author_comment_karma", "author_post_karma"]
df.drop(columns=drop_cols, inplace=True)

# 2. Feature engineering : indicateur de corps non vide
df["has_body"] = (df["body_length_chars"] > 0).astype(int)

# 3. Séparation features / cible
X = df.drop(columns=["is_distress"])
y = df["is_distress"]

# 4. Division stratifiée entraînement / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. Gestion du déséquilibre
# Option A : class_weight="balanced" dans scikit-learn
# Option B : SMOTE sur X_train uniquement (jamais sur X_test)
```

---

## 8. Livrables de la Phase 1

| N° | Livrable | Contenu |
|---|---|---|
| 1 | `cadrage.md` | Ce document |
| 2 | `src/data_collection.py` | Script de collecte reproductible |
| 3 | `data/dataset.csv` + `data/sample.csv` | Dataset complet (12 000 lignes) et extrait (100 lignes) |
| 4 | `DATASET.md` | Documentation complète du dataset |
| 5 | `notebooks/01_discovery.ipynb` | Notebook exploratoire initial avec vérification du déséquilibre |

---

*Document rédigé par : Jihane Chouhe, Alae El Barkouki, Jihane El Hamdaoui*
*Dernière mise à jour : 2026-05-15*