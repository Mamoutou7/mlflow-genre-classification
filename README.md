# Genre Classification

Pipeline MLOps de classification de genres musicaux à partir de **caractéristiques audio** et de **métadonnées textuelles** (`title`, `song_name`).

Le projet orchestre un workflow complet avec **MLflow**, **Hydra** et **Weights & Biases (W&B)** :

- téléchargement du dataset brut ;
- prétraitement et feature engineering ;
- validation des données avec `pytest` ;
- découpage du dataset ;
- entraînement d’un modèle **Random Forest** ;
- export du modèle avec **MLflow** ;
- évaluation finale sur l’ensemble de test.

## Objectif

Prédire le **genre musical** d’un morceau à partir de variables numériques, catégorielles et textuelles.

Genres présents dans le dataset :

- Dark Trap
- Underground Rap
- Trap Metal
- Emo
- Rap
- RnB
- Pop
- Hiphop
- techhouse
- techno
- trance
- psytrance
- trap
- dnb
- hardstyle

## Stack technique

- Python
- MLflow
- Hydra
- Weights & Biases (W&B)
- scikit-learn
- pandas
- SciPy
- matplotlib
- pytest

## Structure du projet

```text
genre-classification/
├── config.yaml                 # Configuration globale du pipeline
├── conda.yml                   # Environnement principal MLflow/Hydra
├── main.py                     # Orchestrateur du pipeline
├── MLproject                   # Point d’entrée MLflow du projet racine
│
├── download/
│   ├── download_data.py        # Téléchargement du dataset et log en artifact W&B
│   ├── conda.yml
│   └── MLproject
│
├── preprocess/
│   ├── run.py                  # Suppression des doublons + création de text_feature
│   ├── conda.yml
│   └── MLproject
│
├── check_data/
│   ├── conftest.py             # Chargement des artifacts de test
│   ├── test_data.py            # Tests de schéma, plage de valeurs, KS test
│   ├── conda.yml
│   └── MLproject
│
├── segregate/
│   ├── run.py                  # Split train / test stratifié
│   ├── conda.yml
│   └── MLproject
│
├── random_forest/
│   ├── run.py                  # Pipeline sklearn + entraînement + export MLflow
│   ├── conda.yml
│   └── MLproject
│
└── evaluate/
    ├── run.py                  # Évaluation sur le jeu de test
    ├── conda.yml
    └── MLproject
```

## Données utilisées

Le dataset est téléchargé depuis l’URL définie dans `config.yaml` :

- format source : **Parquet**
- artifact brut W&B : `raw_data.parquet`
- artifact prétraité : `preprocessed_data.csv`

### Variables exploitées

**Numériques**

- `danceability`
- `energy`
- `loudness`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`
- `duration_ms`

**Catégorielles**

- `time_signature`
- `key`

**Textuelles**

- `text_feature` = concaténation de `title` et `song_name`

**Cible**

- `genre`

## Fonctionnement du pipeline

### 1. Download

Le module `download/` :

- télécharge le dataset distant ;
- le stocke temporairement ;
- l’enregistre comme artifact W&B.

### 2. Preprocess

Le module `preprocess/` :

- charge l’artifact brut ;
- supprime les doublons ;
- remplit les valeurs manquantes de `title` et `song_name` ;
- crée la colonne `text_feature`.

### 3. Check Data

Le module `check_data/` exécute plusieurs contrôles :

- présence des colonnes attendues ;
- vérification des types ;
- validation des plages de valeurs ;
- contrôle de dérive statistique via **Kolmogorov-Smirnov test**.

### 4. Segregate

Le module `segregate/` :

- lit le dataset prétraité ;
- effectue un split **train / test** ;
- applique une stratification sur `genre`.

### 5. Random Forest

Le module `random_forest/` construit un pipeline scikit-learn avec :

- imputation + standardisation pour les variables numériques ;
- imputation + `OrdinalEncoder` pour les variables catégorielles ;
- imputation + `TfidfVectorizer` pour la variable textuelle ;
- classifieur `RandomForestClassifier` avec pondération `class_weight="balanced"`.

Le script journalise notamment :

- l’**AUC multiclasses** (`roc_auc_score`, stratégie OVO) ;
- une matrice de confusion ;
- l’importance des variables ;
- un export du modèle au format MLflow.

### 6. Evaluate

Le module `evaluate/` :

- recharge le modèle exporté ;
- calcule l’AUC sur le jeu de test ;
- journalise une matrice de confusion normalisée.

## Prérequis

Avant de lancer le pipeline, assurez-vous d’avoir :

- **Conda** installé ;
- un compte **Weights & Biases** ;
- un environnement configuré pour MLflow exécutable localement.

Connexion W&B :

```bash
wandb login
```

## Installation

### Option 1 — via Conda

Depuis la racine du projet :

```bash
conda env create -f conda.yml
conda activate download_data
```

### Option 2 — environnement Python existant

Installez au minimum les dépendances suivantes :

```bash
pip install mlflow hydra-core wandb pandas scikit-learn scipy matplotlib pyarrow requests omegaconf pytest
```

## Exécution du pipeline complet

Depuis la racine du dépôt :

```bash
mlflow run .
```

Le pipeline exécute les étapes listées dans `config.yaml` :

```yaml
execute_steps:
  - download
  - preprocess
  - check_data
  - segregate
  - random_forest
  - evaluate
```

## Exécuter uniquement certaines étapes

Vous pouvez surcharger les étapes via Hydra :

```bash
mlflow run . -P hydra_options="main.execute_steps='download,preprocess,check_data'"
```

Autres exemples :

```bash
mlflow run . -P hydra_options="main.execute_steps='download,preprocess'"
mlflow run . -P hydra_options="random_forest_pipeline.tfidf.max_features=100"
mlflow run . -P hydra_options="data.test_size=0.2"
```

## Configuration

Le fichier `config.yaml` centralise les paramètres du projet.

### Paramètres principaux

```yaml
main:
  project_name: exercise_14
  experiment_name: dev
  random_seed: 42
```

### Paramètres data

```yaml
data:
  file_url: <url_du_dataset>
  reference_dataset: exercise_14/preprocessed_data.csv:v0
  ks_alpha: 0.05
  test_size: 0.3
  val_size: 0.3
  stratify: genre
```

### Paramètres modèle

```yaml
random_forest_pipeline:
  random_forest:
    n_estimators: 100
    max_depth: 13
    class_weight: balanced
  tfidf:
    max_features: 10
```

## Artifacts générés

Pendant l’exécution, le projet produit notamment les artifacts suivants dans W&B :

- `raw_data.parquet`
- `preprocessed_data.csv`
- `data_train.csv`
- `data_test.csv`
- `model_export`

## Suivi des expérimentations

Le projet s’appuie sur **Weights & Biases** pour :

- tracer les métriques ;
- stocker les datasets intermédiaires ;
- versionner les artifacts ;
- visualiser les matrices de confusion et l’importance des variables.

Le modèle final est exporté avec **MLflow**, ce qui facilite sa réutilisation ou son déploiement.

## Points d’attention

- Le workflow repose fortement sur les **artifacts W&B**. Une authentification valide est donc nécessaire.
- Le split implémenté dans `segregate/run.py` produit actuellement **train** et **test**. La validation interne du modèle est ensuite réalisée dans `random_forest/run.py` à partir du jeu d’entraînement.
- La variable textuelle utilisée par le modèle est dérivée de `title` et `song_name`, puis vectorisée avec TF-IDF.

## Pistes d’amélioration

- ajouter une vraie étape dédiée au split **train / validation / test** ;
- tester d’autres modèles (XGBoost, LightGBM, Linear SVM, Logistic Regression) ;
- enrichir les features textuelles ;
- ajouter une recherche d’hyperparamètres ;
- industrialiser le packaging et le déploiement du modèle.

## Licence

Ce projet contient un fichier `LICENSE` à la racine du dépôt. Référez-vous à ce fichier pour les conditions d’utilisation.
