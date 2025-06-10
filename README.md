# Projet Data Warehouse – Analyse des Avis Clients des Banques au Maroc

## Présentation Générale
Ce projet vise à centraliser, nettoyer et analyser les avis Google Maps sur les agences bancaires marocaines, afin d’extraire des insights utiles pour les décideurs : tendances de satisfaction, thématiques récurrentes, classement des agences, etc. Le pipeline s’appuie sur une stack moderne et automatisée.

---

## Spécifications et Contexte
- **Sujet** : Centraliser et valoriser les avis clients Google Maps pour les banques marocaines.
- **Objectifs principaux** :
  1. Analyser la satisfaction client (sentiment analysis)
  2. Extraire les sujets/points récurrents (topic modeling)
  3. Classer les agences selon la perception client
  4. Fournir des dashboards interactifs pour la prise de décision
- **Livrables** : Scripts Python, DAGs Airflow, modèles DBT, schéma PostgreSQL, dashboards Looker Studio, documentation complète.

---

## Stack Technique et Versions
| Composant      | Version      |
|---------------|-------------|
| Python        | 3.8.10      |
| DBT           | 1.8.7       |
| Airflow       | 2.8.1       |
| NLP           | bert, nltk  |
| PostgreSQL    | >=12        |
| BI            | Looker Studio |

---

## Pipeline & Démarche détaillée

### 1. Collecte des données (Scraping)
- **Script principal** : `airflow/dags/scrapping.py`
- **Outils** : Playwright, BeautifulSoup, pandas
- **Logique** :
  - Scraping automatisé des avis Google Maps pour chaque banque/ville
  - Gestion des popups, scrolling, extraction des détails agences et avis
  - Stockage des données brutes en JSON/CSV dans `data/`
  - Possibilité de planifier le scraping via un DAG Airflow

### 2. Ingestion dans la base (Staging)
- **Script** : `scripts/load_to_staging.py`
- **Outils** : pandas, psycopg2
- **Logique** :
  - Chargement des fichiers CSV dans la table PostgreSQL `avis_bancaires`
  - Création dynamique de la table selon les colonnes du CSV
  - Gestion des connexions et erreurs

### 3. Nettoyage et enrichissement sémantique
- **Script** : `scripts/enrich_reviews.py`
- **Outils** : pandas, transformers, nltk, spacy, torch, datasets
- **Logique** :
  - Nettoyage textuel (ponctuation, stopwords, etc.)
  - Détection de la langue
  - Analyse de sentiment (BERT, pipeline transformers)
  - Extraction de topics bancaires (pattern matching, NLP)
  - Génération d’un DataFrame enrichi, stocké dans la table `stg_avis_bancaires_enriched`

### 4. Transformation et modélisation (DBT)
- **Répertoire** : `projet_dbt/`
- **Outils** : dbt-core
- **Logique** :
  - Modèles de staging : nettoyage, enrichissement, typage des données (`models/staging/`)
  - Modèles de marts : création d’un schéma en étoile (table de faits `fact_reviews`, dimensions `dim_bank`, `dim_branch`, etc.)
  - Configuration des sources et cibles dans `dbt_project.yml` et `models/sources/sources.yml`

### 5. Visualisation et Dashboards
- **Répertoire** : `dashboards/`
- **Outils** : Looker Studio (Google Data Studio), fichiers PNG
- **Logique** :
  - Création de dashboards interactifs pour explorer :
    - Les tendances de sentiment par banque et agence
    - Les topics (sujets) les plus fréquents
    - Le classement des agences
    - Les points forts/faibles récurrents
  - Export des résultats sous forme de graphiques et visuels dans le dossier `dashboards/`
  - Les dashboards sont accessibles via Looker Studio (liens à intégrer selon votre configuration)

### 6. Orchestration et automatisation
- **Répertoire** : `airflow/`
- **Outils** : Apache Airflow
- **Logique** :
  - Un DAG Airflow orchestre toutes les étapes : scraping, ingestion, enrichissement, transformations dbt
  - Gestion des dépendances, logs, planification quotidienne/hebdomadaire

### 6. Analyse et visualisation
- **Répertoire** : `dashboards/`
- **Outils** : Looker Studio (Google Data Studio)
- **Logique** :
  - Création de dashboards dynamiques : tendances de sentiment, topics majeurs, classement des agences, etc.
  - Export des résultats sous forme de graphiques PNG

---

## Arborescence du projet
```
DW_projet/
├── airflow/           # Orchestration Airflow (DAGs, logs)
├── dashboards/        # Visualisations (ex: PNG)
├── data/              # Données brutes (CSV, JSON)
├── projet_dbt/        # Projet dbt (modèles, sources, macros, ...)
├── scripts/           # Scripts Python (ETL, enrichissement)
```

---

## Prérequis & Installation
- Python 3.8.10
- DBT 1.8.7
- Airflow 2.8.1
- PostgreSQL >=12

### Installation des dépendances Python
```bash
pip install -r requirements.txt
```

### Installation d'Airflow (recommandé en environnement virtuel)
```bash
# Créer et activer un environnement virtuel (optionnel mais recommandé)
python -m venv venv
# Sous Windows :
venv\Scripts\activate
# Sous Linux/Mac :
source venv/bin/activate

# Installer Airflow (la version est déjà dans requirements.txt, mais voici la commande explicite)
pip install apache-airflow==2.8.1
```

### Installation de Playwright et initialisation
```bash
pip install playwright
playwright install
```

---

## Exécution du pipeline (étape par étape)
1. **Scraping et collecte des avis**
   > ⚠️ La collecte se fait exclusivement par scraping web avec Playwright et BeautifulSoup (pas d’API Google Maps ni de Scrapy).
   ```bash
   python airflow/dags/scrapping.py
   # ou planifier via Airflow
   ```
2. **Chargement dans PostgreSQL**
   ```bash
   python scripts/load_to_staging.py
   ```
3. **Enrichissement sémantique**
   ```bash
   python scripts/enrich_reviews.py
   ```
4. **Transformation et modélisation dbt**
   ```bash
   cd projet_dbt
   dbt run
   dbt test
   ```
5. **Orchestration complète**
   - Lancer le DAG Airflow pour automatiser tout le pipeline

---

## Explications pédagogiques et bonnes pratiques
- **Scraping** : Utiliser des délais aléatoires et gérer les popups pour éviter le blocage par Google.
- **Nettoyage** : Toujours vérifier la qualité des données avant l’analyse NLP.
- **NLP** : Choisir des modèles adaptés à la langue (français/anglais), ajuster les patterns pour le domaine bancaire.
- **DBT** : Versionner les modèles, documenter chaque transformation, tester les modèles (`dbt test`).
- **Airflow** : Monitorer les DAGs, prévoir des alertes en cas d’échec.
- **Sécurité** : Ne jamais commiter de mots de passe ou credentials dans le code source.

---

## Auteurs
Projet réalisé par KAJJOUT FATIMA ZOHRA 

---

## Pour aller plus loin
- Ajouter des sources d’avis complémentaires (Trustpilot, Facebook…)
- Tester des modèles NLP plus avancés (LDA, CamemBERT…)
- Intégrer des alertes automatisées (email, Slack) via Airflow
- Déployer sur le cloud (GCP, AWS) pour mise à l’échelle

---

## Contact
Pour toute question, contactez : fatimakajjout7@gmail.com
