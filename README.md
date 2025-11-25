# Premier League Predictor - Documentation Complète

## Table des Matières
- [Description du Projet](#description-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Architecture](#architecture)
- [Modèles](#modèles)
- [MLOps](#mlops)
- [API](#api)
- [Déploiement](#déploiement)
- [Développement](#développement)

## Description du Projet

### Objectif Principal
Système complet de prédiction des matchs de Premier League anglaise utilisant le Machine Learning et les bonnes pratiques MLOps.

### Fonctionnalités Principales
- **Prédiction de matchs individuels** avec régression logistique (60.58% accuracy)
- **Simulation de saison complète** par méthode Monte Carlo (10,000 simulations)
- **Tracking MLOps** avec MLflow pour l'expérimentation
- **Containerisation** Docker pour le déploiement
- **Interface utilisateur** Streamlit interactive

### Performance du Modèle
| Métrique | Valeur |
|----------|--------|
| Accuracy | 60.58% |
| Baseline | 43.55% |
| Amélioration | +17.03% |
| Matchs analysés | 15,960 |
| Saisons couvertes | 7 (2019-2026) |

## Installation

### Prérequis
- Python 3.9+
- Docker (optionnel)
- Git

### Méthode 1 : Environnement Conda
```bash
conda create -n premier_league_env python=3.9
conda activate premier_league_env
pip install -r requirements.txt
```

### Méthode 2 : Docker
```bash
docker build -t premier-league-predictor .
docker run -p 7860:7860 premier-league-predictor
```

### Dépendances Principales
```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.2.0
matplotlib>=3.5.0
seaborn>=0.12.0
mlflow>=2.3.0
xgboost>=1.7.0
lightgbm>=3.3.0
```

## Utilisation

### Lancement de l'Application
```bash
streamlit run app/complete_dashboard.py
```

### Navigation dans l'Interface
1. **Accueil** : Vue d'ensemble et métriques
2. **Prédire un Match** : Analyse match par match
3. **Vue Saison** : Simulations Monte Carlo
4. **Comparaisons** : Analyse des méthodes

### Prédiction d'un Match
- Sélection automatique des xG depuis les données historiques
- Calcul des probabilités en temps réel
- Analyse contextuelle des enjeux
- Visualisations graphiques interactives

### Simulation de Saison
- 10,000 simulations Monte Carlo
- Probabilités de titre, top 4, relégation
- Classements prédictifs
- Analyses de risque

## Architecture

### Structure des Fichiers
```
premier_league_predictor/
├── app/
│   └── complete_dashboard.py
├── src/
│   ├── data_scraping/
│   │   ├── scraper.py
│   │   ├── cleaner.py
│   │   └── pipeline.py
│   ├── modeling/
│   │   ├── improved_baseline.py
│   │   ├── advanced_models.py
│   │   ├── optimized_logistic.py
│   │   └── monte_carlo_simulator.py
│   └── monitoring/
│       ├── mlflow_setup.py
│       └── mlflow_logger.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── mlruns/
├── deployment_data/
├── scripts/
│   └── prepare_deployment.py
├── Dockerfile
├── requirements.txt
└── README.md
```

### Flux de Données
1. **Collecte** : Scraping FBref → Données brutes
2. **Nettoyage** : Parsing scores, gestion valeurs manquantes
3. **Features** : Calcul forme équipes, statistiques temporelles
4. **Entraînement** : Modèles ML avec validation temporelle
5. **Prédiction** : Interface utilisateur + API

## Modèles

### Régression Logistique (Meilleur Modèle)
- **Accuracy** : 60.58%
- **Features** : home_xg, away_xg
- **Validation** : Time Series Cross-Validation (3 folds)
- **Regularisation** : L2 avec C=0.1

### Modèles Comparés
| Modèle | Accuracy | Statut |
|--------|----------|--------|
| Logistic Regression | 60.58% | 🏆 Meilleur |
| XGBoost | 52.46% | 📊 Bon |
| LightGBM | 52.40% | 📊 Bon |
| Voting Classifier | 52.92% | 📊 Bon |

### Features Utilisées
- **xG domicile** : Expected Goals équipe à domicile
- **xG extérieur** : Expected Goals équipe à l'extérieur
- **Forme récente** : Points/buts 5 derniers matchs
- **Variables temporelles** : Mois, jour de la semaine

## MLOps

### MLflow Tracking
```python
import mlflow

mlflow.set_experiment("Premier_League_Prediction")
with mlflow.start_run():
    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_metric("accuracy", 0.6058)
    mlflow.sklearn.log_model(model, "model")
```

### Métriques Trackées
- Accuracy par fold
- Precision/Recall par classe
- Importance des features
- Matrices de confusion

### Versioning des Données
- DVC configuré avec Google Drive
- Pipeline de données reproductible
- Historique des jeux de données

## API

### Structure API FastAPI (Planifiée)
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class MatchPredictionRequest(BaseModel):
    home_team: str
    away_team: str
    home_xg: float
    away_xg: float

@app.post("/predict/match")
async def predict_match(request: MatchPredictionRequest):
    # Implémentation de la prédiction
    return {"prediction": "H", "probabilities": {"H": 0.52, "D": 0.25, "A": 0.23}}
```

### Endpoints Prévisionnels
- `POST /predict/match` : Prédiction match unique
- `GET /simulate/season` : Simulation saison complète
- `GET /teams/{team}/stats` : Statistiques équipe
- `GET /models/performance` : Métriques modèles

## Déploiement

### Configuration Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["streamlit", "run", "app/complete_dashboard.py", "--server.port=7860", "--server.address=0.0.0.0"]
```

### Hugging Face Spaces
- SDK : Docker
- Port : 7860
- Build automatique sur push
- Documentation automatique

### Commandes de Déploiement
```bash
# Build local
docker build -t premier-league-predictor .

# Déploiement HF
git add .
git commit -m "Deploy to HF"
git push huggingface main
```

## Développement

### Workflow de Développement
1. **Expérimentation** : Notebooks → Scripts ML
2. **Tracking** : MLflow pour métriques
3. **Validation** : Tests unitaires + validation croisée
4. **Packaging** : Docker + requirements
5. **Déploiement** : Hugging Face Spaces

### Tests et Validation
```bash
# Tests données
python -m src.data_scraping.pipeline

# Tests modèles
python src/modeling/optimized_logistic.py

# Validation complète
python src/modeling/monte_carlo_simulator.py
```

### Qualité de Code
- Formatage : Black
- Import sorting : isort
- Linting : Flake8
- CI/CD : GitHub Actions

## Résultats et Analyses

### Simulations Saison 2025-2026
| Équipe | Titre | Top 4 | Relégation |
|--------|-------|-------|------------|
| Manchester City | 76.4% | 99.9% | 0.0% |
| Liverpool | 19.4% | 97.6% | 0.0% |
| Arsenal | 2.5% | 77.2% | 0.0% |
| Chelsea | 1.3% | 75.0% | 0.0% |
| Luton Town | 0.0% | 0.0% | 99.9% |

### Insights Clés
- **Avantage domicile** significatif dans les prédictions
- **xG** meilleur prédicteur que les résultats bruts
- **Forme récente** améliore légèrement les performances
- **Régression logistique** plus robuste que modèles complexes

## Améliorations Futures

### Court Terme
- [ ] API FastAPI complète
- [ ] Déploiement Hugging Face Spaces
- [ ] Documentation technique étendue

### Moyen Terme
- [ ] Intégration données temps réel
- [ ] Features additionnelles (blessures, compositions)
- [ ] Monitoring performance en production

### Long Terme
- [ ] Modèles deep learning
- [ ] Prédictions en temps réel
- [ ] Scaling cloud multi-région

## Support et Contact

### Documentation Additionnelle
- Documentation MLflow : `/mlruns`
- Données historiques : `/data/processed`
- Modèles entraînés : `/models`

### Dépannage
- Problèmes Docker : vérifier installation WSL2
- Erreurs données : exécuter `scripts/prepare_deployment.py`
- Problèmes modèles : vérifier versions dépendances

## Licence et Contribution

### Licence
Projet sous licence MIT - libre usage et modification.

### Contribution
1. Fork du repository
2. Branche feature dédiée
3. Tests et validation
4. Pull request documentée

### Standards de Code
- PEP8 compliance
- Docstrings complètes
- Tests unitaires
- Validation des données

---

*Dernière mise à jour : Novembre 2024*