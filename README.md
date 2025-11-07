# Premier League Predictor 🏆

Un système de prédiction des matchs de Premier League anglaise utilisant le machine learning.

## Objectif
Prédire les résultats des matchs de Premier League (victoire domicile/extérieur/match nul) basé sur les données historiques et les performances des équipes.

## Fonctionnalités

- **Scraping automatique** des données depuis FBref (2019-2026)
- **Nettoyage et feature engineering** des données
- **Tracking MLOps** avec MLflow
- **Modèles de machine learning** pour la prédiction

## Structure général

premier_league_predictor/
├── data/           # Données
├── src/            # Code source
├── models/         # Modèles ML
├── notebooks/      # Jupyter notebooks
├── app/            # Application Streamlit
└── mlops/          # Pipeline MLOps

## Structure du scraping

src/data_scraping/
├── pipeline.py # Pipeline principal
├── scraper.py # Scraping FBref
├── cleaner.py # Nettoyage des données
├── config.py # Configuration
└── features.py # Feature engineering

## Installation

```bash
# Clone le projet
git clone https://github.com/NakadBellon/machine-learning-premier-league-predictor.git
cd premier_league_predictor

# Crée l'environnement
python -m venv premier_league_env
source premier_league_env/bin/activate  # Linux/Mac
# OU
premier_league_env\Scripts\activate  # Windows

# Installe les dépendances
pip install -r requirements.txt
```

## Objectifs

- Scraping données Premier League
- Feature engineering temporel
- Modèles prédiction matchs
- Simulation saison Monte Carlo
- App Streamlit interactive
- Pipeline MLOps automatisé

## Données

- Période : Saisons 2019-2020 à 2025-2026
- Matchs : 15,960 matchs historiques
- Features : Scores, xG, forme des équipes, etc.

## Prochaines étapes

- Feature engineering avancé
- Entraînement des modèles
- Optimisation hyperparamètres
- Interface de prédiction
=======
# machine-learning-premier-league-predictor
Machine Learning Pipeline for Premier League Match Predictions
>>>>>>> c624cd5e1ead4099352eb8cf063ae2d07c3d7ac2
