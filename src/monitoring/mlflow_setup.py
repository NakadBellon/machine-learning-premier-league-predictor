"""Configuration et setup de MLflow pour le tracking des expériences."""

import mlflow
import os
from pathlib import Path
import logging

def setup_mlflow():
    """Configure MLflow pour le projet."""
    
    # Créer le dossier mlruns s'il n'existe pas
    mlruns_dir = Path("mlruns")
    mlruns_dir.mkdir(exist_ok=True)
    
    # Configurer l'URI de tracking (local pour le moment)
    mlflow.set_tracking_uri("mlruns/")
    
    # Créer l'expérience si elle n'existe pas
    experiment_name = "Premier_League_Predictor"
    
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Nouvelle expérience créée: {experiment_name} (ID: {experiment_id})")
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        print(f"Expérience existante chargée: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)
    
    print("MLflow configuré avec succès!")
    print(f"Dossier de tracking: {mlruns_dir.absolute()}")
    
    return experiment_id

def test_mlflow_tracking():
    """Test basique du tracking MLflow."""
    
    with mlflow.start_run(run_name="test_run"):
        # Loguer des paramètres
        mlflow.log_param("model_type", "test")
        mlflow.log_param("data_version", "1.0")
        
        # Loguer des métriques
        mlflow.log_metric("accuracy", 0.85)
        mlflow.log_metric("precision", 0.82)
        
        # Loguer un tag
        mlflow.set_tag("purpose", "testing")
        
        print("Test MLflow réussi! Données loguées avec succès.")

if __name__ == "__main__":
    setup_mlflow()
    test_mlflow_tracking()