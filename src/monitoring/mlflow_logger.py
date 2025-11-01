"""Logger avancé pour MLflow avec des fonctions utilitaires."""

import mlflow
import pandas as pd
from datetime import datetime
import json

class MLflowLogger:
    def __init__(self, experiment_name="Premier_League_Predictor"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name=None, tags=None):
        """Démarre une nouvelle run MLflow."""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run = mlflow.start_run(run_name=run_name)
        
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        
        return self.run
    
    def log_params(self, params):
        """Logue un dictionnaire de paramètres."""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics, step=None):
        """Logue un dictionnaire de métriques."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_dataset_info(self, dataset_name, shape, columns=None):
        """Logue des informations sur le dataset."""
        mlflow.log_param(f"{dataset_name}_shape", shape)
        if columns:
            mlflow.log_param(f"{dataset_name}_columns", json.dumps(columns))
    
    def log_model(self, model, model_name):
        """Logue un modèle (à implémenter plus tard)."""
        print(f" Modèle {model_name} prêt à être logué")
        # mlflow.sklearn.log_model(model, model_name)  # Décommenter plus tard

# Exemple d'utilisation
if __name__ == "__main__":
    logger = MLflowLogger()
    
    with logger.start_run(run_name="test_advanced", tags={"env": "testing", "phase": "4"}):
        # Loguer des paramètres
        params = {
            "model_type": "random_forest",
            "n_estimators": 100,
            "max_depth": 10
        }
        logger.log_params(params)
        
        # Loguer des métriques
        metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.78,
            "f1_score": 0.80
        }
        logger.log_metrics(metrics)
        
        # Loguer des infos dataset
        logger.log_dataset_info("training_data", (1000, 20))
        
        print("Test du logger avancé réussi!")