"""
Modèle baseline pour la prédiction des matchs de Premier League
"""
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import mlflow
import mlflow.sklearn
from datetime import datetime
import os
import sys

class BaselineModel:
    """Classe pour les modèles baseline"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
    def load_data(self):
        """Charge les données processed les plus récentes"""
        try:
            # Chemin absolu vers le dossier processed
            processed_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
            processed_dir = os.path.abspath(processed_dir)
            
            self.logger.info(f"Recherche dans: {processed_dir}")
            
            # Vérifier si le dossier existe
            if not os.path.exists(processed_dir):
                raise FileNotFoundError(f"Dossier processed non trouvé: {processed_dir}")
            
            # Trouver le fichier processed le plus récent
            files = [f for f in os.listdir(processed_dir) 
                    if f.startswith('premier_league_processed') and f.endswith('.csv')]
            
            self.logger.info(f"Fichiers trouvés: {files}")
            
            if not files:
                raise FileNotFoundError("Aucun fichier processed trouvé")
                
            latest_file = sorted(files)[-1]
            file_path = os.path.join(processed_dir, latest_file)
            
            self.logger.info(f"Chargement des données: {latest_file}")
            df = pd.read_csv(file_path)
            
            self.logger.info(f"✅ Données chargées: {df.shape}")
            self.logger.info(f"Colonnes disponibles: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement données: {e}")
            raise
    
    def prepare_features_target(self, df):
        """Prépare les features et la target"""
        self.logger.info("Préparation des features et target...")
    
        # Vérifier les colonnes disponibles
        available_cols = df.columns.tolist()
        self.logger.info(f"Colonnes disponibles: {available_cols}")
    
        # Features de base (à adapter selon tes données)
        feature_columns = []
    
        # Ajouter les features de forme si elles existent
        form_features = [col for col in df.columns if 'last_5_' in col]
        feature_columns.extend(form_features)
        
        # Si pas de features de forme, utiliser des features basiques
        if not feature_columns:
            self.logger.warning("Aucune feature de forme trouvée, utilisation features basiques")
            # Features basiques alternatives - EXCLURE 'score' qui est un string
            basic_features = [col for col in df.columns if any(x in col for x in ['xg', 'goals'])]
            feature_columns.extend(basic_features)
        
        # Vérifier et convertir les types de données
        for col in feature_columns:
            if df[col].dtype == 'object':
                self.logger.info(f"Conversion de {col} en numérique...")
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Vérifier la target
        if 'result' not in df.columns:
            raise ValueError("Colonne 'result' manquante pour la target")
        
        # Nettoyer les données
        df_clean = df.dropna(subset=feature_columns + ['result'])
        
        X = df_clean[feature_columns]
        y = df_clean['result']
        
        self.logger.info(f"✅ Features shape: {X.shape}")
        self.logger.info(f"🎯 Target distribution: {y.value_counts().to_dict()}")
        self.logger.info(f"📊 Features utilisées: {feature_columns}")
        self.logger.info(f"🔧 Types des features: {X.dtypes.to_dict()}")
        
        return X, y, df_clean
    
    def time_series_split(self, X, y, n_splits=3):
        """Validation croisée temporelle"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return tscv.split(X), tscv
    
    def train_baseline_models(self):
        """Entraîne et évalue les modèles baseline"""
        self.logger.info("Début entraînement modèles baseline...")
        
        # Charger les données
        df = self.load_data()
        
        # Préparer features et target
        X, y, df_clean = self.prepare_features_target(df)
        
        if X.empty:
            self.logger.error("Aucune donnée après nettoyage")
            return
        
        # Validation temporelle
        splits, tscv = self.time_series_split(X, y)
        
        results = {}
        
        for model_name, model in self.models.items():
            self.logger.info(f"🤖 Entraînement {model_name}...")
            
            with mlflow.start_run(run_name=f"baseline_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Loguer les paramètres
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("n_features", X.shape[1])
                mlflow.log_param("n_samples", X.shape[0])
                mlflow.log_param("features", str(X.columns.tolist()))
                
                # Validation croisée
                fold_accuracies = []
                
                for fold, (train_idx, test_idx) in enumerate(splits):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    # Entraînement
                    model.fit(X_train, y_train)
                    
                    # Prédiction
                    y_pred = model.predict(X_test)
                    
                    # Évaluation
                    accuracy = accuracy_score(y_test, y_pred)
                    fold_accuracies.append(accuracy)
                    
                    self.logger.info(f"📊 Fold {fold+1} - Accuracy: {accuracy:.4f}")
                    
                    # Loguer les métriques par fold
                    mlflow.log_metric(f"fold_{fold+1}_accuracy", accuracy)
                
                # Métriques globales
                mean_accuracy = np.mean(fold_accuracies)
                std_accuracy = np.std(fold_accuracies)
                
                # Loguer les résultats
                mlflow.log_metric("mean_accuracy", mean_accuracy)
                mlflow.log_metric("std_accuracy", std_accuracy)
                
                # Loguer le modèle
                mlflow.sklearn.log_model(model, f"baseline_{model_name}")
                
                # Sauvegarder les résultats
                results[model_name] = {
                    'mean_accuracy': mean_accuracy,
                    'std_accuracy': std_accuracy,
                    'fold_accuracies': fold_accuracies,
                    'model': model,
                    'feature_names': X.columns.tolist()
                }
                
                self.logger.info(f"✅ {model_name} - Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        
        return results

def main():
    """Fonction principale"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 DÉMARRAGE MODÈLES BASELINE")
        
        # Configurer MLflow
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        mlflow_tracking_uri = "file:///" + os.path.abspath(os.path.join(project_root, "mlruns")).replace("\\", "/")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("Premier_League_Baseline_Models")
        
        logger.info(f"MLflow Tracking URI: {mlflow_tracking_uri}")
        
        # Entraîner les modèles
        baseline_model = BaselineModel()
        results = baseline_model.train_baseline_models()
        
        logger.info("✅ MODÈLES BASELINE TERMINÉS")
        
        # Afficher le meilleur modèle
        if results:
            best_model = max(results.items(), key=lambda x: x[1]['mean_accuracy'])
            logger.info(f"🏆 MEILLEUR MODÈLE: {best_model[0]} avec accuracy: {best_model[1]['mean_accuracy']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ ERREUR: {e}")
        raise

if __name__ == "__main__":
    main()