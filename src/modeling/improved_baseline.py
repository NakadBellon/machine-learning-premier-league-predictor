"""
Version corrigée avec seulement les features PRÉDICTIVES (pas les scores réels)
"""
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
from datetime import datetime
import os

class CorrectedBaselineModel:
    """Utilise seulement les features disponibles AVANT le match"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
        }
        self.imputer = SimpleImputer(strategy='median')
        
    def load_data(self):
        """Charge les données processed les plus récentes"""
        try:
            processed_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
            processed_dir = os.path.abspath(processed_dir)
            
            files = [f for f in os.listdir(processed_dir) 
                    if f.startswith('premier_league_processed') and f.endswith('.csv')]
            
            if not files:
                raise FileNotFoundError("Aucun fichier processed trouvé")
                
            latest_file = sorted(files)[-1]
            file_path = os.path.join(processed_dir, latest_file)
            
            self.logger.info(f"Chargement des données: {latest_file}")
            df = pd.read_csv(file_path)
            
            self.logger.info(f"✅ Données chargées: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement données: {e}")
            raise
    
    def prepare_features_target(self, df):
        """Prépare les features PRÉDICTIVES uniquement"""
        self.logger.info("Préparation des features PRÉDICTIVES...")
        
        # ⚠️ NE PAS UTILISER les scores réels (home_score, away_score) 
        # car ils ne sont pas disponibles avant le match
        
        # Features PRÉDICTIVES (disponibles avant le match)
        predictive_features = ['home_xg', 'away_xg']  # xG des matchs précédents
        
        # Vérifier la disponibilité
        available_features = [col for col in predictive_features if col in df.columns]
        
        if not available_features:
            self.logger.warning("Aucune feature prédictive trouvée, recherche de features alternatives...")
            # Chercher d'autres features potentiellement prédictives
            potential_features = [col for col in df.columns if any(x in col for x in ['form', 'last', 'average', 'rating'])]
            available_features = potential_features[:4]  # Prendre max 4 features
        
        self.logger.info(f"Features prédictives utilisées: {available_features}")
        
        # Conversion numérique
        for col in available_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Nettoyage
        initial_count = len(df)
        df_clean = df.dropna(subset=available_features + ['result'])
        final_count = len(df_clean)
        
        self.logger.info(f"Données après nettoyage: {final_count}/{initial_count} ({final_count/initial_count*100:.1f}%)")
        
        if final_count < 1000:
            self.logger.warning("⚠️ Peu de données après nettoyage, résultats peu fiables")
        
        X = df_clean[available_features]
        y = df_clean['result']
        
        # Imputation pour les valeurs manquantes restantes
        X_imputed = self.imputer.fit_transform(X)
        
        self.logger.info(f"✅ Features shape: {X_imputed.shape}")
        self.logger.info(f"🎯 Target distribution: {y.value_counts().to_dict()}")
        
        return X_imputed, y, df_clean, available_features
    
    def calculate_baseline_accuracy(self, y):
        """Calcule l'accuracy baseline (prédire toujours le résultat le plus fréquent)"""
        most_common = y.value_counts().index[0]
        baseline_pred = [most_common] * len(y)
        baseline_accuracy = accuracy_score(y, baseline_pred)
        return baseline_accuracy
    
    def train_models(self):
        """Entraîne et évalue les modèles"""
        self.logger.info("Début entraînement modèles CORRIGÉS...")
        
        df = self.load_data()
        X, y, df_clean, feature_names = self.prepare_features_target(df)
        
        if X.shape[0] < 100:
            self.logger.error("Pas assez de données après nettoyage")
            return
        
        # Accuracy baseline
        baseline_accuracy = self.calculate_baseline_accuracy(y)
        self.logger.info(f"📈 Accuracy baseline (prédire toujours 'H'): {baseline_accuracy:.4f}")
        
        # Validation temporelle
        tscv = TimeSeriesSplit(n_splits=3)
        
        results = {}
        
        for model_name, model in self.models.items():
            self.logger.info(f"🤖 Entraînement {model_name}...")
            
            with mlflow.start_run(run_name=f"corrected_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                fold_accuracies = []
                fold_predictions = []
                fold_true = []
                
                for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    # Entraînement
                    model.fit(X_train, y_train)
                    
                    # Prédiction et évaluation
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    fold_accuracies.append(accuracy)
                    
                    # Stocker pour analyse
                    fold_predictions.extend(y_pred)
                    fold_true.extend(y_test)
                    
                    self.logger.info(f"📊 Fold {fold+1} - Accuracy: {accuracy:.4f}")
                    mlflow.log_metric(f"fold_{fold+1}_accuracy", accuracy)
                
                # Métriques finales
                mean_accuracy = np.mean(fold_accuracies)
                std_accuracy = np.std(fold_accuracies)
                
                # Rapport de classification détaillé
                class_report = classification_report(fold_true, fold_predictions, output_dict=True)
                
                mlflow.log_metric("mean_accuracy", mean_accuracy)
                mlflow.log_metric("std_accuracy", std_accuracy)
                mlflow.log_metric("baseline_accuracy", baseline_accuracy)
                mlflow.log_metric("improvement_over_baseline", mean_accuracy - baseline_accuracy)
                
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("n_features", X.shape[1])
                mlflow.log_param("feature_names", str(feature_names))
                
                # Loguer le rapport de classification
                for class_name, metrics in class_report.items():
                    if class_name in ['H', 'A', 'D']:
                        mlflow.log_metric(f"precision_{class_name}", metrics['precision'])
                        mlflow.log_metric(f"recall_{class_name}", metrics['recall'])
                        mlflow.log_metric(f"f1_{class_name}", metrics['f1-score'])
                
                mlflow.sklearn.log_model(model, model_name)
                
                results[model_name] = {
                    'mean_accuracy': mean_accuracy,
                    'std_accuracy': std_accuracy,
                    'baseline_accuracy': baseline_accuracy,
                    'improvement': mean_accuracy - baseline_accuracy,
                    'fold_accuracies': fold_accuracies,
                    'classification_report': class_report
                }
                
                self.logger.info(f"✅ {model_name} - Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
                self.logger.info(f"📈 Amélioration vs baseline: {mean_accuracy - baseline_accuracy:.4f}")
        
        return results

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 DÉMARRAGE MODÈLES CORRIGÉS (Features PRÉDICTIVES uniquement)")
        
        # MLflow setup
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        mlflow_tracking_uri = "file:///" + os.path.abspath(os.path.join(project_root, "mlruns")).replace("\\", "/")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("Premier_League_Corrected_Models")
        
        # Entraînement
        model = CorrectedBaselineModel()
        results = model.train_models()
        
        logger.info("✅ MODÈLES CORRIGÉS TERMINÉS")
        
        if results:
            best_model = max(results.items(), key=lambda x: x[1]['mean_accuracy'])
            best_acc = best_model[1]['mean_accuracy']
            baseline = best_model[1]['baseline_accuracy']
            improvement = best_model[1]['improvement']
            
            logger.info(f"🏆 MEILLEUR MODÈLE: {best_model[0]}")
            logger.info(f"🎯 Accuracy: {best_acc:.4f}")
            logger.info(f"📈 Baseline: {baseline:.4f}")
            logger.info(f"💪 Amélioration: +{improvement:.4f}")
            
            if improvement > 0:
                logger.info("✅ Le modèle bat la baseline !")
            else:
                logger.info("⚠️  Le modèle ne bat pas la baseline")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ ERREUR: {e}")
        raise

if __name__ == "__main__":
    main()