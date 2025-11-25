"""
Modèles avancés avec les nouvelles features de forme
"""
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from datetime import datetime
import os

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

class AdvancedModelsWithFeatures:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        
        self.models = {}
        if XGB_AVAILABLE:
            self.models['xgb'] = XGBClassifier(n_estimators=200, max_depth=6, random_state=42)
        if LGBM_AVAILABLE:
            self.models['lgbm'] = LGBMClassifier(n_estimators=200, max_depth=6, random_state=42, verbose=-1)

    def load_data(self):
        """Charge les données avec features"""
        try:
            processed_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
            processed_dir = os.path.abspath(processed_dir)
            
            # Chercher le fichier avec features
            files = [f for f in os.listdir(processed_dir)
                    if f.startswith('premier_league_with_features') and f.endswith('.csv')]
            
            if not files:
                raise FileNotFoundError("Aucun fichier avec features trouvé")
                
            latest_file = sorted(files)[-1]
            file_path = os.path.join(processed_dir, latest_file)
            
            self.logger.info(f"Chargement des données avec features: {latest_file}")
            df = pd.read_csv(file_path)
            
            self.logger.info(f"✅ Données chargées: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement données: {e}")
            raise

    def prepare_features_target(self, df):
        """Prépare les features avancées"""
        self.logger.info("Préparation des features avancées...")
        
        # Toutes les features numériques sauf les scores réels
        exclude_features = ['home_score', 'away_score', 'result', 'date', 'season', 
                           'home_team', 'away_team', 'venue', 'referee', 'match_report', 
                           'notes', 'game_id', 'week', 'day', 'time', 'attendance', 'score']
        
        # Features disponibles
        all_features = [col for col in df.columns if col not in exclude_features]
        
        self.logger.info(f"Features utilisées ({len(all_features)}): {all_features}")
        
        # Conversion numérique
        for col in all_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Nettoyage
        initial_count = len(df)
        df_clean = df.dropna(subset=all_features + ['result'])
        final_count = len(df_clean)
        
        self.logger.info(f"Données après nettoyage: {final_count}/{initial_count} ({final_count/initial_count*100:.1f}%)")
        
        X = df_clean[all_features]
        y = df_clean['result']
        
        # Imputation et scaling
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        self.logger.info(f"✅ Features shape: {X_scaled.shape}")
        self.logger.info(f"🎯 Target distribution: {y.value_counts().to_dict()}")
        
        return X_scaled, y, df_clean, all_features

    def encode_labels(self, y):
        """Encode les labels textuels en numériques"""
        label_mapping = {'H': 0, 'D': 1, 'A': 2}
        reverse_mapping = {0: 'H', 1: 'D', 2: 'A'}
        y_encoded = y.map(label_mapping)
        return y_encoded, label_mapping, reverse_mapping

    def decode_predictions(self, y_pred_encoded, reverse_mapping):
        return [reverse_mapping[pred] for pred in y_pred_encoded]

    def train_models(self):
        """Entraîne les modèles avec les nouvelles features"""
        self.logger.info("Début entraînement avec NOUVELLES FEATURES...")
        
        df = self.load_data()
        X, y, df_clean, feature_names = self.prepare_features_target(df)
        
        # Encodage
        y_encoded, label_mapping, reverse_mapping = self.encode_labels(y)
        
        baseline_accuracy = self.calculate_baseline_accuracy(y_encoded)
        self.logger.info(f"📈 Accuracy baseline: {baseline_accuracy:.4f}")
        
        tscv = TimeSeriesSplit(n_splits=3)
        results = {}
        
        for model_name, model in self.models.items():
            self.logger.info(f"🤖 Entraînement {model_name.upper()}...")
            
            with mlflow.start_run(run_name=f"with_features_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                fold_accuracies = []
                
                for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y_encoded.iloc[train_idx], y_encoded.iloc[test_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred_encoded = model.predict(X_test)
                    y_pred = self.decode_predictions(y_pred_encoded, reverse_mapping)
                    y_true_decoded = self.decode_predictions(y_test, reverse_mapping)
                    
                    accuracy = accuracy_score(y_true_decoded, y_pred)
                    fold_accuracies.append(accuracy)
                    self.logger.info(f"📊 Fold {fold+1} - Accuracy: {accuracy:.4f}")
                
                mean_accuracy = np.mean(fold_accuracies)
                self.logger.info(f"✅ {model_name} - Accuracy: {mean_accuracy:.4f}")
                self.logger.info(f"📈 Amélioration vs baseline: {mean_accuracy - baseline_accuracy:.4f}")
                
                results[model_name] = {
                    'mean_accuracy': mean_accuracy,
                    'baseline_accuracy': baseline_accuracy,
                    'improvement': mean_accuracy - baseline_accuracy
                }
        
        return results

    def calculate_baseline_accuracy(self, y_encoded):
        most_common = y_encoded.value_counts().index[0]
        baseline_pred = [most_common] * len(y_encoded)
        return accuracy_score(y_encoded, baseline_pred)

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 TEST AVEC NOUVELLES FEATURES")
        
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        mlflow_tracking_uri = "file:///" + os.path.abspath(os.path.join(project_root, "mlruns")).replace("\\", "/")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("Premier_League_With_Features")
        
        advanced_models = AdvancedModelsWithFeatures()
        results = advanced_models.train_models()
        
        if results:
            best_model = max(results.items(), key=lambda x: x[1]['mean_accuracy'])
            best_name, best_result = best_model
            
            logger.info(f"\n🏆 MEILLEUR MODÈLE: {best_name.upper()}")
            logger.info(f"🎯 Accuracy: {best_result['mean_accuracy']:.4f}")
            logger.info(f"💪 Amélioration vs baseline: +{best_result['improvement']:.4f}")
            
            # Comparaison avec ancien baseline (60.5%)
            improvement_vs_old = best_result['mean_accuracy'] - 0.6051
            logger.info(f"📊 vs ancien modèle (60.5%): {improvement_vs_old:+.4f}")
            
            if improvement_vs_old > 0:
                logger.info("🎉 MEILLEUR que l'ancien modèle !")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ ERREUR: {e}")
        raise

if __name__ == "__main__":
    main()