"""
Optimisation de la régression logistique avec les nouvelles features
"""
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from datetime import datetime
import os

class OptimizedLogistic:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Charge les données avec features"""
        processed_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
        processed_dir = os.path.abspath(processed_dir)
        
        files = [f for f in os.listdir(processed_dir)
                if f.startswith('premier_league_with_features') and f.endswith('.csv')]
        
        latest_file = sorted(files)[-1]
        file_path = os.path.join(processed_dir, latest_file)
        
        self.logger.info(f"Chargement: {latest_file}")
        return pd.read_csv(file_path)

    def prepare_features_target(self, df, use_simple_features=True):
        """Prépare les features - option simple ou avancée"""
        
        if use_simple_features:
            # Utiliser seulement les 2 meilleures features (comme le baseline)
            features = ['home_xg', 'away_xg']
        else:
            # Utiliser toutes les features mais avec sélection
            exclude = ['home_score', 'away_score', 'result', 'date', 'season', 
                      'home_team', 'away_team', 'venue', 'referee', 'match_report', 
                      'notes', 'game_id', 'week', 'day', 'time', 'attendance', 'score']
            features = [col for col in df.columns if col not in exclude]
        
        self.logger.info(f"Features utilisées ({len(features)}): {features}")
        
        # Conversion numérique
        for col in features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df_clean = df.dropna(subset=features + ['result'])
        
        X = df_clean[features]
        y = df_clean['result']
        
        # Imputation et scaling
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        self.logger.info(f"✅ Features shape: {X_scaled.shape}")
        return X_scaled, y, df_clean, features

    def train_optimized_model(self):
        """Entraîne une régression logistique optimisée"""
        self.logger.info("🚀 RÉGRESSION LOGISTIQUE OPTIMISÉE")
        
        df = self.load_data()
        
        # Test avec différentes configurations
        results = {}
        
        # 1. Features simples (comme baseline original)
        X_simple, y_simple, _, features_simple = self.prepare_features_target(df, use_simple_features=True)
        results['simple'] = self._train_logistic(X_simple, y_simple, features_simple, "simple")
        
        # 2. Features avancées avec régularisation
        X_advanced, y_advanced, _, features_advanced = self.prepare_features_target(df, use_simple_features=False)
        results['advanced'] = self._train_logistic(X_advanced, y_advanced, features_advanced, "advanced")
        
        # 3. Features sélectionnées manuellement
        selected_features = ['home_xg', 'away_xg', 'home_last_5_points', 'away_last_5_points']
        X_selected = df[selected_features]
        y_selected = df['result']
        for col in selected_features:
            X_selected[col] = pd.to_numeric(X_selected[col], errors='coerce')
        X_selected = self.imputer.fit_transform(X_selected)
        X_selected = self.scaler.fit_transform(X_selected)
        results['selected'] = self._train_logistic(X_selected, y_selected, selected_features, "selected")
        
        return results

    def _train_logistic(self, X, y, feature_names, config_name):
        """Entraîne un modèle logistique"""
        tscv = TimeSeriesSplit(n_splits=3)
        fold_accuracies = []
        
        with mlflow.start_run(run_name=f"logistic_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Modèle avec régularisation
                model = LogisticRegression(
                    C=0.1,  # Forte régularisation
                    penalty='l2',
                    max_iter=1000,
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                fold_accuracies.append(accuracy)
                
                self.logger.info(f"📊 {config_name} - Fold {fold+1}: {accuracy:.4f}")
            
            mean_accuracy = np.mean(fold_accuracies)
            
            mlflow.log_metric("mean_accuracy", mean_accuracy)
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("feature_set", config_name)
            mlflow.log_param("features", str(feature_names))
            
            mlflow.sklearn.log_model(model, "model")
            
            return {
                'mean_accuracy': mean_accuracy,
                'fold_accuracies': fold_accuracies,
                'features': feature_names
            }

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        mlflow_tracking_uri = "file:///" + os.path.abspath(os.path.join(project_root, "mlruns")).replace("\\", "/")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("Premier_League_Optimized_Logistic")
        
        optimizer = OptimizedLogistic()
        results = optimizer.train_optimized_model()
        
        logger.info("\n" + "="*50)
        logger.info("📊 RÉSULTATS FINAUX")
        logger.info("="*50)
        
        for config, result in results.items():
            logger.info(f"{config.upper()}: {result['mean_accuracy']:.4f}")
        
        best_config = max(results.items(), key=lambda x: x[1]['mean_accuracy'])
        logger.info(f"\n🏆 MEILLEURE CONFIG: {best_config[0].upper()}")
        logger.info(f"🎯 Accuracy: {best_config[1]['mean_accuracy']:.4f}")
        
        # Comparaison avec baseline original
        improvement = best_config[1]['mean_accuracy'] - 0.6051
        logger.info(f"📈 vs baseline original: {improvement:+.4f}")
        
        if improvement > 0:
            logger.info("🎉 NOUVEAU RECORD !")
        
    except Exception as e:
        logger.error(f"❌ ERREUR: {e}")
        raise

if __name__ == "__main__":
    main()