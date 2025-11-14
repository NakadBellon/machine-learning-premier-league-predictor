"""
Chargeur de modèle depuis MLflow
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import os
import logging

class ModelLoader:
    """Charge le modèle entraîné depuis MLflow"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.feature_names = ['home_xg', 'away_xg']
        
    def load_latest_model(self):
        """Charge le dernier modèle depuis MLflow"""
        try:
            # Configuration MLflow
            mlflow_tracking_uri = "file:///" + os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlruns')).replace("\\", "/")
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            
            # Chercher la dernière run du modèle logistic regression
            experiment_name = "Premier_League_Corrected_Models"
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                self.logger.warning("Aucune expérience trouvée")
                return False
                
            # Récupérer les runs
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.mlflow.runName LIKE 'corrected_logistic_regression%'",
                order_by=["start_time DESC"]
            )
            
            if runs.empty:
                self.logger.warning("Aucun modèle logistic regression trouvé")
                return False
                
            # Prendre la dernière run
            latest_run = runs.iloc[0]
            run_id = latest_run['run_id']
            
            # Charger le modèle
            model_uri = f"runs:/{run_id}/logistic_regression"
            self.model = mlflow.sklearn.load_model(model_uri)
            
            # Récupérer les noms de features depuis les paramètres
            feature_param = latest_run.get('params.feature_names', '[]')
            if feature_param and feature_param != '[]':
                # Extraire les noms de features du string
                import ast
                self.feature_names = ast.literal_eval(feature_param)
            
            self.logger.info(f"✅ Modèle chargé: {run_id}")
            self.logger.info(f"📊 Features: {self.feature_names}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement modèle: {e}")
            return False
    
    def predict_match(self, home_xg, away_xg, home_form=None, away_form=None):
        """Prédit le résultat d'un match"""
        if self.model is None:
            raise ValueError("Modèle non chargé")
        
        # Préparer les features dans le bon ordre
        features = {}
        
        # Features de base
        features['home_xg'] = home_xg
        features['away_xg'] = away_xg
        
        # Ajouter d'autres features si disponibles
        if home_form is not None and 'home_last_5_points' in self.feature_names:
            features['home_last_5_points'] = home_form
        if away_form is not None and 'away_last_5_points' in self.feature_names:
            features['away_last_5_points'] = away_form
        
        # Créer le DataFrame dans le bon ordre
        feature_data = []
        for feature_name in self.feature_names:
            feature_data.append(features.get(feature_name, 0.0))
        
        X_pred = pd.DataFrame([feature_data], columns=self.feature_names)
        
        # Prédiction
        probabilities = self.model.predict_proba(X_pred)[0]
        prediction = self.model.predict(X_pred)[0]
        
        # Mapping des classes
        class_mapping = {0: 'H', 1: 'A', 2: 'D'} if len(self.model.classes_) == 3 else {'H': 'H', 'A': 'A', 'D': 'D'}
        
        # Organiser les probabilités
        prob_dict = {}
        for i, class_label in enumerate(self.model.classes_):
            prob_dict[class_mapping.get(class_label, class_label)] = probabilities[i]
        
        return prob_dict.get('H', 0), prob_dict.get('D', 0), prob_dict.get('A', 0), prediction

    def get_team_historical_stats(self, df, team_name, is_home=True):
        """Récupère les statistiques historiques d'une équipe"""
        if df is None:
            return 1.5, 6.0  # Valeurs par défaut
            
        try:
            if is_home:
                team_data = df[df['home_team'] == team_name]
                xg_col = 'home_xg'
            else:
                team_data = df[df['away_team'] == team_name]
                xg_col = 'away_xg'
            
            if len(team_data) == 0:
                return 1.5, 6.0
                
            # Moyenne xG
            avg_xg = team_data[xg_col].mean() if xg_col in team_data.columns else 1.5
            
            # Forme (points derniers matchs)
            recent_matches = team_data.tail(5)
            if 'result' in recent_matches.columns:
                form_points = 0
                for _, match in recent_matches.iterrows():
                    if is_home:
                        if match['result'] == 'H':
                            form_points += 3
                        elif match['result'] == 'D':
                            form_points += 1
                    else:
                        if match['result'] == 'A':
                            form_points += 3
                        elif match['result'] == 'D':
                            form_points += 1
            else:
                form_points = 6.0
                
            return float(avg_xg), float(form_points)
            
        except:
            return 1.5, 6.0