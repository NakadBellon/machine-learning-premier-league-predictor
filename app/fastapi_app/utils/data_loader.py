# 📁 app/fastapi_app/utils/data_loader.py
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.data_paths = [
            "deployment_data/premier_league_data.csv",
            "data/processed/premier_league_with_features_20251111_123454.csv",
            "../deployment_data/premier_league_data.csv"
        ]
    
    def load_data(self):
        """Charge les données de matchs"""
        for path in self.data_paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    logger.info(f"Données chargées depuis: {path}")
                    return df
                except Exception as e:
                    logger.warning(f"Erreur chargement {path}: {e}")
        
        logger.error("Aucune donnée trouvée")
        return None
    
    def get_team_stats(self, team_name: str):
        """Récupère les statistiques d'une équipe"""
        df = self.load_data()
        if df is None:
            return None
        
        try:
            home_matches = df[df['home_team'] == team_name]
            away_matches = df[df['away_team'] == team_name]
            
            stats = {
                'home_xg_avg': home_matches['home_xg'].mean() if not home_matches.empty else 1.5,
                'away_xg_avg': away_matches['away_xg'].mean() if not away_matches.empty else 1.2,
                'total_matches': len(home_matches) + len(away_matches)
            }
            return stats
        except Exception as e:
            logger.error(f"Erreur stats équipe {team_name}: {e}")
            return None