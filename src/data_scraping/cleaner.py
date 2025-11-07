"""Nettoyage des données de matchs."""

import pandas as pd
import logging


class DataCleaner:
    """Nettoie et prépare les données de matchs."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def clean_matches_data(self, df):
        """Nettoie et enrichit les données."""
        self.logger.info("🧹 Début du nettoyage des données...")
        
        df_clean = df.copy()
        initial_count = len(df_clean)
        
        try:
            # Parser le score
            df_clean[['home_score', 'away_score']] = df_clean['score'].str.split('–', expand=True)
            df_clean['home_score'] = pd.to_numeric(df_clean['home_score'], errors='coerce')
            df_clean['away_score'] = pd.to_numeric(df_clean['away_score'], errors='coerce')
            
            # Créer la variable cible (résultat)
            df_clean['result'] = df_clean.apply(self._get_result, axis=1)
            
            # Convertir la date
            df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
            
            # Supprimer les matchs non joués
            df_clean = df_clean.dropna(subset=['home_score', 'away_score'])
            
            # Trier par date
            df_clean = df_clean.sort_values('date').reset_index(drop=True)
            
            final_count = len(df_clean)
            result_dist = df_clean['result'].value_counts().to_dict()
            
            self.logger.info(f"✅ Données nettoyées: {final_count}/{initial_count} matchs conservés")
            self.logger.info(f"📊 Distribution résultats: {result_dist}")
            
            return df_clean
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du nettoyage: {e}")
            raise
    
    @staticmethod
    def _get_result(row):
        """Détermine le résultat du match."""
        if pd.isna(row['home_score']) or pd.isna(row['away_score']):
            return None
        if row['home_score'] > row['away_score']:
            return 'H'  # Home win
        elif row['home_score'] < row['away_score']:
            return 'A'  # Away win
        else:
            return 'D'  # Draw