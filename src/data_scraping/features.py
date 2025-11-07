"""Création de features pour le machine learning."""

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from .config import SCRAPING_CONFIG


class FeatureEngineer:
    """Crée des features basées sur l'historique des équipes."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.n_matches = SCRAPING_CONFIG['features']['form_matches_window']
    
    def create_team_form_features(self, df):
        """Crée des features basées sur la forme des équipes."""
        self.logger.info(f"⚙️ Création des features de forme (derniers {self.n_matches} matchs)...")
        
        df_features = df.copy()
        
        # Initialiser les colonnes
        for prefix in ['home', 'away']:
            df_features[f'{prefix}_last_{self.n_matches}_points'] = 0.0
            df_features[f'{prefix}_last_{self.n_matches}_goals_scored'] = 0.0
            df_features[f'{prefix}_last_{self.n_matches}_goals_conceded'] = 0.0
            df_features[f'{prefix}_last_{self.n_matches}_xg'] = 0.0
        
        teams = pd.concat([df_features['home_team'], df_features['away_team']]).unique()
        
        self.logger.info(f"🔄 Traitement de {len(teams)} équipes...")
        
        for team in tqdm(teams, desc="Processing teams"):
            self._calculate_team_features(df_features, team)
        
        self.logger.info(f"✅ Features de forme créées avec succès")
        return df_features
    
    def _calculate_team_features(self, df, team):
        """Calcule les features pour une équipe."""
        # Matchs à domicile et extérieur
        home_matches = df[df['home_team'] == team].index
        away_matches = df[df['away_team'] == team].index
        
        # Tous les matchs de l'équipe (triés chronologiquement)
        all_matches = sorted(list(home_matches) + list(away_matches))
        
        for i, match_idx in enumerate(all_matches):
            if i < self.n_matches:
                continue  # Pas assez d'historique
            
            # Récupérer les n derniers matchs
            last_matches_idx = all_matches[max(0, i - self.n_matches):i]
            
            # Calculer les statistiques
            stats = self._calculate_stats(df, team, last_matches_idx)
            
            # Assigner les features selon si match à domicile ou extérieur
            prefix = 'home' if match_idx in home_matches else 'away'
            self._assign_features(df, match_idx, prefix, stats)
    
    def _calculate_stats(self, df, team, match_indices):
        """Calcule les statistiques sur une liste de matchs."""
        points = 0
        goals_scored = 0
        goals_conceded = 0
        xg = 0
        
        for prev_idx in match_indices:
            prev_match = df.loc[prev_idx]
            
            if prev_match['home_team'] == team:
                # L'équipe jouait à domicile
                goals_scored += prev_match['home_score']
                goals_conceded += prev_match['away_score']
                xg += prev_match['home_xg'] if pd.notna(prev_match['home_xg']) else 0
                
                if prev_match['result'] == 'H':
                    points += 3
                elif prev_match['result'] == 'D':
                    points += 1
            else:
                # L'équipe jouait à l'extérieur
                goals_scored += prev_match['away_score']
                goals_conceded += prev_match['home_score']
                xg += prev_match['away_xg'] if pd.notna(prev_match['away_xg']) else 0
                
                if prev_match['result'] == 'A':
                    points += 3
                elif prev_match['result'] == 'D':
                    points += 1
        
        return {
            'points': points,
            'goals_scored': goals_scored,
            'goals_conceded': goals_conceded,
            'xg': xg
        }
    
    def _assign_features(self, df, match_idx, prefix, stats):
        """Assigne les features calculées au DataFrame."""
        df.at[match_idx, f'{prefix}_last_{self.n_matches}_points'] = stats['points']
        df.at[match_idx, f'{prefix}_last_{self.n_matches}_goals_scored'] = stats['goals_scored']
        df.at[match_idx, f'{prefix}_last_{self.n_matches}_goals_conceded'] = stats['goals_conceded']
        df.at[match_idx, f'{prefix}_last_{self.n_matches}_xg'] = stats['xg']