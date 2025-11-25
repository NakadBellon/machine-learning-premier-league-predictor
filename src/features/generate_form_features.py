"""
Génération des features de forme pour les équipes
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def calculate_team_form(df, team, match_date, n_matches=5):
    """Calcule la forme d'une équipe sur les n derniers matchs"""
    
    # Matchs précédents de l'équipe
    team_matches = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['date'] < match_date)
    ].sort_values('date').tail(n_matches)
    
    if len(team_matches) == 0:
        return {
            'points': 0,
            'goals_scored': 0,
            'goals_conceded': 0,
            'xg_for': 0,
            'xg_against': 0
        }
    
    points = 0
    goals_scored = 0
    goals_conceded = 0
    xg_for = 0
    xg_against = 0
    
    for _, match in team_matches.iterrows():
        if match['home_team'] == team:
            # Équipe à domicile
            goals_scored += match['home_score']
            goals_conceded += match['away_score']
            xg_for += match.get('home_xg', 0)
            xg_against += match.get('away_xg', 0)
            
            if match['result'] == 'H':
                points += 3
            elif match['result'] == 'D':
                points += 1
        else:
            # Équipe à l'extérieur
            goals_scored += match['away_score']
            goals_conceded += match['home_score']
            xg_for += match.get('away_xg', 0)
            xg_against += match.get('home_xg', 0)
            
            if match['result'] == 'A':
                points += 3
            elif match['result'] == 'D':
                points += 1
    
    return {
        'points': points,
        'goals_scored': goals_scored,
        'goals_conceded': goals_conceded,
        'xg_for': xg_for,
        'xg_against': xg_against
    }

def add_form_features(df, n_matches=5):
    """Ajoute les features de forme au DataFrame"""
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"Ajout des features de forme (derniers {n_matches} matchs)...")
    
    # Initialiser les colonnes
    for prefix in ['home', 'away']:
        df[f'{prefix}_last_{n_matches}_points'] = 0
        df[f'{prefix}_last_{n_matches}_goals_scored'] = 0
        df[f'{prefix}_last_{n_matches}_goals_conceded'] = 0
        df[f'{prefix}_last_{n_matches}_xg_for'] = 0
        df[f'{prefix}_last_{n_matches}_xg_against'] = 0
    
    # Calculer pour chaque match
    for idx, match in tqdm(df.iterrows(), total=len(df)):
        match_date = match['date']
        home_team = match['home_team']
        away_team = match['away_team']
        
        # Forme domicile
        home_form = calculate_team_form(df.iloc[:idx], home_team, match_date, n_matches)
        df.at[idx, 'home_last_5_points'] = int(home_form['points'])
        df.at[idx, 'home_last_5_goals_scored'] = int(home_form['goals_scored'])
        df.at[idx, 'home_last_5_goals_conceded'] = int(home_form['goals_conceded'])
        df.at[idx, 'home_last_5_xg_for'] = float(home_form['xg_for'])
        df.at[idx, 'home_last_5_xg_against'] = float(home_form['xg_against'])
        
        # Forme extérieur
        away_form = calculate_team_form(df.iloc[:idx], away_team, match_date, n_matches)
        df.at[idx, 'away_last_5_points'] = int(away_form['points'])
        df.at[idx, 'away_last_5_goals_scored'] = int(away_form['goals_scored'])
        df.at[idx, 'away_last_5_goals_conceded'] = int(away_form['goals_conceded'])
        df.at[idx, 'away_last_5_xg_for'] = float(away_form['xg_for'])
        df.at[idx, 'away_last_5_xg_against'] = float(away_form['xg_against'])
    
    return df

def main():
    """Fonction principale"""
    # Charger les données
    input_file = 'data/processed/premier_league_processed_20251111_123454.csv'
    output_file = 'data/processed/premier_league_with_features_20251111_123454.csv'
    
    print("Chargement des données...")
    df = pd.read_csv(input_file)
    
    # Ajouter les features
    df_with_features = add_form_features(df)
    
    # Sauvegarder
    df_with_features.to_csv(output_file, index=False)
    print(f"✅ Données sauvegardées avec features: {output_file}")
    print(f"📊 Shape: {df_with_features.shape}")
    print(f"🎯 Colonnes ajoutées: {[col for col in df_with_features.columns if 'last' in col]}")

if __name__ == "__main__":
    main()