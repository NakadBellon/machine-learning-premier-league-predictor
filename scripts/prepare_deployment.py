# scripts/prepare_deployment.py
import pandas as pd
import os
from datetime import datetime

def prepare_data_for_deployment():
    """Prépare les données pour le déploiement"""
    
    # Charger vos données
    input_path = "data/processed/premier_league_with_features_20251111_123454.csv"
    output_dir = "deployment_data"
    
    # Créer le dossier
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Copier les données essentielles
        df = pd.read_csv(input_path)
        
        # Garder seulement les colonnes nécessaires pour réduire la taille
        essential_columns = [
            'home_team', 'away_team', 'home_xg', 'away_xg', 'result',
            'home_last_5_points', 'away_last_5_points'
        ]
        
        # Vérifier que les colonnes existent
        available_columns = [col for col in essential_columns if col in df.columns]
        df_slim = df[available_columns].copy()
        df_slim.to_csv(f"{output_dir}/premier_league_data.csv", index=False)
        
        print(f"✅ Données préparées pour déploiement: {len(df_slim)} lignes")
        print(f"📊 Colonnes conservées: {available_columns}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        # Créer un fichier sample si les données ne sont pas disponibles
        sample_data = pd.DataFrame({
            'home_team': ['Liverpool', 'Man City'],
            'away_team': ['Man City', 'Liverpool'],
            'home_xg': [1.8, 2.1],
            'away_xg': [1.5, 1.7],
            'result': ['H', 'A']
        })
        sample_data.to_csv(f"{output_dir}/premier_league_data.csv", index=False)
        print("✅ Fichier sample créé")
    
    # Copier aussi à la racine pour HF Spaces
    df_slim.to_csv("deployment_data/premier_league_data.csv", index=False)
    
    print("✅ Données prêtes pour Docker et HF Spaces")

if __name__ == "__main__":
    prepare_data_for_deployment()