# 📁 scripts/prepare_deployment.py
import pandas as pd
import numpy as np
import os
from datetime import datetime
import shutil

def prepare_deployment_data():
    """Prépare toutes les données nécessaires pour le déploiement Docker"""
    
    print("🚀 Préparation des données pour le déploiement...")
    
    # Créer le dossier de déploiement
    deployment_dir = "deployment_data"
    os.makedirs(deployment_dir, exist_ok=True)
    
    # 1. Données principales
    try:
        # Chercher le fichier de données le plus récent
        processed_dir = "data/processed"
        if os.path.exists(processed_dir):
            files = [f for f in os.listdir(processed_dir) 
                    if f.startswith('premier_league') and f.endswith('.csv')]
            
            if files:
                latest_file = sorted(files)[-1]
                source_path = os.path.join(processed_dir, latest_file)
                df = pd.read_csv(source_path)
                
                # Garder seulement les colonnes essentielles pour réduire la taille
                essential_columns = [
                    'home_team', 'away_team', 'home_xg', 'away_xg', 'result',
                    'home_score', 'away_score', 'date', 'season'
                ]
                
                available_columns = [col for col in essential_columns if col in df.columns]
                df_deploy = df[available_columns].copy()
                
                # Sauvegarder
                deploy_path = os.path.join(deployment_dir, "premier_league_data.csv")
                df_deploy.to_csv(deploy_path, index=False)
                print(f"✅ Données principales sauvegardées: {len(df_deploy)} lignes, {len(available_columns)} colonnes")
            else:
                create_sample_data(deployment_dir)
        else:
            create_sample_data(deployment_dir)
            
    except Exception as e:
        print(f"❌ Erreur préparation données: {e}")
        create_sample_data(deployment_dir)
    
    # 2. Modèles (si disponibles)
    models_src = "models"
    models_dest = os.path.join(deployment_dir, "models")
    
    if os.path.exists(models_src):
        os.makedirs(models_dest, exist_ok=True)
        try:
            # Copier les modèles les plus importants
            for model_file in os.listdir(models_src):
                if model_file.endswith('.pkl') or model_file.endswith('.joblib'):
                    shutil.copy2(
                        os.path.join(models_src, model_file),
                        os.path.join(models_dest, model_file)
                    )
                    print(f"✅ Modèle copié: {model_file}")
        except Exception as e:
            print(f"⚠️  Erreur copie modèles: {e}")
    
    # 3. Fichier de configuration
    create_config_file(deployment_dir)
    
    print("🎉 Préparation des données terminée!")
    print(f"📁 Dossier de déploiement: {deployment_dir}")

def create_sample_data(deployment_dir):
    """Crée des données d'exemple si les vraies données ne sont pas disponibles"""
    print("📝 Création de données d'exemple...")
    
    sample_data = pd.DataFrame({
        'home_team': ['Liverpool', 'Manchester City', 'Arsenal', 'Chelsea'],
        'away_team': ['Manchester City', 'Liverpool', 'Chelsea', 'Arsenal'],
        'home_xg': [1.8, 2.1, 1.6, 1.7],
        'away_xg': [1.5, 1.8, 1.4, 1.5],
        'home_score': [3, 2, 1, 0],
        'away_score': [1, 2, 1, 2],
        'result': ['H', 'D', 'D', 'A'],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
        'season': ['2023-2024'] * 4
    })
    
    sample_path = os.path.join(deployment_dir, "premier_league_data.csv")
    sample_data.to_csv(sample_path, index=False)
    print(f"✅ Données d'exemple créées: {len(sample_data)} matchs")

def create_config_file(deployment_dir):
    """Crée un fichier de configuration"""
    config = {
        "version": "1.0.0",
        "deployment_date": datetime.now().isoformat(),
        "model_accuracy": 0.6058,
        "features_used": ["home_xg", "away_xg", "team_form"],
        "data_sources": ["FBref"],
        "last_training_date": "2024-01-01"
    }
    
    import json
    config_path = os.path.join(deployment_dir, "deployment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Fichier de configuration créé")

if __name__ == "__main__":
    prepare_deployment_data()