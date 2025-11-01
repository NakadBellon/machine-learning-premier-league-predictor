"""Setup script for Premier League Predictor project."""

import subprocess
import sys
from pathlib import Path

def run_command(command):
    """Execute une commande shell."""
    try:
        subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur avec la commande: {command}")
        print(f"Message d'erreur: {e}")
        return False

def main():
    """Setup principal du projet."""
    print("🚀 Configuration du projet Premier League Predictor...")
    
    # Vérifie que Python est installé
    if not run_command("python --version"):
        sys.exit(1)
    
    # Crée la structure de dossiers
    folders = [
        'data/raw', 'data/processed', 'data/external',
        'models/training', 'models/serialized', 'models/evaluation',
        'src/data_scraping', 'src/feature_engineering', 
        'src/modeling', 'src/monitoring',
        'notebooks', 'tests',
        'mlops/workflow', 'mlops/monitoring', 'mlops/deployment',
        'app', '.github/workflows'
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"✅ Dossier créé: {folder}")
    
    print("\n🎉 Setup terminé!")
    print("Prochaine étape:")
    print("1. Active ton environnement virtuel")
    print("2. Lance: pip install -r requirements.txt")
    print("3. Ouvre le projet dans VSCode")

if __name__ == "__main__":
    main()