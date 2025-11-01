"""Configuration de DVC avec Google Drive pour le versioning des données."""

import subprocess
import sys
from pathlib import Path

def run_dvc_command(command):
    """Exécute une commande DVC et gère les erreurs."""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f" {command}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f" Erreur avec: {command}")
        print(f"Message: {e.stderr}")
        return None

def setup_dvc_remote():
    """Configure un remote DVC avec Google Drive."""
    
    print(" Configuration du remote DVC...")
    
    # Créer un dossier de test pour les données
    test_data_dir = Path("data/test")
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Créer un fichier de test
    test_file = test_data_dir / "test_data.txt"
    test_file.write_text("Données de test pour DVC")
    
    # Ajouter le fichier à DVC
    run_dvc_command("dvc add data/test/test_data.txt")
    
    print("\n Instructions pour Google Drive:")
    print("1. Va sur https://drive.google.com")
    print("2. Crée un dossier 'premier_league_data'")
    print("3. Partage le dossier en mode 'Editeur'")
    print("4. Copie l'ID du dossier depuis l'URL")
    print("5. Exécute la commande suivante:")
    print("   dvc remote add -d myremote gdrive://[TON_ID_DOSSIER]")
    
    return test_file

def basic_dvc_workflow():
    """Montre le workflow basique DVC."""
    
    print("\n Workflow DVC de base:")
    print("1. dvc add data/raw/mon_fichier.csv")
    print("2. git add data/raw/mon_fichier.csv.dvc .gitignore")
    print("3. git commit -m 'Add data with DVC'")
    print("4. dvc push (pour envoyer sur le remote)")
    print("5. dvc pull (pour récupérer les données)")

if __name__ == "__main__":
    setup_dvc_remote()
    basic_dvc_workflow()