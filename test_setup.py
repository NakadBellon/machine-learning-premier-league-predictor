"""Test que l'environnement est correctement configuré."""

import sys

def test_imports():
    """Test que les imports principaux fonctionnent."""
    try:
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        import plotly.express as px
        import streamlit as st
        import mlflow
        
        print("✅ Tous les imports fonctionnent!")
        return True
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return False

def test_structure():
    """Test que la structure de dossiers est correcte."""
    import os
    required_folders = [
        'data/raw',
        'src/data_scraping', 
        'models/training',
        'notebooks'
    ]
    
    for folder in required_folders:
        if not os.path.exists(folder):
            print(f"❌ Dossier manquant: {folder}")
            return False
    
    print("✅ Structure de dossiers correcte!")
    return True

if __name__ == "__main__":
    print("🧪 Test de l'environnement...")
    
    success = True
    success &= test_imports()
    success &= test_structure()
    
    if success:
        print("\n🎉 Environnement configuré avec succès!")
        print("Tu peux commencer à coder! 🚀")
    else:
        print("\n❌ Il y a des problèmes avec la configuration.")
        sys.exit(1)