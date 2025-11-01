"""Test complet du setup MLOps."""

import sys
import os
from pathlib import Path

def test_mlflow():
    """Test que MLflow fonctionne."""
    try:
        import mlflow
        mlflow.set_tracking_uri("mlruns/")
        experiments = mlflow.search_experiments()
        print(f" MLflow: {len(experiments)} expérience(s) trouvée(s)")
        return True
    except Exception as e:
        print(f" MLflow error: {e}")
        return False

def test_dvc():
    """Test que DVC est initialisé."""
    try:
        dvc_dir = Path(".dvc")
        if dvc_dir.exists():
            print(" DVC: Initialisé")
            return True
        else:
            print(" DVC: Non initialisé")
            return False
    except Exception as e:
        print(f" DVC error: {e}")
        return False

def test_github_actions():
    """Test que les workflows GitHub existent."""
    workflows_dir = Path(".github/workflows")
    if workflows_dir.exists():
        yaml_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
        print(f"GitHub Actions: {len(yaml_files)} workflow(s) trouvé(s)")
        return True
    else:
        print("GitHub Actions: Dossier workflows manquant")
        return False

def main():
    print("Test complet du setup MLOps...\n")
    
    tests_passed = 0
    total_tests = 3
    
    tests_passed += test_mlflow()
    tests_passed += test_dvc() 
    tests_passed += test_github_actions()
    
    print(f"\n Résultat: {tests_passed}/{total_tests} tests passés")
    
    if tests_passed == total_tests:
        print(" Phase 4 MLOps terminée avec succès!")
        print("  Prêt pour la Phase 5: Collecte des données!")
    else:
        print(" Certains tests ont échoué. Vérifie la configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main()