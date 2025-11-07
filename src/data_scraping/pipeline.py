"""Pipeline principal de scraping et traitement des données."""

import logging
import pandas as pd
from datetime import datetime
import os

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow non disponible - le tracking sera désactivé")

from scraper import PremierLeagueScraper
from config import config

# Imports conditionnels pour éviter les erreurs
try:
    from cleaner import DataCleaner
    CLEANER_AVAILABLE = True
except ImportError:
    CLEANER_AVAILABLE = False
    print("⚠️ Cleaner non disponible - phase de nettoyage ignorée")

try:
    from features import FeatureEngineer
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    print("⚠️ FeatureEngineer non disponible - phase features ignorée")


def setup_logging():
    """Configure le logging."""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/data_pipeline.log')
        ]
    )


def run_pipeline():
    """Pipeline complet de données avec tracking MLOps."""
    setup_logging()
    logger = logging.getLogger('data_pipeline')
    
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("Premier_League_Data_Pipeline")
        run = mlflow.start_run(run_name=f"data_scraping_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    try:
        logger.info("=" * 80)
        logger.info("🚀 DÉMARRAGE DU PIPELINE PREMIER LEAGUE")
        logger.info("=" * 80)
        
        # Loguer la configuration
        if MLFLOW_AVAILABLE:
            mlflow.log_params({
                'seasons_count': len(config.scraper.seasons),
                'league': config.scraper.league
            })
        
        # 1. SCRAPING
        logger.info("\n" + "=" * 80)
        logger.info("📡 PHASE 1: SCRAPING DES DONNÉES")
        logger.info("=" * 80)
        
        scraper = PremierLeagueScraper()
        raw_data = scraper.get_all_matches()
        
        if raw_data is None:
            raise Exception("Échec du scraping - aucune donnée récupérée")
        
        if MLFLOW_AVAILABLE:
            mlflow.log_metric('raw_matches_count', len(raw_data))
        
        current_data = raw_data
        
        # 2. NETTOYAGE (si disponible)
        if CLEANER_AVAILABLE:
            logger.info("\n" + "=" * 80)
            logger.info("🧹 PHASE 2: NETTOYAGE DES DONNÉES")
            logger.info("=" * 80)
            
            cleaner = DataCleaner()
            current_data = cleaner.clean_matches_data(raw_data)
            
            if MLFLOW_AVAILABLE:
                mlflow.log_metric('cleaned_matches_count', len(current_data))
        else:
            logger.info("⏭️ Phase nettoyage ignorée (cleaner non disponible)")
        
        # 3. FEATURE ENGINEERING (si disponible)
        if FEATURES_AVAILABLE and CLEANER_AVAILABLE:
            logger.info("\n" + "=" * 80)
            logger.info("⚙️ PHASE 3: FEATURE ENGINEERING")
            logger.info("=" * 80)
            
            feature_engineer = FeatureEngineer()
            final_data = feature_engineer.create_team_form_features(current_data)
            
            if MLFLOW_AVAILABLE:
                mlflow.log_metric('final_features_count', len(final_data.columns))
        else:
            final_data = current_data
            logger.info("⏭️ Phase feature engineering ignorée")
        
        # 4. SAUVEGARDE
        logger.info("\n" + "=" * 80)
        logger.info("💾 PHASE 4: SAUVEGARDE DES DONNÉES")
        logger.info("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        raw_path = f'data/raw/premier_league_raw_{timestamp}.csv'
        processed_path = f'data/processed/premier_league_processed_{timestamp}.csv'
        
        raw_data.to_csv(raw_path, index=False)
        final_data.to_csv(processed_path, index=False)
        
        logger.info(f"✅ Données brutes sauvegardées: {raw_path}")
        logger.info(f"✅ Données traitées sauvegardées: {processed_path}")
        
        if MLFLOW_AVAILABLE:
            mlflow.log_artifact(raw_path)
            mlflow.log_artifact(processed_path)
        
        # 5. STATISTIQUES FINALES
        logger.info("\n" + "=" * 80)
        logger.info("📊 PHASE 5: STATISTIQUES FINALES")
        logger.info("=" * 80)
        
        stats = {
            'total_matches': len(final_data),
            'seasons': final_data['season'].nunique() if 'season' in final_data.columns else 'N/A',
            'teams': pd.concat([final_data['home_team'], final_data['away_team']]).nunique() if 'home_team' in final_data.columns else 'N/A',
            'features': len(final_data.columns),
        }
        
        if 'date' in final_data.columns:
            stats['date_range'] = f"{final_data['date'].min()} → {final_data['date'].max()}"
        
        for key, value in stats.items():
            logger.info(f"   • {key}: {value}")
            if MLFLOW_AVAILABLE and key != 'date_range' and value != 'N/A':
                mlflow.log_metric(key, value)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ PIPELINE TERMINÉ AVEC SUCCÈS!")
        logger.info("=" * 80)
        
        return final_data, stats
        
    except Exception as e:
        logger.error(f"\n❌ ERREUR DANS LE PIPELINE: {e}")
        if MLFLOW_AVAILABLE:
            mlflow.log_param('error', str(e))
        raise
    
    finally:
        if MLFLOW_AVAILABLE:
            mlflow.end_run()


if __name__ == "__main__":
    run_pipeline()