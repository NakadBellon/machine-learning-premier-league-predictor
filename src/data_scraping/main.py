"""Pipeline principal de scraping et traitement des données."""

import logging
import pandas as pd
from datetime import datetime
import os

import mlflow

from .fbref_scraper import PremierLeagueScraper
from .data_cleaner import DataCleaner
from .feature_engineer import FeatureEngineer
from .config import SCRAPING_CONFIG


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


def main():
    """Pipeline complet de données avec tracking MLOps."""
    setup_logging()
    logger = logging.getLogger('data_pipeline')
    
    mlflow.set_experiment("Premier_League_Data_Pipeline")
    
    with mlflow.start_run(run_name=f"data_scraping_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        try:
            logger.info("Démarrage du pipeline de données Premier League...")
            
            mlflow.log_params({
                'seasons_count': len(SCRAPING_CONFIG['seasons']),
                'form_matches_window': SCRAPING_CONFIG['features']['form_matches_window']
            })
            
            # 1. Scraping
            logger.info("Phase 1: Scraping des données...")
            scraper = PremierLeagueScraper()
            raw_data = scraper.get_all_matches()
            
            if raw_data is None:
                raise Exception("Échec du scraping")
            
            mlflow.log_metric('raw_matches_count', len(raw_data))
            
            # 2. Nettoyage
            logger.info("Phase 2: Nettoyage des données...")
            cleaner = DataCleaner()
            cleaned_data = cleaner.clean_matches_data(raw_data)
            
            mlflow.log_metric('cleaned_matches_count', len(cleaned_data))
            
            # 3. Feature Engineering
            logger.info("Phase 3: Feature Engineering...")
            feature_engineer = FeatureEngineer()
            final_data = feature_engineer.create_team_form_features(cleaned_data)
            
            mlflow.log_metric('final_features_count', len(final_data.columns))
            
            # 4. Sauvegarde
            logger.info("Phase 4: Sauvegarde des données...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            os.makedirs('data/raw', exist_ok=True)
            os.makedirs('data/processed', exist_ok=True)
            
            raw_path = f'data/raw/premier_league_raw_{timestamp}.csv'
            processed_path = f'data/processed/premier_league_processed_{timestamp}.csv'
            
            raw_data.to_csv(raw_path, index=False)
            final_data.to_csv(processed_path, index=False)
            
            logger.info(f"Données sauvegardées: {raw_path}, {processed_path}")
            
            mlflow.log_artifact(raw_path)
            mlflow.log_artifact(processed_path)
            
            logger.info("Pipeline de données terminé avec succès!")
            
            return final_data
            
        except Exception as e:
            logger.error(f"Erreur dans le pipeline: {e}")
            raise


if __name__ == "__main__":
    main()