"""Scraper FBref pour Premier League."""

import soccerdata as sd
import pandas as pd
import logging
from config import config  # Importer l'instance config existante


class PremierLeagueScraper:
    """Scraper pour récupérer les données Premier League depuis FBref."""
    
    def __init__(self):
        self.fbref = sd.FBref(leagues=config.scraper.league)  # Utiliser config.scraper.league
        self.seasons = config.scraper.seasons  # Utiliser config.scraper.seasons
        self.logger = logging.getLogger(__name__)
    
    def get_season_matches(self, season):
        """Récupère tous les matchs d'une saison."""
        self.logger.info(f"📅 Scraping saison {season}...")
        try:
            matches = self.fbref.read_schedule(season)
            matches['season'] = season
            self.logger.info(f"✅ {len(matches)} matchs trouvés")
            return matches
        except Exception as e:
            self.logger.error(f"❌ Erreur {season}: {e}")
            return None
    
    def get_all_matches(self):
        """Récupère tous les matchs 2019-2026."""
        self.logger.info("🚀 Début du scraping Premier League 2019-2026...")
        all_matches = []
        
        for season_name, season_code in self.seasons.items():
            season_matches = self.get_season_matches(season_code)
            if season_matches is not None:
                all_matches.append(season_matches)
        
        if all_matches:
            df = pd.concat(all_matches, ignore_index=True)
            self.logger.info(f"🎉 TOTAL: {len(df)} matchs sur {len(self.seasons)} saisons")
            self.logger.info(f"📊 Shape: {df.shape}")
            self.logger.info(f"📋 Colonnes: {list(df.columns)}")
            return df
        
        self.logger.error("❌ Aucune donnée récupérée")
        return None