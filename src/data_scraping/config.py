"""
Configuration centralisée pour le projet Premier League Prediction
"""
from dataclasses import dataclass
from typing import List, Dict
import os
from pathlib import Path

@dataclass
class ScraperConfig:
    """Configuration du scraper FBref"""
    league: str = 'ENG-Premier League'
    seasons: Dict[str, str] = None
    
    def __post_init__(self):
        if self.seasons is None:
            self.seasons = {
                '2019-2020': '2019-2020',
                '2020-2021': '2020-2021', 
                '2021-2022': '2021-2022',
                '2022-2023': '2022-2023',
                '2023-2024': '2023-2024',
                '2024-2025': '2024-2025',
                '2025-2026': '2025-2026'
            }

@dataclass
class FeatureConfig:
    """Configuration du feature engineering"""
    n_last_matches: int = 5
    include_xg: bool = True
    include_form: bool = True

@dataclass
class PathConfig:
    """Configuration des chemins de fichiers"""
    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Path = None
    raw_data_dir: Path = None
    processed_data_dir: Path = None
    models_dir: Path = None
    mlruns_dir: Path = None
    
    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.project_root / 'data'
        if self.raw_data_dir is None:
            self.raw_data_dir = self.data_dir / 'raw'
        if self.processed_data_dir is None:
            self.processed_data_dir = self.data_dir / 'processed'
        if self.models_dir is None:
            self.models_dir = self.project_root / 'models'
        if self.mlruns_dir is None:
            self.mlruns_dir = self.project_root / 'mlruns'
        
        # Créer les dossiers s'ils n'existent pas
        for path in [self.data_dir, self.raw_data_dir, 
                     self.processed_data_dir, self.models_dir, self.mlruns_dir]:
            path.mkdir(parents=True, exist_ok=True)

@dataclass
class MLflowConfig:
    """Configuration MLflow"""
    tracking_uri: str = None
    experiment_name: str = "premier_league_prediction"
    
    def __post_init__(self):
        if self.tracking_uri is None:
            self.tracking_uri = f"file://{PathConfig().mlruns_dir}"

class Config:
    """Configuration globale du projet"""
    def __init__(self):
        self.scraper = ScraperConfig()
        self.features = FeatureConfig()
        self.paths = PathConfig()
        self.mlflow = MLflowConfig()

# Instance globale
config = Config()