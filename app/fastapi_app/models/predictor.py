# 📁 app/fastapi_app/models/predictor.py
from pydantic import BaseModel, ConfigDict
from typing import Dict, List, Optional, Any
from datetime import datetime

# Configuration pour éviter les warnings
class BaseModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

# Requêtes
class MatchPredictionRequest(BaseModelConfig):
    home_team: str
    away_team: str
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None

class MonteCarloRequest(BaseModelConfig):
    n_simulations: Optional[int] = 1000
    season: Optional[str] = "2025-2026"

class TeamAnalysisRequest(BaseModelConfig):
    team_name: str

# Réponses
class MatchPredictionResponse(BaseModelConfig):
    prediction: str
    probabilities: Dict[str, float]
    confidence: float
    home_team: str
    away_team: str
    model_used: str = "logistic_regression"
    timestamp: datetime

class MonteCarloResponse(BaseModelConfig):
    championship_probabilities: Dict[str, float]
    top4_probabilities: Dict[str, float] 
    relegation_probabilities: Dict[str, float]
    simulation_count: int
    season: str
    timestamp: datetime

class TeamAnalysisResponse(BaseModelConfig):
    team_name: str
    championship_prob: float
    top4_prob: float
    relegation_prob: float
    avg_home_xg: float
    avg_away_xg: float
    form_rating: float
    risk_level: str

class APIStatusResponse(BaseModelConfig):
    status: str
    version: str
    endpoints: Dict[str, str]
    model_accuracy: str
    uptime: Optional[float] = None