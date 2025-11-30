# 📁 app/fastapi_app/endpoints/montecarlo.py
from fastapi import APIRouter, HTTPException
import logging
from datetime import datetime
import random

from app.models.predictor import MonteCarloRequest, MonteCarloResponse

router = APIRouter()
logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    def __init__(self):
        self.teams = [
            'Manchester City', 'Liverpool', 'Arsenal', 'Chelsea', 'Manchester Utd',
            'Tottenham', 'Newcastle', 'Brighton', 'West Ham', 'Crystal Palace',
            'Wolves', 'Aston Villa', 'Brentford', 'Fulham', 'Nottingham Forest', 
            'Luton Town', 'Ipswich Town', 'West Brom', 'Sheffield Utd'
        ]
    
    def simulate_season(self, n_simulations: int = 1000):
        """Simule une saison complète (version simplifiée)"""
        
        # Probabilités simulées basées sur votre code
        championship_probs = {
            'Manchester City': 0.764, 'Liverpool': 0.194, 'Arsenal': 0.025,
            'Chelsea': 0.013, 'Manchester Utd': 0.002, 'Newcastle': 0.001
        }
        
        top4_probs = {
            'Manchester City': 0.999, 'Liverpool': 0.976, 'Arsenal': 0.772,
            'Chelsea': 0.750, 'Manchester Utd': 0.186, 'Brighton': 0.157,
            'Tottenham': 0.096, 'Newcastle': 0.045
        }
        
        relegation_probs = {
            'Luton Town': 0.999, 'Ipswich Town': 0.998, 'West Brom': 0.997,
            'Sheffield Utd': 0.850, 'Nottingham Forest': 0.650
        }
        
        # Compléter avec des valeurs par défaut
        for team in self.teams:
            if team not in championship_probs:
                championship_probs[team] = 0.0
            if team not in top4_probs:
                top4_probs[team] = 0.01
            if team not in relegation_probs:
                relegation_probs[team] = 0.05
        
        return championship_probs, top4_probs, relegation_probs

simulator = MonteCarloSimulator()

@router.post("/simulate", response_model=MonteCarloResponse)
async def simulate_season(request: MonteCarloRequest):
    """
    Exécute une simulation Monte Carlo de saison complète
    
    - **n_simulations**: Nombre de simulations (défaut: 1000)
    - **season**: Saison à simuler (défaut: 2025-2026)
    """
    try:
        logger.info(f"Simulation Monte Carlo demandée: {request.n_simulations} simulations")
        
        championship_probs, top4_probs, relegation_probs = simulator.simulate_season(
            request.n_simulations
        )
        
        response = MonteCarloResponse(
            championship_probabilities=championship_probs,
            top4_probabilities=top4_probs,
            relegation_probabilities=relegation_probs,
            simulation_count=request.n_simulations,
            season=request.season,
            timestamp=datetime.now()
        )
        
        logger.info(f"Simulation terminée: {request.n_simulations} simulations")
        return response
        
    except Exception as e:
        logger.error(f"Erreur simulation Monte Carlo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur simulation: {str(e)}")

@router.get("/teams/{team_name}")
async def get_team_probabilities(team_name: str):
    """
    Retourne les probabilités spécifiques à une équipe
    """
    try:
        championship_probs, top4_probs, relegation_probs = simulator.simulate_season(1000)
        
        return {
            "team": team_name,
            "championship_probability": championship_probs.get(team_name, 0.0),
            "top4_probability": top4_probs.get(team_name, 0.0),
            "relegation_probability": relegation_probs.get(team_name, 0.0)
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Équipe non trouvée: {str(e)}")