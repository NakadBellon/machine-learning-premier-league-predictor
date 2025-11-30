# 📁 app/fastapi_app/endpoints/analytics.py
from fastapi import APIRouter, HTTPException
import logging
from datetime import datetime
import pandas as pd
import numpy as np

from app.models.predictor import TeamAnalysisRequest, TeamAnalysisResponse

router = APIRouter()
logger = logging.getLogger(__name__)

class AnalyticsEngine:
    def __init__(self):
        self.teams = [
            'Manchester City', 'Liverpool', 'Arsenal', 'Chelsea', 'Manchester Utd',
            'Tottenham', 'Newcastle', 'Brighton', 'West Ham', 'Crystal Palace',
            'Wolves', 'Aston Villa', 'Brentford', 'Fulham', 'Nottingham Forest',
            'Luton Town', 'Ipswich Town', 'West Brom', 'Sheffield Utd'
        ]
    
    def analyze_team(self, team_name: str):
        """Analyse une équipe spécifique"""
        
        # Données simulées (à remplacer par vos vraies données)
        team_data = {
            'Manchester City': {'home_xg': 2.1, 'away_xg': 1.9, 'form': 0.9},
            'Liverpool': {'home_xg': 2.0, 'away_xg': 1.8, 'form': 0.85},
            'Arsenal': {'home_xg': 1.8, 'away_xg': 1.6, 'form': 0.8},
            'Chelsea': {'home_xg': 1.7, 'away_xg': 1.5, 'form': 0.75},
            'Manchester Utd': {'home_xg': 1.6, 'away_xg': 1.4, 'form': 0.7},
        }
        
        # Données par défaut si équipe non trouvée
        default_data = {'home_xg': 1.5, 'away_xg': 1.2, 'form': 0.5}
        data = team_data.get(team_name, default_data)
        
        # Probabilités simulées
        monte_carlo_probs = {
            'Manchester City': {'champ': 0.764, 'top4': 0.999, 'releg': 0.0},
            'Liverpool': {'champ': 0.194, 'top4': 0.976, 'releg': 0.0},
            'Arsenal': {'champ': 0.025, 'top4': 0.772, 'releg': 0.0},
            'Chelsea': {'champ': 0.013, 'top4': 0.750, 'releg': 0.0},
            'Manchester Utd': {'champ': 0.002, 'top4': 0.186, 'releg': 0.0},
        }
        
        probs = monte_carlo_probs.get(team_name, {'champ': 0.0, 'top4': 0.1, 'releg': 0.1})
        
        # Niveau de risque
        if probs['releg'] > 0.5:
            risk_level = "Élevé"
        elif probs['releg'] > 0.2:
            risk_level = "Modéré"
        else:
            risk_level = "Faible"
        
        return {
            'avg_home_xg': data['home_xg'],
            'avg_away_xg': data['away_xg'],
            'form_rating': data['form'],
            'championship_prob': probs['champ'],
            'top4_prob': probs['top4'],
            'relegation_prob': probs['releg'],
            'risk_level': risk_level
        }

analytics_engine = AnalyticsEngine()

@router.post("/team", response_model=TeamAnalysisResponse)
async def analyze_team(request: TeamAnalysisRequest):
    """
    Analyse complète d'une équipe
    
    - **team_name**: Nom de l'équipe à analyser
    """
    try:
        logger.info(f"Analyse demandée pour: {request.team_name}")
        
        if request.team_name not in analytics_engine.teams:
            raise HTTPException(status_code=404, detail="Équipe non trouvée")
        
        analysis = analytics_engine.analyze_team(request.team_name)
        
        response = TeamAnalysisResponse(
            team_name=request.team_name,
            championship_prob=analysis['championship_prob'],
            top4_prob=analysis['top4_prob'],
            relegation_prob=analysis['relegation_prob'],
            avg_home_xg=analysis['avg_home_xg'],
            avg_away_xg=analysis['avg_away_xg'],
            form_rating=analysis['form_rating'],
            risk_level=analysis['risk_level']
        )
        
        logger.info(f"Analyse terminée pour: {request.team_name}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur analyse équipe: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur analyse: {str(e)}")

@router.get("/stats/overview")
async def get_league_overview():
    """
    Retourne une vue d'ensemble de la ligue
    """
    try:
        overview = {
            "total_teams": 20,
            "model_accuracy": "60.58%",
            "simulations_count": 1000,
            "top_contenders": ["Manchester City", "Liverpool", "Arsenal"],
            "relegation_candidates": ["Luton Town", "Ipswich Town", "West Brom"],
            "most_unpredictable_matches": ["Liverpool vs Manchester City", "Arsenal vs Chelsea"],
            "timestamp": datetime.now()
        }
        return overview
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur overview: {str(e)}")