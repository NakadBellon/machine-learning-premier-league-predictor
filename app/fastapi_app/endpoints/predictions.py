# 📁 app/fastapi_app/endpoints/predictions.py
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import logging
from datetime import datetime

from app.models.predictor import (
    MatchPredictionRequest, 
    MatchPredictionResponse
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Simulateur de prédiction (à remplacer par votre vrai modèle)
class MatchPredictor:
    def __init__(self):
        self.model_accuracy = 0.6058
        self.model_name = "optimized_logistic_regression"
    
    def predict_match(self, home_team: str, away_team: str, home_xg: float, away_xg: float):
        """Simule la prédiction d'un match (à intégrer avec votre vrai modèle)"""
        
        # Logique basée sur xG (comme dans votre code existant)
        total_xg = home_xg + away_xg + 0.1
        
        prob_home = (home_xg / total_xg) * 0.7 + 0.15
        prob_away = (away_xg / total_xg) * 0.7 + 0.15  
        prob_draw = 1 - prob_home - prob_away
        
        # Normalisation
        total = prob_home + prob_away + prob_draw
        probabilities = {
            'H': prob_home / total,
            'A': prob_away / total,
            'D': prob_draw / total
        }
        
        # Résultat le plus probable
        prediction = max(probabilities.items(), key=lambda x: x[1])[0]
        confidence = probabilities[prediction]
        
        return prediction, probabilities, confidence

predictor = MatchPredictor()

@router.post("/match", response_model=MatchPredictionResponse)
async def predict_match(request: MatchPredictionRequest):
    """
    Prédit le résultat d'un match de Premier League
    
    - **home_team**: Équipe à domicile
    - **away_team**: Équipe à l'extérieur  
    - **home_xg**: xG attendu à domicile (optionnel)
    - **away_xg**: xG attendu à l'extérieur (optionnel)
    """
    try:
        logger.info(f"Prédiction demandée: {request.home_team} vs {request.away_team}")
        
        # Valeurs par défaut si xG non fournies
        home_xg = request.home_xg or 1.5
        away_xg = request.away_xg or 1.2
        
        # Obtenir la prédiction
        prediction, probabilities, confidence = predictor.predict_match(
            request.home_team, request.away_team, home_xg, away_xg
        )
        
        response = MatchPredictionResponse(
            prediction=prediction,
            probabilities=probabilities,
            confidence=confidence,
            home_team=request.home_team,
            away_team=request.away_team,
            timestamp=datetime.now()
        )
        
        logger.info(f"Prédiction terminée: {prediction} (confiance: {confidence:.2f})")
        return response
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

@router.get("/teams")
async def get_available_teams():
    """
    Retourne la liste des équipes disponibles pour les prédictions
    """
    teams = [
        'Manchester City', 'Liverpool', 'Arsenal', 'Chelsea', 'Manchester Utd',
        'Tottenham', 'Newcastle', 'Brighton', 'West Ham', 'Crystal Palace',
        'Wolves', 'Aston Villa', 'Brentford', 'Fulham', 'Nottingham Forest',
        'Luton Town', 'Ipswich Town', 'West Brom', 'Sheffield Utd'
    ]
    return {"teams": teams}

@router.get("/batch")
async def predict_multiple_matches(
    matches: str = Query(..., description="Liste de matchs au format 'équipe1,équipe2;équipe3,équipe4'")
):
    """
    Prédit plusieurs matchs en une seule requête
    
    Format: "Liverpool,Man City; Arsenal,Chelsea; ..."
    """
    try:
        matches_list = matches.split(';')
        results = []
        
        for match in matches_list:
            if match.strip():
                home_team, away_team = match.strip().split(',')
                prediction, probabilities, confidence = predictor.predict_match(
                    home_team.strip(), away_team.strip(), 1.5, 1.2
                )
                
                results.append({
                    "home_team": home_team.strip(),
                    "away_team": away_team.strip(), 
                    "prediction": prediction,
                    "probabilities": probabilities,
                    "confidence": confidence
                })
        
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Format invalide: {str(e)}")