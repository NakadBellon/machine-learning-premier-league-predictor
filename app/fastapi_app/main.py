# app/fastapi_app/main.py 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
import sys
import os

# Ajouter le chemin de l'application au PYTHONPATH
sys.path.append('/app')
sys.path.append('/app/app')

try:
    from app.endpoints import predictions, montecarlo, analytics
    print("✅ Modules endpoints importés avec succès")
except ImportError as e:
    print(f"❌ Erreur import endpoints: {e}")
    # Fallback : import relatif
    from .endpoints import predictions, montecarlo, analytics

app = FastAPI(
    title="Premier League Predictor API 🏆",
    description="API de prédiction des matchs de Premier League utilisant le Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclure les routers
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["Predictions"])
app.include_router(montecarlo.router, prefix="/api/v1/montecarlo", tags=["Monte Carlo Simulations"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Premier League Predictor API 🏆</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    max-width: 800px; 
                    margin: 40px auto; 
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }
                .container {
                    background: rgba(255,255,255,0.1);
                    padding: 30px;
                    border-radius: 15px;
                    backdrop-filter: blur(10px);
                }
                a { 
                    color: #ffd700; 
                    text-decoration: none;
                    font-weight: bold;
                }
                a:hover { text-decoration: underline; }
                .links { margin-top: 30px; }
                .link-item { 
                    background: rgba(255,255,255,0.2); 
                    padding: 15px; 
                    margin: 10px 0; 
                    border-radius: 8px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏆 Premier League Predictor API</h1>
                <p><strong>Système complet de prédiction football avec Machine Learning</strong></p>
                
                <div class="links">
                    <div class="link-item">
                        <h3>📚 Documentation</h3>
                        <p><a href="/docs">Documentation Interactive (Swagger)</a></p>
                        <p><a href="/redoc">Documentation Alternative (ReDoc)</a></p>
                    </div>
                    
                    <div class="link-item">
                        <h3>🔗 Endpoints Disponibles</h3>
                        <p><strong>Prédictions :</strong> /api/v1/predictions/*</p>
                        <p><strong>Simulations :</strong> /api/v1/montecarlo/*</p>
                        <p><strong>Analytics :</strong> /api/v1/analytics/*</p>
                    </div>
                    
                    <div class="link-item">
                        <h3>⚡ Fonctionnalités</h3>
                        <ul>
                            <li>Prédiction des résultats de matchs (60.58% accuracy)</li>
                            <li>Simulation Monte Carlo de saison complète</li>
                            <li>Probabilités de titre, top 4 et relégation</li>
                            <li>Analyses statistiques avancées</li>
                        </ul>
                    </div>
                </div>
            </div>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Endpoint de santé de l'API"""
    return {
        "status": "healthy",
        "service": "Premier League Predictor API",
        "version": "1.0.0"
    }

@app.get("/api/status")
async def api_status():
    """Statut détaillé de l'API"""
    return {
        "status": "operational",
        "version": "1.0.0",
        "endpoints": {
            "predictions": "active",
            "monte_carlo": "active", 
            "analytics": "active"
        },
        "model_accuracy": "60.58%"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload en développement
        log_level="info"
    )