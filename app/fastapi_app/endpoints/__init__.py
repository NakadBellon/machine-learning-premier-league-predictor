# app/fastapi_app/endpoints/__init__.py
from .predictions import router as predictions_router
from .montecarlo import router as montecarlo_router  
from .analytics import router as analytics_router

__all__ = ["predictions_router", "montecarlo_router", "analytics_router"]