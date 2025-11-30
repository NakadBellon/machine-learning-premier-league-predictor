# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements d'abord (optimisation du cache Docker)
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier l'application
COPY app/ ./app/
COPY src/ ./src/
COPY deployment_data/ ./data/
COPY models/ ./models/
COPY scripts/ ./scripts/

# Créer les dossiers nécessaires
RUN mkdir -p data/raw data/processed mlruns logs

# Exposer les ports
EXPOSE 7860
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Commande par défaut (Streamlit)
CMD ["streamlit", "run", "app/streamlit_with_fastapi.py", "--server.port=7860", "--server.address=0.0.0.0"]