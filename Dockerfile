# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements d'abord (meilleur caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier l'application
COPY app/ ./app/
COPY src/ ./src/

# Copier les données de déploiement
COPY deployment_data/ ./data/

# Créer les dossiers nécessaires
RUN mkdir -p data/raw data/processed models mlruns logs

# Exposer le port pour Streamlit
EXPOSE 7860

# Commande pour lancer l'application
CMD ["streamlit", "run", "app/complete_dashboard.py", "--server.port=7860", "--server.address=0.0.0.0"]