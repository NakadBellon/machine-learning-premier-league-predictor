# 📁 app/streamlit_with_fastapi.py
"""
Dashboard Streamlit complet - Système MLOps de prédiction football
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys
import requests

# Configuration de l'API FastAPI
# Utilise la variable d'environnement définie dans docker-compose.yml
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# Adapter les chemins pour Hugging Face Spaces
if 'SPACE_ID' in os.environ:
    DATA_PATH = "/data"
else:
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')

class CompletePredictor:
    """Classe qui utilise l'API FastAPI pour les prédictions"""
    
    def __init__(self):
        self.teams = self.load_teams()
    
    def load_teams(self):
        """Charge la liste des équipes depuis l'API"""
        try:
            response = requests.get(f"{API_BASE_URL}/predictions/teams", timeout=5)
            if response.status_code == 200:
                return response.json()["teams"]
        except:
            pass
        
        # Fallback si l'API n'est pas disponible
        return [
            'Manchester City', 'Liverpool', 'Arsenal', 'Chelsea', 'Manchester Utd',
            'Tottenham', 'Newcastle', 'Brighton', 'West Ham', 'Crystal Palace',
            'Wolves', 'Aston Villa', 'Brentford', 'Fulham', 'Nottingham Forest',
            'Luton Town', 'Ipswich Town', 'West Brom', 'Sheffield Utd'
        ]
    
    def get_team_xg_stats(self, team):
        """Récupère les statistiques xG moyennes d'une équipe"""
        try:
            # Essayer différents chemins pour les données
            possible_paths = [
                os.path.join(DATA_PATH, 'premier_league_data.csv'),
                os.path.join(DATA_PATH, 'processed', 'premier_league_with_features_20251111_123454.csv'),
                'deployment_data/premier_league_data.csv',
                '../deployment_data/premier_league_data.csv'
            ]
        
            df = None
            for path in possible_paths:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    break
                
            if df is None:
                # Données par défaut
                default_xg = {
                    'Manchester City': {'home_xg': 2.1, 'away_xg': 1.9},
                    'Liverpool': {'home_xg': 2.0, 'away_xg': 1.8},
                    'Arsenal': {'home_xg': 1.8, 'away_xg': 1.6},
                    'Chelsea': {'home_xg': 1.7, 'away_xg': 1.5},
                    'Manchester Utd': {'home_xg': 1.6, 'away_xg': 1.4}
                }
            
                if team in default_xg:
                    stats = default_xg[team]
                    return {
                        'home_xg': stats['home_xg'],
                        'away_xg': stats['away_xg'], 
                        'avg_xg': (stats['home_xg'] + stats['away_xg']) / 2
                    }
                else:
                    return {'home_xg': 1.5, 'away_xg': 1.2, 'avg_xg': 1.35}
        
            # Calculer xG moyen
            home_matches = df[df['home_team'] == team]
            home_xg = home_matches['home_xg'].mean() if not home_matches.empty else 1.5
        
            away_matches = df[df['away_team'] == team]
            away_xg = away_matches['away_xg'].mean() if not away_matches.empty else 1.2
        
            avg_xg = (home_xg + away_xg) / 2
        
            return {
                'home_xg': round(home_xg, 2),
                'away_xg': round(away_xg, 2),
                'avg_xg': round(avg_xg, 2)
            }
        
        except Exception as e:
            return {'home_xg': 1.5, 'away_xg': 1.2, 'avg_xg': 1.35}
    
    def predict_single_match(self, home_team, away_team, home_xg, away_xg):
        """Utilise l'API FastAPI pour prédire un match"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/predictions/match",
                json={
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_xg": home_xg,
                    "away_xg": away_xg
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["prediction"], data["probabilities"]
            else:
                st.error(f"Erreur API: {response.status_code}")
                return None, None
                
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur de connexion à l'API: {e}")
            return None, None
    
    def get_montecarlo_simulation(self, n_simulations=1000):
        """Récupère les simulations Monte Carlo depuis l'API"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/montecarlo/simulate",
                json={
                    "n_simulations": n_simulations,
                    "season": "2025-2026"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Erreur simulation: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur de connexion: {e}")
            return None
    
    def get_team_analysis(self, team_name):
        """Récupère l'analyse d'une équipe depuis l'API"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/analytics/team",
                json={"team_name": team_name},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except:
            return None

def main():
    """Application Streamlit principale avec intégration FastAPI"""
    
    # Configuration
    st.set_page_config(
        page_title="Premier League Predictor Pro",
        page_icon="🏆",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏆 Premier League Predictor Pro")
    st.markdown("**Système MLOps complet : Machine Learning + DevOps**")
    
    # Vérification de la connexion API
    # Extraire l'URL de base sans /api/v1
    api_health_url = API_BASE_URL.replace('/api/v1', '/health')
    
    try:
        health_response = requests.get(api_health_url, timeout=5)
        if health_response.status_code == 200:
            st.sidebar.success(f"✅ API FastAPI connectée")
        else:
            st.sidebar.warning("⚠️ API non accessible")
    except Exception as e:
        st.sidebar.error("❌ API FastAPI non connectée")
        st.sidebar.caption(f"Erreur: {str(e)}")
    
    # Initialisation du prédicteur
    predictor = CompletePredictor()
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Aller à:", [
        "🎯 Accueil", 
        "⚽ Prédire un Match", 
        "🎲 Vue Saison (Monte Carlo)",
        "📊 Analytics",
        "ℹ️ À propos"
    ])
    
    if page == "🎯 Accueil":
        show_home_page(predictor)
    elif page == "⚽ Prédire un Match":
        show_match_prediction_page(predictor)
    elif page == "🎲 Vue Saison (Monte Carlo)":
        show_montecarlo_page(predictor)
    elif page == "📊 Analytics":
        show_analytics_page(predictor)
    elif page == "ℹ️ À propos":
        show_about_page()

def show_home_page(predictor):
    """Page d'accueil avec vue d'ensemble"""
    
    st.header("📊 Tableau de Bord Complet")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Récupérer les données Monte Carlo pour les métriques
        monte_carlo_data = predictor.get_montecarlo_simulation(100)
        if monte_carlo_data:
            top_team = max(monte_carlo_data["championship_probabilities"].items(), key=lambda x: x[1])
            st.metric("🏆 Favorite Titre", top_team[0], f"{top_team[1]:.1%}")
        else:
            st.metric("🏆 Favorite Titre", "Man City", "76.4%")
    
    with col2:
        st.metric("🎯 Accuracy Modèle", "60.6%")
    
    with col3:
        st.metric("🎲 Simulations", "1,000")
    
    with col4:
        if monte_carlo_data:
            risky_team = max(monte_carlo_data["relegation_probabilities"].items(), key=lambda x: x[1])
            st.metric("🔻 Risque Relégation", risky_team[0], f"{risky_team[1]:.1%}")
        else:
            st.metric("🔻 Risque Relégation", "Luton Town", "99.9%")
    
    # Vue comparative
    st.subheader("🔍 Double Expertise : Data Science & DevOps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔬 Data Science & Machine Learning
        
        **Modélisation avancée :**
        - Feature engineering temporel
        - Régression logistique optimisée
        - Validation croisée temporelle
        - Métriques de performance rigoureuses
        
        **Résultats ML :**
        - **Accuracy : 60.58%** vs baseline 43.55%
        - Features : xG, forme des équipes, données historiques
        - 15,960 matchs analysés (2019-2026)
        - Simulations Monte Carlo pour l'incertitude
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 DevOps & MLOps
        
        **Architecture moderne :**
        - API RESTful avec FastAPI
        - Containerisation Docker complète
        - Orchestration Docker Compose
        - Découplage microservices
        
        **Infrastructure :**
        - CI/CD avec GitHub Actions
        - Versioning données (DVC)
        - Tracking ML (MLflow)
        - Déploiement cloud-ready
        """)
    
    
def show_match_prediction_page(predictor):
    """Page de prédiction de match avec API FastAPI"""
    
    st.header("⚽ Prédiction de Match")
    st.markdown("**Modèle de Machine Learning déployé via API RESTful**")
    
    # Sélection des équipes
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox(
            "Équipe à Domicile 🏠", 
            predictor.teams,
            index=predictor.teams.index("Liverpool") if "Liverpool" in predictor.teams else 0
        )
        
        home_stats = predictor.get_team_xg_stats(home_team)
        
        # Contexte de l'équipe
        team_analysis = predictor.get_team_analysis(home_team)
        if team_analysis:
            st.info(f"""
            **Contexte {home_team}:**
            - Titre: {team_analysis['championship_prob']:.1%}
            - Top 4: {team_analysis['top4_prob']:.1%}
            - Relégation: {team_analysis['relegation_prob']:.1%}
            - xG domicile: **{home_stats['home_xg']}**
            - Risque: {team_analysis['risk_level']}
            """)
        else:
            st.info(f"**xG Domicile {home_team}:** {home_stats['home_xg']}")
        
        home_xg = home_stats['home_xg']
    
    with col2:
        away_team = st.selectbox(
            "Équipe à l'Extérieur ✈️", 
            predictor.teams,
            index=predictor.teams.index("Manchester City") if "Manchester City" in predictor.teams else 1
        )
        
        away_stats = predictor.get_team_xg_stats(away_team)
        
        # Contexte de l'équipe
        team_analysis = predictor.get_team_analysis(away_team)
        if team_analysis:
            st.info(f"""
            **Contexte {away_team}:**
            - Titre: {team_analysis['championship_prob']:.1%}
            - Top 4: {team_analysis['top4_prob']:.1%}
            - Relégation: {team_analysis['relegation_prob']:.1%}
            - xG extérieur: **{away_stats['away_xg']}**
            - Risque: {team_analysis['risk_level']}
            """)
        else:
            st.info(f"**xG Extérieur {away_team}:** {away_stats['away_xg']}")
        
        away_xg = away_stats['away_xg']
    
    # Options avancées
    with st.expander("⚙️ Ajustement des paramètres ML (xG)"):
        col1, col2 = st.columns(2)
        with col1:
            home_xg_manual = st.slider(
                f"xG Domicile {home_team}", 
                0.5, 3.0, home_stats['home_xg'], 0.1,
                key="home_xg_manual"
            )
        with col2:
            away_xg_manual = st.slider(
                f"xG Extérieur {away_team}", 
                0.5, 3.0, away_stats['away_xg'], 0.1,
                key="away_xg_manual"
            )
        
        use_manual = st.checkbox("Utiliser les valeurs manuelles")
        if use_manual:
            home_xg = home_xg_manual
            away_xg = away_xg_manual
    
    # Prédiction
    if st.button("🎯 Calculer la Prédiction", type="primary"):
        with st.spinner("Appel de l'API ML..."):
            prediction, probabilities = predictor.predict_single_match(
                home_team, away_team, home_xg, away_xg
            )
            
            if prediction and probabilities:
                display_match_prediction(
                    home_team, away_team, prediction, probabilities, home_xg, away_xg
                )

def display_match_prediction(home_team, away_team, prediction, probabilities, home_xg, away_xg):
    """Affiche les résultats de la prédiction de match"""
    
    st.success("✅ Prédiction ML terminée !")
    
    # Paramètres utilisés
    st.info(f"**Features utilisées:** {home_team} (xG: {home_xg}) vs {away_team} (xG: {away_xg})")
    
    # Résultat principal
    result_text = {
        'H': f"Victoire de **{home_team}** 🏠",
        'A': f"Victoire de **{away_team}** ✈️", 
        'D': "**Match Nul** ⚖️"
    }
    
    st.subheader(f"📊 Résultat prédit: {result_text[prediction]}")
    
    # Probabilités détaillées
    col1, col2, col3 = st.columns(3)
    
    with col1:
        delta_home = probabilities['H'] - 0.33
        st.metric(
            f"✅ {home_team} gagne", 
            f"{probabilities['H']:.1%}",
            delta=f"{delta_home:+.1%}" if abs(delta_home) > 0.05 else None
        )
    
    with col2:
        delta_draw = probabilities['D'] - 0.33
        st.metric(
            "🤝 Match nul", 
            f"{probabilities['D']:.1%}",
            delta=f"{delta_draw:+.1%}" if abs(delta_draw) > 0.05 else None
        )
    
    with col3:
        delta_away = probabilities['A'] - 0.33
        st.metric(
            f"✅ {away_team} gagne", 
            f"{probabilities['A']:.1%}",
            delta=f"{delta_away:+.1%}" if abs(delta_away) > 0.05 else None
        )
    
    # Graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    outcomes = [f'{home_team}\ngagne', 'Match\nnul', f'{away_team}\ngagne']
    probs = [probabilities['H'], probabilities['D'], probabilities['A']]
    colors = ['#2E8B57', '#FFA500', '#1E90FF']
    
    bars = ax.bar(outcomes, probs, color=colors, alpha=0.8)
    ax.set_ylabel('Probabilité')
    ax.set_title('Distribution des Probabilités (Modèle ML)')
    ax.set_ylim(0, 1)
    
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
    
    st.pyplot(fig)

def show_montecarlo_page(predictor):
    """Page dédiée aux résultats Monte Carlo via API"""
    
    st.header("🎲 Simulation de Saison - Monte Carlo")
    st.markdown("**Méthode statistique pour l'analyse d'incertitude**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_simulations = st.slider("Nombre de simulations", 100, 5000, 1000, 100)
    
    with col2:
        if st.button("🔄 Lancer la Simulation", type="primary"):
            with st.spinner(f"Exécution de {n_simulations} simulations Monte Carlo..."):
                monte_carlo_data = predictor.get_montecarlo_simulation(n_simulations)
                
                if monte_carlo_data:
                    display_montecarlo_results(monte_carlo_data)
                else:
                    st.error("Erreur lors de la simulation")

def display_montecarlo_results(data):
    """Affiche les résultats Monte Carlo"""
    
    st.success(f"✅ Simulation terminée: {data['simulation_count']} saisons analysées")
    
    # Sélection de la vue
    view_option = st.radio(
        "Vue:",
        ["🏆 Championnat", "👑 Top 4", "🔻 Relégation"],
        horizontal=True
    )
    
    if view_option == "🏆 Championnat":
        # Tableau championnat
        champ_data = []
        for team, prob in sorted(data["championship_probabilities"].items(), 
                               key=lambda x: x[1], reverse=True):
            if prob > 0.001:
                champ_data.append({
                    'Équipe': team,
                    'Probabilité Titre': f"{prob:.1%}",
                    'Statut': 'Favorite' if prob > 0.5 else 'Candidate' if prob > 0.1 else 'Extérieure'
                })
        
        st.dataframe(pd.DataFrame(champ_data), use_container_width=True)
        
    elif view_option == "👑 Top 4":
        # Tableau Top 4
        top4_data = []
        for team, prob in sorted(data["top4_probabilities"].items(), 
                               key=lambda x: x[1], reverse=True):
            if prob > 0.01:
                status = "✅ Quasi-certain" if prob > 0.9 else "📈 Probable" if prob > 0.5 else "⚡ Possible"
                top4_data.append({
                    'Équipe': team,
                    'Probabilité Top 4': f"{prob:.1%}",
                    'Statut': status
                })
        
        st.dataframe(pd.DataFrame(top4_data), use_container_width=True)
        
    else:  # Relégation
        # Tableau relégation
        releg_data = []
        for team, prob in sorted(data["relegation_probabilities"].items(), 
                               key=lambda x: x[1], reverse=True):
            if prob > 0.1:
                risk = "🔴 Haut risque" if prob > 0.8 else "🟡 Risque moyen" if prob > 0.4 else "🟢 Faible risque"
                releg_data.append({
                    'Équipe': team,
                    'Probabilité Relégation': f"{prob:.1%}",
                    'Niveau de risque': risk
                })
        
        st.dataframe(pd.DataFrame(releg_data), use_container_width=True)

def show_analytics_page(predictor):
    """Page analytics avec données API"""
    
    st.header("📊 Analytics des Équipes")
    
    selected_team = st.selectbox(
        "Sélectionnez une équipe", 
        predictor.teams,
        index=0
    )
    
    if st.button("🔍 Analyser l'équipe"):
        with st.spinner("Récupération des données..."):
            team_analysis = predictor.get_team_analysis(selected_team)
            
            if team_analysis:
                display_team_analysis(team_analysis)
            else:
                st.error("Données non disponibles")

def display_team_analysis(analysis):
    """Affiche l'analyse d'une équipe"""
    
    st.subheader(f"📈 Analyse de {analysis['team_name']}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏆 Titre", f"{analysis['championship_prob']:.1%}")
    
    with col2:
        st.metric("👑 Top 4", f"{analysis['top4_prob']:.1%}")
    
    with col3:
        st.metric("🔻 Relégation", f"{analysis['relegation_prob']:.1%}")
    
    with col4:
        st.metric("📊 Forme", f"{analysis['form_rating']:.0%}")
    
    # Statistiques détaillées
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("⚽ xG Domicile", f"{analysis['avg_home_xg']:.2f}")
        st.metric("🎯 Niveau de risque", analysis['risk_level'])
    
    with col2:
        st.metric("⚽ xG Extérieur", f"{analysis['avg_away_xg']:.2f}")
        st.metric("📈 Performance", "Élevée" if analysis['form_rating'] > 0.7 else "Moyenne" if analysis['form_rating'] > 0.4 else "Faible")

def show_about_page():
    """Page À propos"""
    
    st.header("ℹ️ À propos du Projet")
    
    st.markdown("""
    ### 🎯 Projet MLOps Complet : Data Science & DevOps
    
    **Double expertise démontrée à travers ce système de prédiction football :**
    
    ### 🔬 Partie Data Science & Machine Learning
    
    **Modélisation Prédictive :**
    - **Dataset** : 15,960 matchs de Premier League (2019-2026)
    - **Features** : xG (Expected Goals), forme des équipes, données temporelles
    - **Modèle** : Régression logistique avec validation croisée temporelle
    - **Performance** : **60.58% accuracy** vs baseline de 43.55% (+17.03%)
    - **Méthodes** : Feature engineering, simulations Monte Carlo, analyse d'incertitude
    
    **Approche Scientifique :**
    - Validation rigoureuse des modèles
    - Analyse des features importance
    - Gestion du temps dans les données sportives
    - Métriques business-aligned
    
    ### 🚀 Partie DevOps & MLOps
    
    **Architecture Cloud-Native :**
    - **API First** : FastAPI avec documentation auto-générée
    - **Containerisation** : Docker + Docker Compose
    - **Microservices** : Frontend/Backend découplés
    - **CI/CD** : GitHub Actions pour l'intégration continue
    
    **Practices MLOps :**
    - **Versioning** : DVC pour les données et modèles
    - **Tracking** : MLflow pour les expériences ML
    - **Monitoring** : Health checks et métriques
    - **Déploiement** : Architecture prête pour le cloud
    
    ### 🏗️ Stack Technique Complète
    
    **Machine Learning :**
    - Scikit-learn, pandas, numpy
    - Feature engineering temporel
    - Validation croisée
    - Optimisation hyperparamètres
    
    **DevOps & Infrastructure :**
    - FastAPI, Streamlit
    - Docker, Docker Compose
    - GitHub Actions
    - Architecture RESTful
    
    ### 📈 Résultats Concrets
    
    - ✅ **Modèle ML performant** (60.58% accuracy)
    - ✅ **API scalable** avec documentation complète
    - ✅ **Architecture containerisée** prête pour la production
    - ✅ **Pipeline MLOps** de bout en bout
    - ✅ **Déploiement cloud-ready** sur Hugging Face Spaces
    
    Ce projet démontre la capacité à mener un projet data de l'exploration à la mise en production, 
    en combinant rigueur scientifique et expertise technique.
    """)

if __name__ == "__main__":
    main()