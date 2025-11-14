"""
Dashboard Streamlit pour les prédictions Premier League - AVEC VRAI MODÈLE
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime

# Ajouter le chemin pour importer nos modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.model_loader import ModelLoader

# Configuration de la page
st.set_page_config(
    page_title="Premier League Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🏆 Premier League Predictor")
st.markdown("""
Système de prédiction des matchs de Premier League utilisant le Machine Learning
""")

# Initialisation du modèle
@st.cache_resource
def load_model():
    """Charge le modèle une fois pour toutes les sessions"""
    model_loader = ModelLoader()
    success = model_loader.load_latest_model()
    return model_loader if success else None

def load_data():
    """Charge les données les plus récentes"""
    try:
        processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
        files = [f for f in os.listdir(processed_dir) 
                if f.startswith('premier_league_processed') and f.endswith('.csv')]
        
        if files:
            latest_file = sorted(files)[-1]
            file_path = os.path.join(processed_dir, latest_file)
            df = pd.read_csv(file_path)
            return df
        return None
    except:
        return None

def main():
    # Chargement du modèle
    model_loader = load_model()
    
    if model_loader is None:
        st.error("""
        ❌ **Modèle non chargé**
        
        Veuillez d'abord entraîner le modèle en exécutant:
        ```bash
        python src/modeling/corrected_baseline.py
        ```
        """)
        return

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Aller à:", ["🏠 Dashboard", "🔮 Prédictions", "📊 Analyse", "ℹ️ À propos"])
    
    # Chargement des données
    df = load_data()
    
    if page == "🏠 Dashboard":
        show_dashboard(df, model_loader)
    elif page == "🔮 Prédictions":
        show_predictions(df, model_loader)
    elif page == "📊 Analyse":
        show_analysis(df)
    else:
        show_about(model_loader)

def show_dashboard(df, model_loader):
    """Page dashboard principal"""
    st.header("📊 Dashboard Overview")
    
    # Métriques du modèle
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🤖 Modèle", "Regression Logistique")
    
    with col2:
        st.metric("🎯 Accuracy", "60.5%")
    
    with col3:
        st.metric("💪 Amélioration", "+17.0% vs baseline")
    
    if df is not None:
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_matches = len(df)
            st.metric("Total Matchs", f"{total_matches:,}")
        
        with col2:
            seasons = df['season'].nunique() if 'season' in df.columns else 'N/A'
            st.metric("Saisons", seasons)
        
        with col3:
            teams = pd.concat([df['home_team'], df['away_team']]).nunique() if 'home_team' in df.columns else 'N/A'
            st.metric("Équipes", teams)
        
        with col4:
            home_wins = len(df[df['result'] == 'H']) if 'result' in df.columns else 'N/A'
            st.metric("Victoires Domicile", home_wins)
        
        # Distribution des résultats
        st.subheader("📈 Distribution des Résultats")
        if 'result' in df.columns:
            result_counts = df['result'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # H, A, D
            
            bars = ax.bar(['Domicile', 'Extérieur', 'Nul'], 
                         [result_counts.get('H', 0), result_counts.get('A', 0), result_counts.get('D', 0)],
                         color=colors)
            
            ax.set_ylabel('Nombre de Matchs')
            ax.set_title('Distribution des Résultats')
            
            # Ajouter les pourcentages
            total = sum(result_counts)
            for i, bar in enumerate(bars):
                height = bar.get_height()
                percentage = (height / total) * 100
                ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                       f'{percentage:.1f}%', ha='center', va='bottom')
            
            st.pyplot(fig)
    
    else:
        st.warning("❌ Aucune donnée chargée.")

def show_predictions(df, model_loader):
    """Page de prédictions avec VRAI MODÈLE"""
    st.header("🔮 Prédictions de Matchs - Modèle Réel")
    
    st.info("""
    🎯 **Prédictions utilisant le modèle de Regression Logistique entraîné**
    - **Accuracy**: 60.5% 
    - **Features**: Expected Goals (xG) historiques
    """)
    
    if df is not None and 'home_team' in df.columns:
        # Sélection des équipes
        col1, col2 = st.columns(2)
        
        with col1:
            teams = sorted(pd.concat([df['home_team'], df['away_team']]).unique())
            home_team = st.selectbox("Équipe Domicile", teams, key="home_select")
        
        with col2:
            away_options = [team for team in teams if team != home_team]
            away_team = st.selectbox("Équipe Extérieur", away_options, key="away_select")
        
        # Récupération des stats historiques
        home_avg_xg, home_form = model_loader.get_team_historical_stats(df, home_team, is_home=True)
        away_avg_xg, away_form = model_loader.get_team_historical_stats(df, away_team, is_home=False)
        
        # Configuration des features
        st.subheader("⚙️ Configuration des Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**{home_team}** (Domicile)")
            home_xg = st.slider("xG Moyen Domicile", 0.0, 5.0, float(home_avg_xg), 0.1, key="home_xg")
            home_form_input = st.slider("Forme (pts/5 matchs)", 0.0, 15.0, float(home_form), 0.5, key="home_form")
        
        with col2:
            st.write(f"**{away_team}** (Extérieur)")
            away_xg = st.slider("xG Moyen Extérieur", 0.0, 5.0, float(away_avg_xg), 0.1, key="away_xg")
            away_form_input = st.slider("Forme (pts/5 matchs)", 0.0, 15.0, float(away_form), 0.5, key="away_form")
        
        # Bouton de prédiction
        if st.button("🎯 Prédire le Résultat", type="primary"):
            with st.spinner("Calcul des probabilités avec le modèle ML..."):
                try:
                    # PRÉDICTION AVEC LE VRAI MODÈLE
                    home_win_prob, draw_prob, away_win_prob, predicted_result = model_loader.predict_match(
                        home_xg, away_xg, home_form_input, away_form_input
                    )
                    
                    # Affichage des résultats
                    st.subheader("📊 Probabilités de Résultat")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Victoire Domicile", f"{home_win_prob*100:.1f}%")
                    
                    with col2:
                        st.metric("Match Nul", f"{draw_prob*100:.1f}%")
                    
                    with col3:
                        st.metric("Victoire Extérieur", f"{away_win_prob*100:.1f}%")
                    
                    # Graphique
                    fig, ax = plt.subplots(figsize=(10, 6))
                    outcomes = ['Victoire\nDomicile', 'Match\nNul', 'Victoire\nExtérieur']
                    probabilities = [home_win_prob, draw_prob, away_win_prob]
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                    
                    bars = ax.bar(outcomes, probabilities, color=colors)
                    ax.set_ylabel('Probabilité')
                    ax.set_ylim(0, 1)
                    ax.set_title(f'Probabilités de Résultat\n{home_team} vs {away_team}')
                    
                    # Ajouter les pourcentages
                    for bar, prob in zip(bars, probabilities):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{prob*100:.1f}%', ha='center', va='bottom', fontweight='bold')
                    
                    st.pyplot(fig)
                    
                    # Résultat prédit
                    result_text = {
                        'H': f"**{home_team}** devrait l'emporter à domicile",
                        'D': "Le match pourrait se terminer sur un **match nul**",
                        'A': f"**{away_team}** pourrait créer la surprise à l'extérieur"
                    }
                    
                    st.success(f"🎯 **Résultat prédit**: {result_text.get(predicted_result, 'Indéterminé')}")
                    
                    # Détails techniques
                    with st.expander("🔍 Détails techniques"):
                        st.write(f"**Modèle utilisé**: Regression Logistique")
                        st.write(f"**Features**: {model_loader.feature_names}")
                        st.write(f"**xG domicile**: {home_xg}")
                        st.write(f"**xG extérieur**: {away_xg}")
                        if home_form_input != 6.0:
                            st.write(f"**Forme domicile**: {home_form_input} pts/5 matchs")
                        if away_form_input != 6.0:
                            st.write(f"**Forme extérieur**: {away_form_input} pts/5 matchs")
                
                except Exception as e:
                    st.error(f"❌ Erreur lors de la prédiction: {e}")
                    st.info("💡 Essayez avec des valeurs de xG différentes")
    
    else:
        st.warning("❌ Données insuffisantes pour les prédictions.")

def show_analysis(df):
    """Page d'analyse des données"""
    st.header("📊 Analyse des Données")
    
    if df is not None:
        # Statistiques par équipe
        st.subheader("🏟️ Performance par Équipe")
        
        if all(col in df.columns for col in ['home_team', 'away_team', 'home_score', 'away_score', 'result']):
            team_stats = {}
            
            for team in pd.concat([df['home_team'], df['away_team']]).unique():
                home_games = df[df['home_team'] == team]
                away_games = df[df['away_team'] == team]
                
                total_games = len(home_games) + len(away_games)
                wins = len(home_games[home_games['result'] == 'H']) + len(away_games[away_games['result'] == 'A'])
                draws = len(home_games[home_games['result'] == 'D']) + len(away_games[away_games['result'] == 'D'])
                
                team_stats[team] = {
                    'Total': total_games,
                    'Victoires': wins,
                    'Nuls': draws,
                    'Défaites': total_games - wins - draws,
                    '% Victoires': (wins / total_games * 100) if total_games > 0 else 0
                }
            
            stats_df = pd.DataFrame(team_stats).T
            stats_df = stats_df.sort_values('% Victoires', ascending=False)
            
            st.dataframe(stats_df, use_container_width=True)
    
    else:
        st.warning("❌ Aucune donnée disponible pour l'analyse.")

def show_about(model_loader):
    """Page À propos"""
    st.header("ℹ️ À propos")
    
    st.markdown(f"""
    ## Premier League Predictor
    
    Cette application utilise le Machine Learning pour prédire les résultats des matchs de Premier League.
    
    ### 🎯 Fonctionnalités
    - **Prédictions en temps réel** avec modèle entraîné
    - **Analyse statistique** des performances des équipes
    - **Visualisation interactive** des données
    - **Tracking MLOps** avec MLflow
    
    ### 🤖 Modèle Actuel
    - **Algorithm**: Regression Logistique
    - **Accuracy**: 60.5%
    - **Amélioration vs baseline**: +17.0%
    - **Features utilisées**: {model_loader.feature_names}
    
    ### 📊 Données
    - **Période**: 2019-2026
    - **Matchs**: 15,960 matchs historiques
    - **Source**: FBref
    
    ### 🛠️ Stack Technique
    - Python, Scikit-learn
    - Streamlit pour l'interface
    - MLflow pour le tracking des modèles
    - DVC pour le versioning des données
    """)

if __name__ == "__main__":
    main()