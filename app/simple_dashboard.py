"""
Version simplifiée et robuste du dashboard
"""
import streamlit as st
import pandas as pd
import os
import sys

# Configuration basique
st.set_page_config(page_title="PL Predictor", layout="wide")
st.title("🏆 Premier League Predictor - DEBUG")

# Chargement des données simplifié
@st.cache_data
def load_data_simple():
    try:
        processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
        st.write(f"📁 Recherche dans: {processed_dir}")
        
        if not os.path.exists(processed_dir):
            st.error(f"❌ Dossier non trouvé: {processed_dir}")
            return None
            
        files = os.listdir(processed_dir)
        st.write(f"📄 Fichiers trouvés: {files}")
        
        csv_files = [f for f in files if f.startswith('premier_league_processed') and f.endswith('.csv')]
        st.write(f"📊 Fichiers CSV: {csv_files}")
        
        if not csv_files:
            st.error("❌ Aucun fichier processed trouvé")
            return None
            
        latest_file = sorted(csv_files)[-1]
        file_path = os.path.join(processed_dir, latest_file)
        st.write(f"✅ Chargement: {latest_file}")
        
        df = pd.read_csv(file_path)
        st.write(f"✅ Données chargées: {df.shape}")
        st.write(f"📋 Colonnes: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Erreur chargement: {e}")
        return None

def main():
    st.sidebar.title("Debug")
    
    st.header("1. Test Chargement Données")
    df = load_data_simple()
    
    if df is not None:
        st.success("✅ Données chargées avec succès!")
        
        # Afficher un aperçu
        st.subheader("Aperçu des données")
        st.dataframe(df.head(10))
        
        # Statistiques basiques
        st.subheader("Statistiques basiques")
        if 'result' in df.columns:
            st.write("Distribution résultats:", df['result'].value_counts())
        
        # Test de sélection d'équipes
        st.header("2. Test Sélection Équipes")
        if 'home_team' in df.columns:
            teams = sorted(pd.concat([df['home_team'], df['away_team']]).unique())
            st.write(f"Nombre d'équipes: {len(teams)}")
            
            col1, col2 = st.columns(2)
            with col1:
                home_team = st.selectbox("Équipe domicile", teams[:5])  # Juste les 5 premières pour tester
            with col2:
                away_team = st.selectbox("Équipe extérieur", teams[5:10])  # 5 suivantes
            
            st.write(f"Match sélectionné: {home_team} vs {away_team}")
            
            # Test calcul xG simple
            st.header("3. Test Calcul xG")
            try:
                home_matches = df[df['home_team'] == home_team]
                away_matches = df[df['away_team'] == away_team]
                
                if 'home_xg' in home_matches.columns:
                    home_xg = home_matches['home_xg'].mean()
                    st.write(f"xG moyen {home_team} (domicile): {home_xg:.2f}")
                
                if 'away_xg' in away_matches.columns:
                    away_xg = away_matches['away_xg'].mean()
                    st.write(f"xG moyen {away_team} (extérieur): {away_xg:.2f}")
                    
            except Exception as e:
                st.error(f"Erreur calcul xG: {e}")
    
    else:
        st.error("❌ Impossible de charger les données")

if __name__ == "__main__":
    main()