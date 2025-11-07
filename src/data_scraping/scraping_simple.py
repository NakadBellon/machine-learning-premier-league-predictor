# scraping_simple.py - VERSION CORRIGÉE
import pandas as pd
import soccerdata as sd
from datetime import datetime
import os

print("🔧 Initialisation du scraping...")

# Crée les dossiers nécessaires
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Configuration simple
SEASONS = ['2022-2023', '2023-2024']

def scrape_premier_league():
    """Scraping avec la BONNE syntaxe SoccerData"""
    print("🚀 Début du scraping Premier League...")
    
    try:
        # Initialise le scraper AVEC le league_code
        fbref = sd.FBref(leagues='ENG-Premier League')
        print("✅ FBref initialisé pour Premier League")
        
        all_matches = []
        
        for season in SEASONS:
            print(f"📅 Scraping saison {season}...")
            
            try:
                # BONNE syntaxe : pas de paramètre 'league'
                matches = fbref.read_schedule(season)
                matches['season'] = season
                all_matches.append(matches)
                print(f"✅ {len(matches)} matchs récupérés pour {season}")
                
            except Exception as e:
                print(f"⚠️ Erreur saison {season}: {e}")
                continue
        
        if not all_matches:
            print("❌ Aucune donnée récupérée")
            return None
        
        # Combine toutes les données
        df = pd.concat(all_matches, ignore_index=True)
        print(f"🎉 TOTAL: {len(df)} matchs récupérés")
        
        return df
        
    except Exception as e:
        print(f"💥 Erreur critique: {e}")
        return None

def clean_data(df):
    """Nettoie les données"""
    print("🧹 Nettoyage des données...")
    
    df_clean = df.copy()
    
    # 1. Extraire les scores
    if 'score' in df_clean.columns:
        scores = df_clean['score'].str.split('–', expand=True)
        df_clean['home_score'] = pd.to_numeric(scores[0], errors='coerce')
        df_clean['away_score'] = pd.to_numeric(scores[1], errors='coerce')
        print("✅ Scores extraits")
    
    # 2. Nettoyer les dates
    if 'date' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        print("✅ Dates nettoyées")
    
    # 3. Supprimer les matchs sans scores
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['home_score', 'away_score'])
    final_count = len(df_clean)
    
    print(f"✅ {final_count}/{initial_count} matchs après nettoyage")
    
    return df_clean

def main():
    """Point d'entrée principal"""
    print("=" * 50)
    print("🏆 PREMIER LEAGUE SCRAPER - VERSION CORRIGÉE")
    print("=" * 50)
    
    # Étape 1: Scraping
    raw_data = scrape_premier_league()
    if raw_data is None:
        print("❌ Échec du scraping")
        return
    
    # Étape 2: Nettoyage
    cleaned_data = clean_data(raw_data)
    
    # Étape 3: Sauvegarde
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    raw_path = f'data/raw/matches_raw_{timestamp}.csv'
    clean_path = f'data/processed/matches_clean_{timestamp}.csv'
    
    raw_data.to_csv(raw_path, index=False)
    cleaned_data.to_csv(clean_path, index=False)
    
    print(f"💾 Fichiers sauvegardés:")
    print(f"   - Données brutes: {raw_path}")
    print(f"   - Données nettoyées: {clean_path}")
    
    # Étape 4: Statistiques
    print("\n📊 STATISTIQUES:")
    print(f"   • Matchs totaux: {len(cleaned_data)}")
    print(f"   • Saisons: {cleaned_data['season'].nunique()}")
    print(f"   • Équipes uniques: {pd.concat([cleaned_data['home_team'], cleaned_data['away_team']]).nunique()}")
    
    if 'date' in cleaned_data.columns:
        print(f"   • Période: {cleaned_data['date'].min().strftime('%d/%m/%Y')} - {cleaned_data['date'].max().strftime('%d/%m/%Y')}")
    
    print("\n🎉 SCRAPING TERMINÉ AVEC SUCCÈS!")
    
    # Aperçu des données
    print("\n👀 APERÇU DES DONNÉES:")
    print(cleaned_data[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'season']].head(10).to_string())

if __name__ == "__main__":
    main()