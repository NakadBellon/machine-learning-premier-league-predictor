"""
Simulateur Monte Carlo pour la Premier League
Simule 10,000 saisons pour prédire les probabilités de titre, top 4, relégation, etc.
"""

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import mlflow

class MonteCarloSimulator:
    """Simulateur de saison complète par Monte Carlo"""
    
    def __init__(self, n_simulations=10000):
        self.n_simulations = n_simulations
        self.logger = logging.getLogger(__name__)
        self.results = {}
        
    def load_data(self):
        """Charge les données et prépare les fixtures"""
        try:
            processed_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
            processed_dir = os.path.abspath(processed_dir)
            
            files = [f for f in os.listdir(processed_dir)
                    if f.startswith('premier_league_with_features') and f.endswith('.csv')]
            
            latest_file = sorted(files)[-1]
            file_path = os.path.join(processed_dir, latest_file)
            
            self.logger.info(f"Chargement des données: {latest_file}")
            df = pd.read_csv(file_path)
            
            # Préparer les fixtures (matchs restants)
            current_season = "2025-2026"  # Saison actuelle à simuler
            fixtures = self.prepare_fixtures(df, current_season)
            
            return df, fixtures, current_season
            
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement données: {e}")
            raise
    
    def prepare_fixtures(self, df, current_season):
        """Prépare les fixtures de la saison actuelle"""
        # Pour la démo, on utilise les matchs de la dernière saison complète
        last_complete_season = "2024-2025"
        fixtures = df[df['season'] == last_complete_season].copy()
        
        # Simuler que ces matchs sont à venir
        fixtures['simulated'] = True
        
        self.logger.info(f"📅 {len(fixtures)} fixtures préparées pour la saison {current_season}")
        return fixtures
    
    def predict_match_probabilities(self, home_team, away_team, home_xg, away_xg):
        """Prédit les probabilités d'un match (version simplifiée)"""
        # Basé sur votre modèle à 60.58%
        # Pour la démo, on utilise une logique simple basée sur xG
        
        total_xg = home_xg + away_xg + 0.1  # Éviter division par 0
        
        prob_home_win = (home_xg / total_xg) * 0.8 + 0.1  # Pondération
        prob_away_win = (away_xg / total_xg) * 0.8 + 0.1
        prob_draw = 1 - prob_home_win - prob_away_win
        
        # Normalisation
        total = prob_home_win + prob_away_win + prob_draw
        prob_home_win /= total
        prob_away_win /= total
        prob_draw /= total
        
        return {
            'H': prob_home_win,
            'A': prob_away_win, 
            'D': prob_draw
        }
    
    def simulate_match(self, probabilities):
        """Simule le résultat d'un match basé sur les probabilités"""
        outcomes = ['H', 'D', 'A']
        probs = [probabilities['H'], probabilities['D'], probabilities['A']]
        
        return np.random.choice(outcomes, p=probs)
    
    def calculate_standings(self, results):
        """Calcule le classement à partir des résultats"""
        teams = set()
        for match in results:
            teams.add(match['home_team'])
            teams.add(match['away_team'])
        
        standings = {team: {'points': 0, 'goals_for': 0, 'goals_against': 0} for team in teams}
        
        for match in results:
            home_team = match['home_team']
            away_team = match['away_team']
            result = match['result']
            
            # Points
            if result == 'H':
                standings[home_team]['points'] += 3
            elif result == 'A':
                standings[away_team]['points'] += 3
            else:  # Draw
                standings[home_team]['points'] += 1
                standings[away_team]['points'] += 1
            
            # Buts (simplifié)
            standings[home_team]['goals_for'] += match.get('home_score', 1)
            standings[away_team]['goals_for'] += match.get('away_score', 1)
            standings[home_team]['goals_against'] += match.get('away_score', 1)
            standings[away_team]['goals_against'] += match.get('home_score', 1)
        
        # Trier par points
        sorted_standings = sorted(standings.items(), key=lambda x: x[1]['points'], reverse=True)
        
        return sorted_standings
    
    def run_simulation(self):
        """Exécute la simulation Monte Carlo complète"""
        self.logger.info(f"🎲 Début simulation Monte Carlo ({self.n_simulations} saisons)")
        
        df, fixtures, current_season = self.load_data()
        
        # Initialiser les compteurs
        teams = set(pd.concat([fixtures['home_team'], fixtures['away_team']]).unique())
        championship_counts = {team: 0 for team in teams}
        top4_counts = {team: 0 for team in teams}
        relegation_counts = {team: 0 for team in teams}
        
        # Simulation
        for sim in tqdm(range(self.n_simulations), desc="Simulating seasons"):
            season_results = []
            
            # Simuler chaque match de la saison
            for _, match in fixtures.iterrows():
                home_team = match['home_team']
                away_team = match['away_team']
                home_xg = match['home_xg']
                away_xg = match['away_xg']
                
                # Obtenir les probabilités
                probs = self.predict_match_probabilities(home_team, away_team, home_xg, away_xg)
                
                # Simuler le résultat
                result = self.simulate_match(probs)
                
                season_results.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'result': result,
                    'home_score': match.get('home_score', 1),
                    'away_score': match.get('away_score', 1)
                })
            
            # Calculer le classement
            standings = self.calculate_standings(season_results)
            
            # Mettre à jour les compteurs
            champion = standings[0][0]
            championship_counts[champion] += 1
            
            top4_teams = [standings[i][0] for i in range(4)]
            for team in top4_teams:
                top4_counts[team] += 1
            
            relegated_teams = [standings[i][0] for i in range(-3, 0)]  # 3 derniers
            for team in relegated_teams:
                relegation_counts[team] += 1
        
        # Calculer les probabilités
        self.results = {
            'championship_prob': {team: count/self.n_simulations for team, count in championship_counts.items()},
            'top4_prob': {team: count/self.n_simulations for team, count in top4_counts.items()},
            'relegation_prob': {team: count/self.n_simulations for team, count in relegation_counts.items()},
            'simulation_count': self.n_simulations,
            'season': current_season
        }
        
        return self.results
    
    def analyze_results(self):
        """Analyse et présente les résultats de la simulation"""
        if not self.results:
            self.logger.error("❌ Aucun résultat à analyser. Exécutez run_simulation() d'abord.")
            return
        
        # Classement par probabilité de titre
        champ_probs = sorted(self.results['championship_prob'].items(), key=lambda x: x[1], reverse=True)
        top4_probs = sorted(self.results['top4_prob'].items(), key=lambda x: x[1], reverse=True)
        releg_probs = sorted(self.results['relegation_prob'].items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 SIMULATION MONTE CARLO - {self.results['season']}")
        print(f"📊 {self.results['simulation_count']:,} saisons simulées")
        print("\n" + "="*50)
        
        # Champions probables
        print("\n🎯 PROBABILITÉS DE TITRE:")
        for team, prob in champ_probs[:5]:  # Top 5
            if prob > 0.01:  # Seulement si probabilité significative
                print(f"   {team}: {prob:.1%}")
        
        # Top 4
        print("\n👑 PROBABILITÉS TOP 4:")
        for team, prob in top4_probs[:8]:  # Top 8
            if prob > 0.05:
                print(f"   {team}: {prob:.1%}")
        
        # Relégation
        print("\n🔻 PROBABILITÉS DE RELÉGATION:")
        for team, prob in releg_probs[:6]:  # Top 6 risques
            if prob > 0.05:
                print(f"   {team}: {prob:.1%}")
    
    def create_visualizations(self):
        """Crée des visualisations des résultats"""
        if not self.results:
            self.logger.error("❌ Aucun résultat à visualiser")
            return
        
        # Top 10 pour le titre
        champ_probs = sorted(self.results['championship_prob'].items(), key=lambda x: x[1], reverse=True)[:10]
        teams = [item[0] for item in champ_probs]
        probs = [item[1] for item in champ_probs]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Graphique 1: Probabilités de titre
        ax1.barh(teams, probs, color='gold', alpha=0.7)
        ax1.set_xlabel('Probabilité de titre')
        ax1.set_title('🏆 Probabilités de Championnat (Top 10)')
        ax1.grid(axis='x', alpha=0.3)
        
        # Graphique 2: Top 4 probabilités
        top4_probs = sorted(self.results['top4_prob'].items(), key=lambda x: x[1], reverse=True)[:10]
        ax2.barh([item[0] for item in top4_probs], [item[1] for item in top4_probs], color='blue', alpha=0.7)
        ax2.set_xlabel('Probabilité Top 4')
        ax2.set_title('👑 Probabilités de Qualification Ligue des Champions')
        ax2.grid(axis='x', alpha=0.3)
        
        # Graphique 3: Relégation
        releg_probs = sorted(self.results['relegation_prob'].items(), key=lambda x: x[1], reverse=True)[:10]
        ax3.barh([item[0] for item in releg_probs], [item[1] for item in releg_probs], color='red', alpha=0.7)
        ax3.set_xlabel('Probabilité de Relégation')
        ax3.set_title('🔻 Probabilités de Relégation')
        ax3.grid(axis='x', alpha=0.3)
        
        # Graphique 4: Comparaison
        comparison_data = []
        for team in teams[:5]:  # Top 5 pour le titre
            comparison_data.append({
                'Team': team,
                'Champion': self.results['championship_prob'][team],
                'Top 4': self.results['top4_prob'][team],
                'Relegation': self.results['relegation_prob'][team]
            })
        
        df_comp = pd.DataFrame(comparison_data)
        df_comp.plot(x='Team', kind='bar', ax=ax4, alpha=0.7)
        ax4.set_title('📊 Comparaison des Probabilités (Top 5)')
        ax4.set_ylabel('Probabilité')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('monte_carlo_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info("📊 Visualisations sauvegardées: monte_carlo_results.png")

def main():
    """Fonction principale"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🎲 LANCEMENT SIMULATEUR MONTE CARLO")
        
        # MLflow tracking
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        mlflow_tracking_uri = "file:///" + os.path.abspath(os.path.join(project_root, "mlruns")).replace("\\", "/")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("Premier_League_Monte_Carlo")
        
        with mlflow.start_run(run_name=f"monte_carlo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Simulation
            simulator = MonteCarloSimulator(n_simulations=1000)  # 1000 pour test rapide
            results = simulator.run_simulation()
            
            # Analyse
            simulator.analyze_results()
            
            # Visualisations
            simulator.create_visualizations()
            
            # Logguer dans MLflow
            mlflow.log_param("n_simulations", simulator.n_simulations)
            mlflow.log_metric("top_champion_prob", max(results['championship_prob'].values()))
            
            logger.info("✅ SIMULATION TERMINÉE AVEC SUCCÈS")
            
        return results
        
    except Exception as e:
        logger.error(f"❌ ERREUR: {e}")
        raise

if __name__ == "__main__":
    main()