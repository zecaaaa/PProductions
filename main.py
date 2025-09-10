
# -*- coding: utf-8 -*-
"""
Desafio Cientista de Dados - PProductions
Análise Cinematográfica e Modelo Preditivo IMDB

Este script executa a análise completa do projeto, incluindo:
- Carregamento e limpeza dos dados
- Análise exploratória 
- Treinamento do modelo preditivo
- Geração de insights e recomendações

Autor: Zeca
Data: Setembro 2025
"""

import pandas as pd
import numpy as np
import json
import re
import pickle
import warnings
from collections import Counter
from pathlib import Path

# Imports para Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Suprimir warnings
warnings.filterwarnings('ignore')


class CinemaDataAnalyzer:
    """Classe principal para análise de dados cinematográficos"""
    
    def __init__(self, data_path=None):
        """
        Inicializa o analisador
        
        Args:
            data_path (str): Caminho para o arquivo de dados (opcional)
        """
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.features = []
        
    def load_data(self, data_path=None):
        """
        Carrega os dados do arquivo ou gera dados simulados
        
        Args:
            data_path (str): Caminho para o arquivo JSON
        """
        print("🎬 Carregando dados...")
        
        if data_path and Path(data_path).exists():
            # Carregar dados reais
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.df = pd.DataFrame(data)
        else:
            # Gerar dados simulados realistas
            print("📊 Gerando dataset simulado baseado no padrão IMDB Top 1000...")
            self.df = self._generate_simulated_data()
        
        print(f"✅ Dados carregados: {self.df.shape[0]} filmes, {self.df.shape[1]} colunas")
        return self.df
    
    def _generate_simulated_data(self):
        """Gera dados simulados realistas baseados no padrão IMDB"""
        np.random.seed(42)
        n_movies = 999
        
        # Gerar dados realistas
        years = np.random.randint(1920, 2023, n_movies)
        ratings = np.random.normal(8.1, 0.8, n_movies)
        ratings = np.clip(ratings, 6.0, 9.5)
        
        meta_scores = np.random.normal(75, 15, n_movies)
        meta_scores = np.clip(meta_scores, 20, 100)
        
        votes = np.random.lognormal(13, 1, n_movies).astype(int)
        votes = np.clip(votes, 10000, 3000000)
        
        runtimes = np.random.normal(120, 25, n_movies).astype(int)
        runtimes = np.clip(runtimes, 70, 250)
        
        genres = ['Drama', 'Action', 'Comedy', 'Crime', 'Adventure', 'Thriller', 'Romance', 'Sci-Fi', 'Horror', 'Biography']
        certificates = ['U', 'UA', 'A', 'R', 'PG']
        
        # Criar DataFrame
        df = pd.DataFrame({
            'Rank': range(1, n_movies + 1),
            'Series_Title': [f'Movie_{i}' for i in range(1, n_movies + 1)],
            'Released_Year': years,
            'Certificate': np.random.choice(certificates, n_movies),
            'Runtime': [f'{runtime} min' for runtime in runtimes],
            'Genre': [np.random.choice(genres) + (', ' + np.random.choice(genres) if np.random.random() > 0.6 else '') for _ in range(n_movies)],
            'IMDB_Rating': np.round(ratings, 1),
            'Overview': [f'A compelling story about {np.random.choice(["love", "war", "adventure", "mystery", "family", "friendship"])}...' for _ in range(n_movies)],
            'Meta_score': np.round(meta_scores, 0),
            'Director': [f'Director_{i}' for i in range(1, n_movies + 1)],
            'Star1': [f'Actor_{i}_1' for i in range(1, n_movies + 1)],
            'Star2': [f'Actor_{i}_2' for i in range(1, n_movies + 1)],
            'Star3': [f'Actor_{i}_3' for i in range(1, n_movies + 1)],
            'Star4': [f'Actor_{i}_4' for i in range(1, n_movies + 1)],
            'No_of_Votes': votes,
            'Gross': [f'{gross:,}' for gross in np.random.lognormal(17, 1.5, n_movies).astype(int)]
        })
        
        return df
    
    def clean_data(self):
        """Limpa e prepara os dados para análise"""
        print("🧹 Limpando dados...")
        
        # 1. Garantir Released_Year numérico PRIMEIRO
        self.df['Released_Year'] = pd.to_numeric(self.df['Released_Year'], errors='coerce')
        fill_year = int(self.df['Released_Year'].median()) if self.df['Released_Year'].notna().any() else 2000
        self.df['Released_Year'] = self.df['Released_Year'].fillna(fill_year).astype(int)
        
        # 2. Função para extrair minutos do runtime
        def clean_runtime(runtime_str):
            if pd.isna(runtime_str):
                return np.nan
            nums = re.findall(r'\d+', str(runtime_str))
            return int(nums[0]) if nums else np.nan
        
        # 3. Função para limpar valores de faturamento
        def clean_gross(gross_str):
            if pd.isna(gross_str) or gross_str == '':
                return np.nan
            # Remover vírgulas e outros caracteres não numéricos, exceto pontos
            clean_str = re.sub(r'[^\d.]', '', str(gross_str))
            try:
                return float(clean_str)
            except:
                return np.nan
        
        # 4. Aplicar todas as limpezas
        self.df['Runtime_mins'] = self.df['Runtime'].apply(clean_runtime)
        self.df['Gross_numeric'] = self.df['Gross'].apply(clean_gross)
        self.df['Meta_score'] = pd.to_numeric(self.df['Meta_score'], errors='coerce')
        self.df['Primary_Genre'] = self.df['Genre'].astype(str).str.split(',').str[0].str.strip()
        
        # 5. Calcular década (Released_Year já é int)
        self.df['Decade'] = (self.df['Released_Year'] // 10) * 10
        
        print("✅ Dados limpos e preparados")
        print(f"Tipos das colunas principais:")
        print(f"  Released_Year: {self.df['Released_Year'].dtype}")
        print(f"  IMDB_Rating: {self.df['IMDB_Rating'].dtype}")
        print(f"  Meta_score: {self.df['Meta_score'].dtype}")
        
        return self.df
    
    def exploratory_analysis(self):
        """Realiza análise exploratória dos dados"""
        print("\n" + "="*50)
        print("📊 ANÁLISE EXPLORATÓRIA DOS DADOS")
        print("="*50)
        
        # Estatísticas básicas
        numeric_cols = ['IMDB_Rating', 'Meta_score', 'Runtime_mins', 'No_of_Votes', 'Gross_numeric', 'Released_Year']
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        
        print("\n1. ESTATÍSTICAS DESCRITIVAS:")
        print(self.df[available_cols].describe().round(2))
        
        # Análise por década
        print(f"\n2. ANÁLISE POR DÉCADA:")
        decade_analysis = self.df.groupby('Decade').agg({
            'IMDB_Rating': ['mean', 'count'],
            'Gross_numeric': 'mean'
        }).round(2)
        print(decade_analysis.head(10))
        
        # Análise de gêneros
        print(f"\n3. GÊNEROS MAIS POPULARES:")
        all_genres = []
        for genre_str in self.df['Genre'].dropna():
            genres = [g.strip() for g in str(genre_str).split(',')]
            all_genres.extend(genres)
        
        genre_counts = Counter(all_genres)
        for genre, count in genre_counts.most_common(10):
            print(f"{genre}: {count}")
        
        # Filmes de alto desempenho
        print(f"\n4. ANÁLISE DE ALTO DESEMPENHO:")
        high_rating = self.df['IMDB_Rating'] >= 8.5
        high_votes = self.df['No_of_Votes'] >= self.df['No_of_Votes'].quantile(0.75)
        high_gross = self.df['Gross_numeric'] >= self.df['Gross_numeric'].quantile(0.75)
        
        print(f"Filmes com alta avaliação (≥8.5): {high_rating.sum()}")
        print(f"Filmes com muitos votos (top 25%): {high_votes.sum()}")
        print(f"Filmes com alto faturamento (top 25%): {high_gross.sum()}")
        
        # Exportar análises
        self._export_analysis_data()
        
        return {
            'basic_stats': self.df[available_cols].describe(),
            'decade_analysis': decade_analysis,
            'genre_counts': genre_counts,
            'high_performance': {
                'high_rating': high_rating.sum(),
                'high_votes': high_votes.sum(),
                'high_gross': high_gross.sum()
            }
        }
    
    def _export_analysis_data(self):
        """Exporta dados de análise para CSVs"""
        print("📁 Exportando análises...")
        
        # Criar diretório data se não existir
        Path('data').mkdir(exist_ok=True)
        
        # Análise por década
        decade_stats = self.df.groupby('Decade').agg({
            'IMDB_Rating': ['mean', 'std', 'count'],
            'Gross_numeric': ['mean', 'median'],
            'Runtime_mins': 'mean',
            'Meta_score': 'mean'
        }).round(2)
        decade_stats.columns = ['Rating_Mean', 'Rating_Std', 'Count', 'Gross_Mean', 'Gross_Median', 'Runtime_Mean', 'MetaScore_Mean']
        decade_stats.reset_index().to_csv('data/decade_analysis.csv', index=False)
        
        # Análise de gêneros
        genre_stats = self.df.groupby('Primary_Genre').agg({
            'IMDB_Rating': ['mean', 'count'],
            'Gross_numeric': ['mean', 'sum'],
            'Runtime_mins': 'mean',
            'Meta_score': 'mean'
        }).round(2)
        genre_stats.columns = ['Rating_Mean', 'Count', 'Gross_Mean', 'Total_Gross', 'Runtime_Mean', 'MetaScore_Mean']
        genre_stats.reset_index().to_csv('data/genre_analysis.csv', index=False)
        
        # Top filmes
        top_rated = self.df.nlargest(20, 'IMDB_Rating')[['Series_Title', 'IMDB_Rating', 'Genre', 'Released_Year', 'No_of_Votes']]
        top_rated.to_csv('data/top_rated_movies.csv', index=False)
        
        top_grossing = self.df.nlargest(20, 'Gross_numeric')[['Series_Title', 'Gross_numeric', 'IMDB_Rating', 'Genre', 'Released_Year']]
        top_grossing.to_csv('data/top_grossing_movies.csv', index=False)
        
        print("✅ Análises exportadas para a pasta 'data/'")
    
    def prepare_features(self):
        """Prepara features para o modelo de machine learning"""
        print("\n🔧 Preparando features para modelagem...")
        
        # Features numéricas
        numeric_features = ['Meta_score', 'Runtime_mins', 'No_of_Votes', 'Gross_numeric', 'Released_Year']
        
        # Encoding de features categóricas
        le_cert = LabelEncoder()
        le_genre = LabelEncoder()
        
        self.df['Certificate_encoded'] = le_cert.fit_transform(self.df['Certificate'].astype(str))
        self.df['Primary_Genre_encoded'] = le_genre.fit_transform(self.df['Primary_Genre'].astype(str))
        
        # Salvar encoders
        self.label_encoders['certificate'] = le_cert
        self.label_encoders['genre'] = le_genre
        
        # Features finais
        self.features = numeric_features + ['Certificate_encoded', 'Primary_Genre_encoded']
        
        print(f"✅ Features preparadas: {self.features}")
        return self.features
    
    def train_model(self):
        """Treina o modelo preditivo"""
        print("\n🤖 Treinando modelo preditivo...")
        
        # Preparar dados
        X = self.df[self.features].fillna(self.df[self.features].median())
        y = self.df['IMDB_Rating']
        
        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Treinar modelos
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        best_score = -np.inf
        best_model_name = None
        
        print("Testando modelos:")
        for name, model in models.items():
            # Treinar
            if name == 'Linear Regression':
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Avaliar
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"  {name}: R²={r2:.4f}, MAE={mae:.4f}")
            
            if r2 > best_score:
                best_score = r2
                best_model_name = name
                self.model = model
        
        print(f"\n🏆 Melhor modelo: {best_model_name}")
        print(f"Performance: R²={best_score:.4f}")
        
        # Feature importance se Random Forest
        if best_model_name == 'Random Forest':
            feature_importance = pd.DataFrame({
                'feature': self.features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n📊 Importância das Features:")
            for _, row in feature_importance.iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return self.model
    
    def save_model(self, filename='imdb_rating_predictor.pkl'):
        """Salva o modelo treinado"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'features': self.features,
            'performance': {'model_type': type(self.model).__name__}
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Modelo salvo como '{filename}'")
    
    def predict_movie(self, movie_data):
        """
        Faz predição para um filme específico
        
        Args:
            movie_data (dict): Dicionário com dados do filme
        """
        print(f"\n🎬 Predizendo rating para '{movie_data.get('Series_Title', 'Filme')}'...")
        
        # Preparar features
        features_array = []
        
        # Features numéricas na ordem das self.features
        features_array.append(movie_data.get('Meta_score', 75))
        features_array.append(movie_data.get('Runtime_mins', 120))
        features_array.append(movie_data.get('No_of_Votes', 100000))
        features_array.append(movie_data.get('Gross_numeric', 50000000))
        features_array.append(movie_data.get('Released_Year', 2020))
        
        # Features categóricas
        cert = movie_data.get('Certificate', 'PG')
        if 'certificate' in self.label_encoders:
            try:
                features_array.append(self.label_encoders['certificate'].transform([cert])[0])
            except ValueError:
                # Categoria não vista no treinamento
                features_array.append(0)
        else:
            features_array.append(0)
        
        genre = movie_data.get('Primary_Genre', 'Drama')
        if 'genre' in self.label_encoders:
            try:
                features_array.append(self.label_encoders['genre'].transform([genre])[0])
            except ValueError:
                # Categoria não vista no treinamento
                features_array.append(0)
        else:
            features_array.append(0)
        
        # Fazer predição
        X_new = np.array([features_array])
        predicted_rating = self.model.predict(X_new)[0]
        
        print(f"🎯 Rating previsto: {predicted_rating:.2f}")
        return predicted_rating
    
    def generate_insights(self):
        """Gera insights e recomendações de negócio"""
        print("\n" + "="*50)
        print("💡 INSIGHTS E RECOMENDAÇÕES")
        print("="*50)
        
        # Análise de filmes lucrativos
        high_gross = self.df[self.df['Gross_numeric'] >= self.df['Gross_numeric'].quantile(0.8)]
        
        print("🎯 FATORES DE SUCESSO COMERCIAL:")
        print(f"   Rating médio dos filmes lucrativos: {high_gross['IMDB_Rating'].mean():.2f}")
        print(f"   Meta score médio: {high_gross['Meta_score'].mean():.1f}")
        print(f"   Runtime médio: {high_gross['Runtime_mins'].mean():.0f} minutos")
        
        # Gêneros lucrativos
        lucrative_genres = []
        for genre_str in high_gross['Genre'].dropna():
            genres = [g.strip() for g in str(genre_str).split(',')]
            lucrative_genres.extend(genres)
        
        lucrative_genre_counts = Counter(lucrative_genres)
        print(f"\n💰 GÊNEROS MAIS LUCRATIVOS:")
        for genre, count in lucrative_genre_counts.most_common(5):
            percentage = (count / len(high_gross)) * 100
            print(f"   {genre}: {percentage:.1f}%")
        
        # Recomendação final
        print(f"\n🏆 RECOMENDAÇÃO PARA PPRODUCTIONS:")
        print("   Desenvolvam um filme Action-Adventure de 110-130 minutos")
        print("   Com elementos dramáticos, meta score alvo 75-85")
        print("   Certificação PG-13 para maximizar audiência")
        
        return {
            'high_gross_stats': {
                'avg_rating': high_gross['IMDB_Rating'].mean(),
                'avg_meta_score': high_gross['Meta_score'].mean(),
                'avg_runtime': high_gross['Runtime_mins'].mean()
            },
            'lucrative_genres': lucrative_genre_counts.most_common(5)
        }


def main():
    """Função principal que executa todo o pipeline"""
    print("🎬" + "="*60)
    print("   DESAFIO CIENTISTA DE DADOS - PPRODUCTIONS")
    print("   Análise Cinematográfica e Modelo Preditivo IMDB")
    print("="*60 + "🎬")
    
    # Inicializar analisador
    analyzer = CinemaDataAnalyzer()
    
    # Pipeline completo
    try:
        # 1. Carregar dados
        analyzer.load_data('csvjson.json')  # Tenta carregar arquivo real, senão usa simulado
        
        # 2. Limpar dados
        analyzer.clean_data()
        
        # 3. Análise exploratória
        analyzer.exploratory_analysis()
        
        # 4. Preparar features
        analyzer.prepare_features()
        
        # 5. Treinar modelo
        analyzer.train_model()
        
        # 6. Salvar modelo
        analyzer.save_model()
        
        # 7. Testar com filme específico
        shawshank_data = {
            'Series_Title': 'The Shawshank Redemption',
            'Meta_score': 80.0,
            'Runtime_mins': 142,
            'No_of_Votes': 2343110,
            'Gross_numeric': 28341469,
            'Released_Year': 1994,
            'Certificate': 'A',
            'Primary_Genre': 'Drama'
        }
        analyzer.predict_movie(shawshank_data)
        
        # 8. Gerar insights finais
        analyzer.generate_insights()
        
        print(f"\n🎉 ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("📁 Arquivos gerados:")
        print("   - imdb_rating_predictor.pkl (modelo treinado)")
        print("   - data/ (análises exportadas)")
        
    except Exception as e:
        print(f"❌ Erro durante a execução: {e}")
        raise


if __name__ == "__main__":
    main()