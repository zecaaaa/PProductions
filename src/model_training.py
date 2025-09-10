# -*- coding: utf-8 -*-
"""
Model Training Script - Desafio Cientista de Dados PProductions

Este script é focado especificamente no treinamento do modelo preditivo de ratings IMDB.
Executa múltiplos algoritmos, otimização de hiperparâmetros e comparação de performance.

Autor: Zeca
Data: Setembro 2025
"""

import pandas as pd
import numpy as np
import pickle
import json
import re
from pathlib import Path
import warnings

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler

# Suprimir warnings
warnings.filterwarnings('ignore')


class IMDBRatingPredictor:
    """Classe especializada para treinamento de modelo de predição de ratings IMDB"""
    
    def __init__(self, random_state=42):
        """
        Inicializa o preditor de ratings IMDB
        
        Args:
            random_state (int): Seed para reprodutibilidade
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.features = []
        self.performance_metrics = {}
        
    def load_and_prepare_data(self, data_path=None):
        """
        Carrega e prepara os dados para treinamento
        
        Args:
            data_path (str): Caminho para arquivo de dados (opcional)
            
        Returns:
            tuple: (X, y, df) - Features, target, dataframe completo
        """
        print("📊 Carregando e preparando dados...")
        
        # Tentar carregar dados reais primeiro
        if data_path and Path(data_path).exists():
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                print(f"✅ Dados reais carregados: {df.shape}")
            except Exception as e:
                print(f"⚠️ Erro ao carregar {data_path}: {e}")
                print("🎲 Usando dados simulados...")
                df = self._generate_training_data()
        else:
            print("🎲 Gerando dados simulados para treinamento...")
            df = self._generate_training_data()
        
        # Limpar e preparar dados
        df_clean = self._clean_data(df)
        X, y = self._prepare_features(df_clean)
        
        return X, y, df_clean
    
    def _generate_training_data(self):
        """Gera dados de treinamento simulados baseados em padrões reais do IMDB"""
        np.random.seed(self.random_state)
        n_samples = 999
        
        # Gerar anos realistas
        years = np.random.choice(
            range(1920, 2024), 
            size=n_samples, 
            p=self._year_distribution()
        )
        
        # Ratings baseados em distribuição real do IMDB Top 1000
        ratings = np.random.beta(2, 0.8, n_samples) * 3.5 + 6.5  # Entre 6.5 e 10
        ratings = np.clip(ratings, 6.0, 9.5)
        
        # Meta scores correlacionados com ratings
        meta_scores = ratings * 10 + np.random.normal(0, 8, n_samples)
        meta_scores = np.clip(meta_scores, 20, 100)
        
        # Número de votos (distribuição log-normal)
        votes = np.random.lognormal(12.5, 1.2, n_samples).astype(int)
        votes = np.clip(votes, 5000, 2500000)
        
        # Runtime realista
        runtimes = np.random.normal(115, 22, n_samples).astype(int)
        runtimes = np.clip(runtimes, 80, 200)
        
        # Faturamento (correlacionado com rating e votos)
        gross_base = (ratings - 6) * 50000000 + (votes / 100)
        gross_multiplier = np.random.lognormal(0, 0.8, n_samples)
        gross = (gross_base * gross_multiplier).astype(int)
        gross = np.clip(gross, 100000, 1000000000)
        
        # Gêneros com distribuição realista
        genres = ['Drama', 'Action', 'Comedy', 'Thriller', 'Adventure', 
                 'Crime', 'Romance', 'Sci-Fi', 'Horror', 'Biography', 
                 'War', 'Mystery', 'Fantasy', 'Animation']
        genre_weights = [0.25, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01]
        
        # Certificações com distribuição realista
        certificates = ['G', 'PG', 'PG-13', 'R', 'NC-17', 'U', 'UA', 'A']
        cert_weights = [0.05, 0.20, 0.35, 0.25, 0.02, 0.05, 0.05, 0.03]
        
        # Criar DataFrame
        data = {
            'Rank': range(1, n_samples + 1),
            'Series_Title': [f'Movie_{i:03d}' for i in range(1, n_samples + 1)],
            'Released_Year': years,
            'Certificate': np.random.choice(certificates, n_samples, p=cert_weights),
            'Runtime': [f'{runtime} min' for runtime in runtimes],
            'Genre': np.random.choice(genres, n_samples, p=genre_weights),
            'IMDB_Rating': np.round(ratings, 1),
            'Meta_score': np.round(meta_scores, 0),
            'No_of_Votes': votes,
            'Gross': [f'{g:,}' for g in gross],
            'Overview': [f'A compelling {np.random.choice(["drama", "story", "tale", "adventure"])} about {np.random.choice(["love", "survival", "justice", "friendship", "betrayal", "redemption"])}.' for _ in range(n_samples)],
            'Director': [f'Director_{i:03d}' for i in range(1, n_samples + 1)],
            'Star1': [f'Actor_{i:03d}_A' for i in range(1, n_samples + 1)],
            'Star2': [f'Actor_{i:03d}_B' for i in range(1, n_samples + 1)],
            'Star3': [f'Actor_{i:03d}_C' for i in range(1, n_samples + 1)],
            'Star4': [f'Actor_{i:03d}_D' for i in range(1, n_samples + 1)]
        }
        
        df = pd.DataFrame(data)
        print(f"📈 Dataset simulado criado: {df.shape[0]} filmes")
        return df
    
    def _year_distribution(self):
        """Cria distribuição realista de anos (mais filmes recentes)"""
        years = list(range(1920, 2024))
        # Distribuição exponencial favorecendo anos mais recentes
        weights = np.exp(np.linspace(-3, 0, len(years)))
        weights = weights / weights.sum()
        return weights
    
    def _clean_data(self, df):
        """
        Executa limpeza robusta dos dados
        
        Args:
            df (pd.DataFrame): DataFrame bruto
            
        Returns:
            pd.DataFrame: DataFrame limpo
        """
        print("🧹 Executando limpeza robusta dos dados...")
        
        df_clean = df.copy()
        
        # 1. Released_Year - Conversão segura para inteiro
        df_clean['Released_Year'] = pd.to_numeric(df_clean['Released_Year'], errors='coerce')
        median_year = int(df_clean['Released_Year'].median()) if df_clean['Released_Year'].notna().any() else 2000
        df_clean['Released_Year'] = df_clean['Released_Year'].fillna(median_year).astype(int)
        
        # 2. Runtime - Extrair minutos
        def extract_runtime(runtime_str):
            if pd.isna(runtime_str):
                return np.nan
            # Procurar por números na string
            numbers = re.findall(r'\d+', str(runtime_str))
            if numbers:
                return int(numbers[0])
            return np.nan
        
        df_clean['Runtime_mins'] = df_clean['Runtime'].apply(extract_runtime)
        df_clean['Runtime_mins'] = df_clean['Runtime_mins'].fillna(df_clean['Runtime_mins'].median())
        
        # 3. Gross - Limpeza robusta de valores monetários
        def clean_gross(gross_str):
            if pd.isna(gross_str):
                return np.nan
            
            # Converter para string e remover espaços
            gross_clean = str(gross_str).strip()
            
            # Se vazio, retornar NaN
            if gross_clean == '' or gross_clean.lower() == 'nan':
                return np.nan
            
            # Remover vírgulas, cifrões e outros símbolos, manter apenas dígitos e pontos
            gross_clean = re.sub(r'[^\d.]', '', gross_clean)
            
            # Tentar converter para float
            try:
                value = float(gross_clean)
                # Validar range realista (entre 1000 e 3 bilhões)
                if 1000 <= value <= 3_000_000_000:
                    return value
                elif value > 3_000_000_000:
                    # Pode estar em formato diferente, tentar dividir
                    return value / 1000 if value / 1000 <= 3_000_000_000 else np.nan
                else:
                    return np.nan
            except (ValueError, TypeError):
                return np.nan
        
        df_clean['Gross_numeric'] = df_clean['Gross'].apply(clean_gross)
        gross_median = df_clean['Gross_numeric'].median()
        df_clean['Gross_numeric'] = df_clean['Gross_numeric'].fillna(gross_median)
        
        # 4. Meta_score - Conversão numérica
        df_clean['Meta_score'] = pd.to_numeric(df_clean['Meta_score'], errors='coerce')
        df_clean['Meta_score'] = df_clean['Meta_score'].fillna(df_clean['Meta_score'].median())
        df_clean['Meta_score'] = df_clean['Meta_score'].clip(0, 100)
        
        # 5. No_of_Votes - Conversão numérica
        df_clean['No_of_Votes'] = pd.to_numeric(df_clean['No_of_Votes'], errors='coerce')
        df_clean['No_of_Votes'] = df_clean['No_of_Votes'].fillna(df_clean['No_of_Votes'].median())
        df_clean['No_of_Votes'] = df_clean['No_of_Votes'].clip(100, 5_000_000)
        
        # 6. IMDB_Rating - Limpar e validar
        df_clean['IMDB_Rating'] = pd.to_numeric(df_clean['IMDB_Rating'], errors='coerce')
        df_clean['IMDB_Rating'] = df_clean['IMDB_Rating'].fillna(df_clean['IMDB_Rating'].median())
        df_clean['IMDB_Rating'] = df_clean['IMDB_Rating'].clip(1.0, 10.0)
        
        # 7. Certificate - Padronizar
        df_clean['Certificate'] = df_clean['Certificate'].fillna('PG').astype(str)
        
        # 8. Genre - Limpar e extrair gênero principal
        df_clean['Genre'] = df_clean['Genre'].fillna('Drama').astype(str)
        df_clean['Primary_Genre'] = df_clean['Genre'].str.split(',').str[0].str.strip()
        
        # 9. Criar features derivadas
        df_clean['Decade'] = (df_clean['Released_Year'] // 10) * 10
        df_clean['Runtime_Category'] = pd.cut(
            df_clean['Runtime_mins'], 
            bins=[0, 90, 120, 150, 300], 
            labels=['Short', 'Standard', 'Long', 'Epic']
        )
        df_clean['Votes_Category'] = pd.qcut(
            df_clean['No_of_Votes'], 
            q=4, 
            labels=['Low', 'Medium', 'High', 'Very_High']
        )
        
        print("✅ Dados limpos com sucesso")
        print(f"📊 Shape final: {df_clean.shape}")
        print(f"📈 Colunas numéricas: Released_Year({df_clean['Released_Year'].dtype}), "
              f"IMDB_Rating({df_clean['IMDB_Rating'].dtype}), "
              f"Meta_score({df_clean['Meta_score'].dtype})")
        
        return df_clean
    
    def _prepare_features(self, df):
        """
        Prepara features para treinamento do modelo
        
        Args:
            df (pd.DataFrame): DataFrame limpo
            
        Returns:
            tuple: (X, y) - Features e target
        """
        print("🔧 Preparando features para modelagem...")
        
        # Features numéricas base
        numeric_features = [
            'Meta_score', 'Runtime_mins', 'No_of_Votes', 
            'Gross_numeric', 'Released_Year'
        ]
        
        # Encoding de variáveis categóricas
        categorical_features = []
        
        # Certificate
        if 'Certificate' in df.columns:
            le_cert = LabelEncoder()
            df['Certificate_encoded'] = le_cert.fit_transform(df['Certificate'].astype(str))
            self.label_encoders['certificate'] = le_cert
            categorical_features.append('Certificate_encoded')
        
        # Primary_Genre
        if 'Primary_Genre' in df.columns:
            le_genre = LabelEncoder()
            df['Primary_Genre_encoded'] = le_genre.fit_transform(df['Primary_Genre'].astype(str))
            self.label_encoders['genre'] = le_genre
            categorical_features.append('Primary_Genre_encoded')
        
        # Decade
        if 'Decade' in df.columns:
            le_decade = LabelEncoder()
            df['Decade_encoded'] = le_decade.fit_transform(df['Decade'].astype(str))
            self.label_encoders['decade'] = le_decade
            categorical_features.append('Decade_encoded')
        
        # Runtime_Category
        if 'Runtime_Category' in df.columns:
            le_runtime_cat = LabelEncoder()
            df['Runtime_Category_encoded'] = le_runtime_cat.fit_transform(df['Runtime_Category'].astype(str))
            self.label_encoders['runtime_category'] = le_runtime_cat
            categorical_features.append('Runtime_Category_encoded')
        
        # Features finais
        self.features = [f for f in numeric_features if f in df.columns] + categorical_features
        
        # Preparar X e y
        X = df[self.features].copy()
        y = df['IMDB_Rating'].copy()
        
        # Verificar e tratar valores ausentes restantes
        if X.isnull().any().any():
            print("⚠️ Tratando valores ausentes restantes...")
            for col in X.columns:
                if X[col].isnull().any():
                    if X[col].dtype in ['int64', 'float64']:
                        X[col] = X[col].fillna(X[col].median())
                    else:
                        X[col] = X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 0)
        
        print(f"✅ Features preparadas: {len(self.features)} features")
        print(f"📋 Lista de features: {self.features}")
        print(f"📏 Shape final: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def initialize_models(self):
        """Inicializa conjunto diversificado de modelos para comparação"""
        print("🤖 Inicializando modelos para comparação...")
        
        self.models = {
            'Linear Regression': LinearRegression(),
            
            'Ridge Regression': Ridge(
                alpha=1.0, 
                random_state=self.random_state
            ),
            
            'Lasso Regression': Lasso(
                alpha=0.1, 
                random_state=self.random_state,
                max_iter=2000
            ),
            
            'Decision Tree': DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            ),
            
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                random_state=self.random_state
            ),
            
            'Support Vector Regressor': SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                epsilon=0.1
            )
        }
        
        print(f"✅ {len(self.models)} modelos inicializados")
    
    def train_and_evaluate_models(self, X, y):
        """
        Treina e avalia todos os modelos com validação cruzada
        
        Args:
            X: Features de treinamento
            y: Target variable
            
        Returns:
            dict: Métricas de performance de todos os modelos
        """
        print("🏋️‍♂️ Treinando e avaliando modelos...")
        
        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=None
        )
        
        # Scalers para testar
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        
        best_score = -np.inf
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n🔄 Treinando {model_name}...")
            
            try:
                # Determinar se precisa de scaling
                needs_scaling = model_name in [
                    'Linear Regression', 'Ridge Regression', 'Lasso Regression', 
                    'Support Vector Regressor'
                ]
                
                if needs_scaling:
                    # Testar diferentes scalers e escolher o melhor
                    best_scaler_score = -np.inf
                    best_scaler = None
                    
                    for scaler_name, scaler in scalers.items():
                        # Aplicar scaling
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Treinar modelo
                        model_copy = self._get_model_copy(model_name)
                        model_copy.fit(X_train_scaled, y_train)
                        
                        # Avaliar
                        score = model_copy.score(X_test_scaled, y_test)
                        
                        if score > best_scaler_score:
                            best_scaler_score = score
                            best_scaler = scaler
                    
                    # Usar melhor scaler
                    X_train_processed = best_scaler.fit_transform(X_train)
                    X_test_processed = best_scaler.transform(X_test)
                    self.scaler = best_scaler
                    
                else:
                    X_train_processed = X_train
                    X_test_processed = X_test
                
                # Treinar modelo final
                model.fit(X_train_processed, y_train)
                
                # Fazer predições
                y_pred_train = model.predict(X_train_processed)
                y_pred_test = model.predict(X_test_processed)
                
                # Calcular métricas
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train_processed, y_train, 
                    cv=5, scoring='r2', n_jobs=-1
                )
                
                # Salvar métricas
                metrics = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'overfitting': train_r2 - test_r2
                }
                
                self.performance_metrics[model_name] = metrics
                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'scaler': self.scaler if needs_scaling else None
                }
                
                # Log das métricas
                print(f"   📊 Métricas de {model_name}:")
                print(f"      R² Treino: {train_r2:.4f} | R² Teste: {test_r2:.4f}")
                print(f"      MAE Treino: {train_mae:.4f} | MAE Teste: {test_mae:.4f}")
                print(f"      RMSE Treino: {train_rmse:.4f} | RMSE Teste: {test_rmse:.4f}")
                print(f"      CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                print(f"      Overfitting: {train_r2 - test_r2:.4f}")
                
                # Atualizar melhor modelo
                if test_r2 > best_score:
                    best_score = test_r2
                    self.best_model = model
                    self.best_model_name = model_name
            
            except Exception as e:
                print(f"   ❌ Erro ao treinar {model_name}: {str(e)}")
                continue
        
        print(f"\n🏆 Melhor modelo: {self.best_model_name}")
        print(f"🎯 R² Score: {best_score:.4f}")
        
        return results
    
    def _get_model_copy(self, model_name):
        """Retorna uma cópia do modelo para testes"""
        model_configs = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso Regression': Lasso(alpha=0.1, random_state=self.random_state, max_iter=2000),
            'Support Vector Regressor': SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
        }
        return model_configs.get(model_name, LinearRegression())
    
    def optimize_hyperparameters(self, X, y):
        """
        Otimiza hiperparâmetros do melhor modelo usando Grid Search
        
        Args:
            X: Features
            y: Target
        """
        if not self.best_model_name:
            print("⚠️ Nenhum modelo foi treinado ainda")
            return
        
        print(f"\n⚙️ Otimizando hiperparâmetros para {self.best_model_name}...")
        
        # Definir grids de parâmetros para diferentes modelos
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 6, 9],
                'min_samples_split': [2, 5]
            },
            'Ridge Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Support Vector Regressor': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.01, 0.1],
                'epsilon': [0.01, 0.1, 0.2]
            }
        }
        
        if self.best_model_name not in param_grids:
            print(f"⚠️ Otimização não implementada para {self.best_model_name}")
            return
        
        try:
            # Preparar dados para otimização
            needs_scaling = self.best_model_name in [
                'Linear Regression', 'Ridge Regression', 'Lasso Regression', 
                'Support Vector Regressor'
            ]
            
            if needs_scaling:
                X_processed = self.scaler.fit_transform(X)
            else:
                X_processed = X
            
            # Criar modelo base
            base_model = self._get_model_copy(self.best_model_name)
            
            # Grid Search
            grid_search = GridSearchCV(
                base_model,
                param_grids[self.best_model_name],
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_processed, y)
            
            # Atualizar melhor modelo
            self.best_model = grid_search.best_estimator_
            
            print(f"✅ Otimização concluída!")
            print(f"🎯 Melhores parâmetros: {grid_search.best_params_}")
            print(f"📈 Melhor score CV: {grid_search.best_score_:.4f}")
            
            # Atualizar métricas
            self.performance_metrics[self.best_model_name]['optimized_cv_score'] = grid_search.best_score_
            self.performance_metrics[self.best_model_name]['best_params'] = grid_search.best_params_
            
        except Exception as e:
            print(f"❌ Erro durante otimização: {str(e)}")
    
    def get_feature_importance(self):
        """
        Obtém e exibe importância das features (quando disponível)
        
        Returns:
            pd.DataFrame: DataFrame com importância das features
        """
        if not self.best_model:
            print("⚠️ Nenhum modelo foi treinado ainda")
            return None
        
        print(f"\n📊 Analisando importância das features para {self.best_model_name}...")
        
        if hasattr(self.best_model, 'feature_importances_'):
            # Para modelos baseados em árvores
            importance_df = pd.DataFrame({
                'feature': self.features,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("🏆 Top 10 Features mais importantes:")
            for idx, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                print(f"   {idx:2d}. {row['feature']:<20}: {row['importance']:.4f}")
            
            return importance_df
            
        elif hasattr(self.best_model, 'coef_'):
            # Para modelos lineares
            importance_df = pd.DataFrame({
                'feature': self.features,
                'coefficient': self.best_model.coef_,
                'abs_coefficient': np.abs(self.best_model.coef_)
            }).sort_values('abs_coefficient', ascending=False)
            
            print("🏆 Top 10 Features por coeficiente (valor absoluto):")
            for idx, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                print(f"   {idx:2d}. {row['feature']:<20}: {row['coefficient']:8.4f} (|{row['abs_coefficient']:.4f}|)")
            
            return importance_df
        else:
            print("ℹ️ Importância de features não disponível para este modelo")
            return None
    
    def predict_single_movie(self, movie_data):
        """
        Prediz rating para um filme específico
        
        Args:
            movie_data (dict): Dados do filme
            
        Returns:
            float: Rating predito
        """
        if not self.best_model:
            raise ValueError("Nenhum modelo foi treinado ainda")
        
        print(f"\n🎬 Predizendo rating para '{movie_data.get('title', 'filme')}' usando {self.best_model_name}...")
        
        # Preparar features na ordem correta
        features_array = []
        
        for feature in self.features:
            if feature == 'Meta_score':
                features_array.append(movie_data.get('Meta_score', 70))
            elif feature == 'Runtime_mins':
                features_array.append(movie_data.get('Runtime_mins', 110))
            elif feature == 'No_of_Votes':
                features_array.append(movie_data.get('No_of_Votes', 50000))
            elif feature == 'Gross_numeric':
                features_array.append(movie_data.get('Gross_numeric', 50000000))
            elif feature == 'Released_Year':
                features_array.append(movie_data.get('Released_Year', 2023))
            elif feature == 'Certificate_encoded':
                cert = movie_data.get('Certificate', 'PG-13')
                try:
                    if 'certificate' in self.label_encoders:
                        features_array.append(self.label_encoders['certificate'].transform([cert])[0])
                    else:
                        features_array.append(0)
                except ValueError:
                    features_array.append(0)  # Categoria não vista no treino
            elif feature == 'Primary_Genre_encoded':
                genre = movie_data.get('Genre', 'Drama')
                try:
                    if 'genre' in self.label_encoders:
                        features_array.append(self.label_encoders['genre'].transform([genre])[0])
                    else:
                        features_array.append(0)
                except ValueError:
                    features_array.append(0)
            elif feature == 'Decade_encoded':
                decade = (movie_data.get('Released_Year', 2020) // 10) * 10
                try:
                    if 'decade' in self.label_encoders:
                        features_array.append(self.label_encoders['decade'].transform([str(decade)])[0])
                    else:
                        features_array.append(0)
                except ValueError:
                    features_array.append(0)
            elif feature == 'Runtime_Category_encoded':
                runtime = movie_data.get('Runtime_mins', 110)
                if runtime < 90:
                    cat = 'Short'
                elif runtime < 120:
                    cat = 'Standard'
                elif runtime < 150:
                    cat = 'Long'
                else:
                    cat = 'Epic'
                try:
                    if 'runtime_category' in self.label_encoders:
                        features_array.append(self.label_encoders['runtime_category'].transform([cat])[0])
                    else:
                        features_array.append(0)
                except ValueError:
                    features_array.append(0)
        
        # Converter para array numpy
        X_new = np.array([features_array])
        
        # Aplicar scaling se necessário
        if self.best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Support Vector Regressor']:
            X_new = self.scaler.transform(X_new)
        
        # Fazer predição
        prediction = self.best_model.predict(X_new)[0]
        
        print(f"🎯 Rating previsto: {prediction:.2f}")
        return prediction
    
    def save_model(self, filename='imdb_rating_predictor.pkl'):
        """
        Salva o modelo treinado e todos os componentes necessários
        
        Args:
            filename (str): Nome do arquivo para salvar
            
        Returns:
            str: Caminho do arquivo salvo
        """
        if not self.best_model:
            raise ValueError("Nenhum modelo foi treinado ainda")
        
        model_package = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'features': self.features,
            'performance_metrics': self.performance_metrics,
            'metadata': {
                'created_date': pd.Timestamp.now().isoformat(),
                'random_state': self.random_state,
                'n_features': len(self.features),
                'best_cv_score': self.performance_metrics.get(self.best_model_name, {}).get('cv_mean', 0),
                'feature_names': self.features
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"💾 Modelo salvo em: {filename}")
        print(f"📦 Componentes salvos:")
        print(f"   - Modelo: {self.best_model_name}")
        print(f"   - Scaler: {type(self.scaler).__name__}")
        print(f"   - Label Encoders: {len(self.label_encoders)}")
        print(f"   - Features: {len(self.features)}")
        print(f"   - Métricas: {len(self.performance_metrics)} modelos")
        
        return filename
    
    def generate_comprehensive_report(self):
        """Gera relatório completo e detalhado do treinamento"""
        print("\n" + "="*80)
        print("📋 RELATÓRIO COMPLETO DO TREINAMENTO")
        print("="*80)
        
        # 1. Resumo Executivo
        print(f"\n🎯 RESUMO EXECUTIVO:")
        print(f"   Melhor Modelo: {self.best_model_name}")
        if self.best_model_name in self.performance_metrics:
            best_metrics = self.performance_metrics[self.best_model_name]
            print(f"   R² Score: {best_metrics['test_r2']:.4f}")
            print(f"   MAE: {best_metrics['test_mae']:.4f}")
            print(f"   Cross-Validation: {best_metrics['cv_mean']:.4f} ± {best_metrics['cv_std']:.4f}")
            print(f"   Overfitting: {best_metrics['overfitting']:.4f}")
        
        # 2. Comparação de Modelos
        print(f"\n📊 COMPARAÇÃO DE TODOS OS MODELOS:")
        print(f"{'Modelo':<25} {'R² Test':<10} {'MAE Test':<10} {'CV Score':<12} {'Overfitting':<12}")
        print("-" * 75)
        
        for model_name, metrics in self.performance_metrics.items():
            print(f"{model_name:<25} {metrics['test_r2']:<10.4f} {metrics['test_mae']:<10.4f} "
                  f"{metrics['cv_mean']:<12.4f} {metrics['overfitting']:<12.4f}")
        
        # 3. Análise de Features
        print(f"\n🔧 CONFIGURAÇÃO DE FEATURES:")
        print(f"   Total de Features: {len(self.features)}")
        print(f"   Features Numéricas: {sum(1 for f in self.features if not f.endswith('_encoded'))}")
        print(f"   Features Categóricas: {sum(1 for f in self.features if f.endswith('_encoded'))}")
        print(f"   Label Encoders: {len(self.label_encoders)}")
        
        # 4. Recomendações
        print(f"\n💡 RECOMENDAÇÕES:")
        
        if self.best_model_name in self.performance_metrics:
            metrics = self.performance_metrics[self.best_model_name]
            
            if metrics['test_r2'] >= 0.8:
                print("   ✅ Modelo com excelente performance (R² ≥ 0.8)")
            elif metrics['test_r2'] >= 0.6:
                print("   ✅ Modelo com boa performance (R² ≥ 0.6)")
            else:
                print("   ⚠️ Modelo com performance moderada - considere:")
                print("      - Coletar mais dados de treinamento")
                print("      - Engenharia de features adicionais")
                print("      - Modelos mais complexos")
            
            if metrics['overfitting'] > 0.1:
                print("   ⚠️ Possível overfitting detectado - considere:")
                print("      - Regularização adicional")
                print("      - Mais dados de validação")
                print("      - Feature selection")
            else:
                print("   ✅ Overfitting sob controle")
            
            if metrics['test_mae'] < 0.5:
                print("   ✅ Erro absoluto baixo - predições confiáveis")
            elif metrics['test_mae'] > 1.0:
                print("   ⚠️ Erro absoluto alto - revisar qualidade dos dados")
        
        print(f"\n🎉 Relatório completo gerado!")


def main():
    """Função principal para execução completa do treinamento"""
    print("🤖" + "="*60)
    print("   TREINAMENTO AVANÇADO DO MODELO IMDB RATING PREDICTOR")
    print("   Desafio Cientista de Dados - PProductions")
    print("="*60 + "🤖")
    
    # Inicializar predictor
    predictor = IMDBRatingPredictor(random_state=42)
    
    try:
        print("\n🚀 Iniciando pipeline completo de treinamento...")
        
        # 1. Carregar e preparar dados
        X, y, df = predictor.load_and_prepare_data('csvjson.json')
        
        # 2. Inicializar modelos
        predictor.initialize_models()
        
        # 3. Treinar e avaliar todos os modelos
        model_results = predictor.train_and_evaluate_models(X, y)
        
        # 4. Otimizar hiperparâmetros do melhor modelo
        predictor.optimize_hyperparameters(X, y)
        
        # 5. Analisar importância das features
        feature_importance = predictor.get_feature_importance()
        
        # 6. Teste com exemplo real (The Shawshank Redemption)
        test_movie = {
            'title': 'The Shawshank Redemption',
            'Meta_score': 80,
            'Runtime_mins': 142,
            'No_of_Votes': 2343110,
            'Gross_numeric': 28341469,
            'Released_Year': 1994,
            'Certificate': 'R',
            'Genre': 'Drama'
        }
        
        predicted_rating = predictor.predict_single_movie(test_movie)
        
        # 7. Salvar modelo
        model_file = predictor.save_model('imdb_rating_predictor.pkl')
        
        # 8. Gerar relatório completo
        predictor.generate_comprehensive_report()
        
        print(f"\n🎉 TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print(f"📁 Arquivo do modelo: {model_file}")
        print(f"🎯 Melhor modelo: {predictor.best_model_name}")
        print(f"📈 Performance: R²={predictor.performance_metrics[predictor.best_model_name]['test_r2']:.4f}")
        print(f"🎬 Teste Shawshank: Rating previsto = {predicted_rating:.2f}")
        
    except Exception as e:
        print(f"\n❌ Erro durante o treinamento: {e}")
        import traceback
        print(f"📝 Detalhes do erro:")
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()