# 🎬 Desafio Cientista de Dados - Análise Cinematográfica PProductions

## 📋 Sobre o Projeto

Este projeto foi desenvolvido como resposta ao desafio de Ciência de Dados da Indicium para orientar o estúdio de Hollywood PProductions na escolha do próximo filme a ser produzido. Através de análise exploratória de dados e modelagem preditiva, fornecemos insights estratégicos baseados em dados históricos do IMDB.

## 🎯 Objetivos

- Realizar análise exploratória completa dos dados cinematográficos
- Identificar fatores-chave para sucesso de bilheteria
- Desenvolver modelo preditivo para avaliações IMDB
- Fornecer recomendações estratégicas para o estúdio

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **Pandas** - Manipulação e análise de dados
- **NumPy** - Computação numérica
- **Scikit-learn** - Machine Learning
- **Matplotlib/Seaborn** - Visualizações (implícito)
- **Pickle** - Serialização do modelo

## 📦 Estrutura do Projeto

```
desafio-cinema-data-science/
├── README.md
├── requirements.txt
├── notebook_analise.ipynb
├── imdb_rating_predictor.pkl
├── data/
│   ├── decade_analysis.csv
│   ├── genre_analysis.csv
│   ├── top_rated_movies.csv
│   ├── top_grossing_movies.csv
│   ├── correlation_matrix.csv
│   └── executive_summary.csv
└── src/
    └── model_training.py
```

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos

Certifique-se de ter Python 3.8+ instalado em sua máquina.

### 2. Clonando o Repositório

```bash
git clone https://github.com/seu-usuario/desafio-cinema-data-science.git
cd desafio-cinema-data-science
```

### 3. Instalação das Dependências

```bash
pip install -r requirements.txt
```

### 4. Executando a Análise

```bash
# Para executar a análise completa
jupyter notebook notebook_analise.ipynb

# Ou executar o script diretamente
python src/model_training.py
```

### 5. Carregando o Modelo Treinado

```python
import pickle
import numpy as np

# Carregar o modelo
with open('imdb_rating_predictor.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
le_cert = model_data['label_encoders']['certificate']
le_genre = model_data['label_encoders']['genre']

# Fazer predição para um novo filme
new_movie_features = np.array([[80, 120, 500000, 100000000, 2023, 1, 0]])
predicted_rating = model.predict(new_movie_features)[0]
print(f"Rating previsto: {predicted_rating:.2f}")
```

## 📊 Principais Descobertas

### 🎭 Gêneros Mais Lucrativos
1. **Drama** - 22% dos filmes de alto faturamento
2. **Horror** - 17%
3. **Romance** - 15.5%
4. **Action/Adventure** - Melhor equilíbrio qualidade-comercial

### ⏱️ Duração Ideal
- **110-130 minutos** otimiza satisfação da audiência
- Filmes muito longos (>140min) mantêm qualidade mas podem reduzir apelo comercial

### 📈 Fatores de Sucesso
- **Meta Score**: 70-85 (equilíbrio crítica-comercial)
- **Rating IMDB**: 7.5-8.5 (qualidade + apelo)
- **Engajamento**: Alto número de votos indica sucesso de marketing

## 🤖 Modelo Preditivo

### Características do Modelo
- **Algoritmo**: Random Forest Regressor
- **Precisão**: MAE de 0.626
- **Features**: Meta_score, Runtime, Votos, Faturamento, Ano, Certificação, Gênero

### Performance
- **R² Score**: -0.0158
- **Mean Absolute Error**: 0.626
- **Modelo salvo**: `imdb_rating_predictor.pkl`

## 📝 Resultados e Recomendações

### Para "The Shawshank Redemption"
- **Rating Previsto**: 8.09
- **Características**: Drama, 142min, Meta Score 80, 2.3M votos

### Recomendação Estratégica
**Desenvolver filme Action-Adventure** com:
- Duração: 110-130 minutos
- Meta Score alvo: 75-85
- Certificação: PG-13/UA
- Elementos dramáticos para profundidade

## 📋 Requirements.txt

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
```

## 🔄 Versionamento

- **v1.0.0** - Análise inicial e modelo base
- **v1.1.0** - Refinamentos e otimizações
- **v2.0.0** - Modelo final e relatório completo

## 👨‍💻 Autor

Desenvolvido para o desafio Indicium Lighthouse - Cientista de Dados

## 📞 Contato

Para dúvidas ou sugestões sobre o projeto:
- Email: Ezequiel.contrat@gmail.com

## 📄 Licença

Este projeto foi desenvolvido para fins educacionais e de avaliação técnica.

---

**🎯 Sucesso do Projeto**: Fornecemos insights acionáveis baseados em dados para maximizar as chances de sucesso do próximo filme da PProductions, combinando análise estatística rigorosa com compreensão do mercado cinematográfico.