
import pandas as pd
import numpy as np
import joblib

# === CONFIGURAÇÕES ===
MODELO_PATH = "../models/modelo_eolica_extratrees.pkl"
ENTRADA_CSV = "../data/processed/dados_combinados.csv"
SAIDA_CSV = "../tests/previsoes_eolica_extratrees.csv"

# === 1. CARREGAR MODELO ===
modelo = joblib.load(MODELO_PATH)

# === 2. LER DADOS CONSOLIDADOS ===
dados = pd.read_csv(ENTRADA_CSV)
dados['Data'] = pd.to_datetime(dados['Data'])
dados['mes'] = dados['Data'].dt.month
dados['dia_da_semana'] = dados['Data'].dt.dayofweek

# === 3. FEATURE ENGINEERING ===
dados['vento_sin'] = np.sin(np.radians(dados['Direcao_Media']))
dados['vento_cos'] = np.cos(np.radians(dados['Direcao_Media']))
dados['Intensidade_Media_lag1'] = dados['Intensidade_Media'].shift(1)
dados['Temperatura_Media_lag1'] = dados['Temperatura_Media'].shift(1)
dados['Eólica_lag1'] = dados['Eólica'].shift(1)

# === 4. LIMPAR DADOS ===
dados = dados.dropna()
X = dados.drop(columns=['Data', 'Eólica', 'Direcao_Media'])

# === 5. PREVER ===
dados['Eolica_Prevista'] = modelo.predict(X)

# === 6. GUARDAR RESULTADOS ===
saida = dados[['Data', 'Eolica_Prevista']]
saida.to_csv(SAIDA_CSV, index=False)
print(f"✅ Previsões salvas em {SAIDA_CSV}")
