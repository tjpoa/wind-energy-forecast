import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

# === CONFIGURAÇÕES ===
ENTRADA_CSV = "../data/processed/dados_combinados.csv"
MODELO_PATH = "../models/modelo_eolica_extratrees.pkl"

# === 1. LER DADOS ===
dados = pd.read_csv(ENTRADA_CSV)
dados['Data'] = pd.to_datetime(dados['Data'])
dados['mes'] = dados['Data'].dt.month
dados['dia_da_semana'] = dados['Data'].dt.dayofweek
dados['vento_sin'] = np.sin(np.radians(dados['Direcao_Media']))
dados['vento_cos'] = np.cos(np.radians(dados['Direcao_Media']))
dados['Intensidade_Media_lag1'] = dados['Intensidade_Media'].shift(1)
dados['Temperatura_Media_lag1'] = dados['Temperatura_Media'].shift(1)
dados['Eólica_lag1'] = dados['Eólica'].shift(1)
dados = dados.dropna()

X = dados.drop(columns=['Data', 'Eólica', 'Direcao_Media'])
y = dados['Eólica']

# === 2. TREINO ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = ExtraTreesRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# === 3. GUARDAR MODELO ===
joblib.dump(modelo, MODELO_PATH)
print(f"✅ Modelo guardado com sucesso em {MODELO_PATH}")
