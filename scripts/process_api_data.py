import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os

# --- Configuration ---
API_KEY = '0681d32725f94e8b9cd95717252905'  
LOCALIZACAO_API = '41.8345,-7.7889'  
DAYS_TO_FETCH_API = 44

# Define file paths
BASE_DATA_PATH = Path("../data") 
RAW_DATA_PATH = BASE_DATA_PATH / "raw"
PROCESSED_DATA_PATH = BASE_DATA_PATH / "processed"
HISTORICAL_PROCESSED_FILE = PROCESSED_DATA_PATH / "agg_data_ml.csv"
PRODUCAO_RAW_FILE = RAW_DATA_PATH / "ReparticaoProducao.csv"

# --- Helper Functions ---

def fetch_weather_api_data(api_key: str, location: str, num_days: int) -> pd.DataFrame:
    """
    Fetches historical weather data for the last num_days from WeatherAPI.
    """
    print(f"Fetching weather data for the last {num_days} days...")
    today = datetime.today()
    dates_to_fetch = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, num_days + 1)]
    
    records = []
    for date_str in dates_to_fetch:
        url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={location}&dt={date_str}"
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()  
            data = res.json()
            day_data = data['forecast']['forecastday'][0]['day']
            wind_dir_noon = data['forecast']['forecastday'][0]['hour'][12]['wind_degree']
            
            records.append({
                'Data': pd.to_datetime(date_str),
                'Temperatura_Media': day_data['avgtemp_c'],
                'Intensidade_Media': day_data['maxwind_kph'] / 3.6, # Convert km/h to m/s
                'Direcao_Media': wind_dir_noon
            })
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {date_str}: {e}")
        except KeyError as e:
            print(f"Error parsing data for {date_str} (KeyError: {e}): {data}")

    if not records:
        print("No data fetched from API. Exiting.")
        return pd.DataFrame()
        
    df_api = pd.DataFrame(records)
    df_api = df_api.sort_values('Data').reset_index(drop=True)
    print(f"Successfully fetched {len(df_api)} records from API.")
    return df_api

def load_and_process_production_data(filepath: Path) -> pd.DataFrame:
    """
    Loads and processes the raw production data to get daily sums.
    Mirrors the logic from DataPreparation.ipynb.
    """
    print(f"Loading production data from: {filepath}")
    df_producao_raw = pd.read_csv(filepath, na_values=-990, sep=';', skiprows=2)
    df_producao_raw.columns = df_producao_raw.columns.str.strip()
    df_producao = df_producao_raw[['Data e Hora', 'Eólica']].copy()
    df_producao['Data e Hora'] = pd.to_datetime(df_producao['Data e Hora'])
    df_producao.set_index('Data e Hora', inplace=True)
    
    df_producao_diaria = df_producao.resample('D').sum().reset_index()
    df_producao_diaria.rename(columns={'Data e Hora': 'Data', 'Eólica_Total_Dia': 'Eólica'}, inplace=True)
    if 'Eólica_Total_Dia' in df_producao_diaria.columns:
         df_producao_diaria.rename(columns={'Eólica_Total_Dia': 'Eólica'}, inplace=True)
    else: # Ensure the column is named 'Eólica'
         df_producao_diaria.rename(columns={'Eólica': 'Eólica'}, inplace=True)


    print(f"Processed production data. Shape: {df_producao_diaria.shape}")
    return df_producao_diaria[['Data', 'Eólica']]


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature engineering steps from DataPreparation.ipynb (section 5).
    Assumes df has 'Data', 'Eólica', 'Intensidade_Media', 'Temperatura_Media', 'Direcao_Media'.
    """
    print("Applying feature engineering...")
    df_eng = df.copy()
    df_eng['Data'] = pd.to_datetime(df_eng['Data'])
    df_eng = df_eng.sort_values('Data').reset_index(drop=True)

    df_eng['mes'] = df_eng['Data'].dt.month
    df_eng['dia_da_semana'] = df_eng['Data'].dt.dayofweek
    df_eng['dia_do_ano'] = df_eng['Data'].dt.dayofyear
    df_eng['semana_do_ano'] = df_eng['Data'].dt.isocalendar().week.astype(int)
    df_eng['trimestre'] = df_eng['Data'].dt.quarter
    df_eng['eh_fim_de_semana'] = df_eng['dia_da_semana'].isin([5, 6]).astype(int)

    df_eng['vento_sin'] = np.sin(np.radians(df_eng['Direcao_Media']))
    df_eng['vento_cos'] = np.cos(np.radians(df_eng['Direcao_Media']))
    df_eng['dia_semana_sin'] = np.sin(2 * np.pi * df_eng['dia_da_semana'] / 7)
    df_eng['dia_semana_cos'] = np.cos(2 * np.pi * df_eng['dia_da_semana'] / 7)
    df_eng['mes_sin'] = np.sin(2 * np.pi * df_eng['mes'] / 12)
    df_eng['mes_cos'] = np.cos(2 * np.pi * df_eng['mes'] / 12)
    df_eng['dia_ano_sin'] = np.sin(2 * np.pi * df_eng['dia_do_ano'] / 366) # Use 366 for leap year safety
    df_eng['dia_ano_cos'] = np.cos(2 * np.pi * df_eng['dia_do_ano'] / 366)

    lags_eolica = [1, 2, 3, 7, 14]
    lags_meteo = [1, 2, 3, 7]

    for lag in lags_eolica:
        df_eng[f'Eólica_lag{lag}'] = df_eng['Eólica'].shift(lag)

    for lag in lags_meteo:
        df_eng[f'Intensidade_Media_lag{lag}'] = df_eng['Intensidade_Media'].shift(lag)
        df_eng[f'Temperatura_Media_lag{lag}'] = df_eng['Temperatura_Media'].shift(lag)
        df_eng[f'vento_sin_lag{lag}'] = df_eng['vento_sin'].shift(lag)
        df_eng[f'vento_cos_lag{lag}'] = df_eng['vento_cos'].shift(lag)

    window_sizes = [3, 7, 14]
    for window in window_sizes:
        # Eólica
        df_eng[f'Eolica_roll_mean_{window}'] = df_eng['Eólica'].shift(1).rolling(window=window, min_periods=1).mean()
        df_eng[f'Eolica_roll_std_{window}'] = df_eng['Eólica'].shift(1).rolling(window=window, min_periods=1).std()
        # Intensidade Média
        df_eng[f'Intensidade_Media_roll_mean_{window}'] = df_eng['Intensidade_Media'].shift(1).rolling(window=window, min_periods=1).mean()
        df_eng[f'Intensidade_Media_roll_std_{window}'] = df_eng['Intensidade_Media'].shift(1).rolling(window=window, min_periods=1).std()
        # Temperatura Média
        df_eng[f'Temperatura_Media_roll_mean_{window}'] = df_eng['Temperatura_Media'].shift(1).rolling(window=window, min_periods=1).mean()
        df_eng[f'Temperatura_Media_roll_std_{window}'] = df_eng['Temperatura_Media'].shift(1).rolling(window=window, min_periods=1).std()
    
    print("Feature engineering applied.")
    return df_eng

def handle_final_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles NaNs created by lag/rolling features using backfill.
    Mirrors logic from DataPreparation.ipynb (section 6).
    """
    print("Handling final NaNs...")
    df_filled = df.copy()
    cols_com_lags_roll = [col for col in df_filled.columns if '_lag' in col or '_roll_' in col]
    

    for col in cols_com_lags_roll:
        df_filled[col] = df_filled[col].bfill()

    nans_finais = df_filled.isnull().sum()
    if nans_finais.sum() > 0:
        print("WARNING: NaNs remaining after bfill:")
        print(nans_finais[nans_finais > 0])
        print("Attempting to fill remaining NaNs with 0 (this might not be ideal).")
        df_filled.fillna(0, inplace=True) 
    else:
        print("No NaNs remaining after bfill.")
    return df_filled

def main():
    df_historical = None
    if HISTORICAL_PROCESSED_FILE.exists():
        print(f"Loading historical processed data (for column reference) from: {HISTORICAL_PROCESSED_FILE}")
        df_historical = pd.read_csv(HISTORICAL_PROCESSED_FILE, parse_dates=['Data'])
    else:
        print(f"WARNING: Historical processed file not found: {HISTORICAL_PROCESSED_FILE}")
        print("Continuing without it, relying solely on API data for features.")

    # 2. Fetch new weather data from API
    #    Certifique-se que DAYS_TO_FETCH_API é grande o suficiente para cobrir
    #    o período de interesse + o máximo de dias de lag/rolling.
    #    Ex: Se quer prever Maio, e dados de Abril são input, e max_lag=14:
    #    Fetch desde meados de Março até o final de Abril.
    #    Ex: DAYS_TO_FETCH_API = 45 (para cobrir ~30 dias de Abril + 15 dias de contexto de Março)
    #        e ajuste as datas de início/fim da API implicitamente pelo loop em fetch_weather_api_data
    
    num_context_days_for_api_lags = 15 # Dias extras para buscar ANTES do seu período de interesse
                                       # Deve ser >= max(lags_eolica, lags_meteo, window_sizes)
    # Suponha que seu período de interesse para features é Abril.
    # Você precisa buscar dados da API desde meados de Março.
    # A função fetch_weather_api_data já busca "para trás" a partir de hoje.
    # Se você quer dados de Abril e Maio e hoje é, digamos, fim de Maio,
    # DAYS_TO_FETCH_API deve cobrir todo esse período + contexto.
    
    # Exemplo: Se hoje é 29 de Maio e você quer features para Abril e Maio.
    # E seu maior lag/janela é 14 dias.
    # Você precisa de dados desde ~15 de Março para popular lags para 1º de Abril.
    # Então DAYS_TO_FETCH_API deveria ser ~ (31mar+30abr+29mai) = 90 dias.
    # Ajuste DAYS_TO_FETCH_API na configuração do script.

    df_api_meteo = fetch_weather_api_data(API_KEY, LOCALIZACAO_API, DAYS_TO_FETCH_API)
    if df_api_meteo.empty:
        return

    # 3. Load recent production data
    if not PRODUCAO_RAW_FILE.exists():
        print(f"ERROR: Raw production file not found: {PRODUCAO_RAW_FILE}")
        return
    df_producao_recent = load_and_process_production_data(PRODUCAO_RAW_FILE)

    # 4. Merge API weather with recent production data
    print("Merging API weather data with recent production data...")
    df_new_data_raw = pd.merge(df_api_meteo, df_producao_recent, on='Data', how='inner')
    if df_new_data_raw.empty:
        print("No matching dates found between API weather data and production data. Cannot proceed.")
        return
    df_new_data_raw = df_new_data_raw.sort_values('Data').reset_index(drop=True)
    print(f"Merged new data. Shape: {df_new_data_raw.shape}")

    # 5. O DataFrame para feature engineering é AGORA APENAS os dados da API + produção recente
    df_for_feature_eng = df_new_data_raw.copy()
    
    # Se você quiser focar em um subconjunto específico (ex: só Abril para input)
    # mas buscou mais dados para contexto de lag, você pode filtrar aqui DEPOIS da feature engineering.
    # Por agora, vamos processar tudo que foi baixado.

    print(f"Data for feature engineering (from API/Recent Production). Shape: {df_for_feature_eng.shape}")
    
    # 6. Apply feature engineering
    df_features_applied = apply_feature_engineering(df_for_feature_eng)
    
    # 7. Handle NaNs from lags/rolling windows
    #    O bfill aqui vai preencher os NaNs no início do período da API
    #    usando os primeiros valores calculáveis dentro do período da API.
    df_processed_combined = handle_final_nans(df_features_applied)

    # 8. Definir o período de interesse para output
    #    Ex: Se você buscou 60 dias, mas só quer os últimos 30 com features completas.
    #    O handle_final_nans já fez bfill, então os primeiros dias terão features (embora preenchidas).
    #    Se você quiser remover os primeiros dias onde o lag/rolling não pôde ser totalmente calculado
    #    a partir de dados "reais" passados (e foram preenchidos por bfill), você pode dropar.
    #    Por exemplo, se o maior lag/janela é 14, os primeiros 13-14 dias de
    #    df_processed_combined terão algumas features preenchidas por bfill.
    
    # Suponha que você quer as features para um período específico (ex: Abril em diante)
    # E que DAYS_TO_FETCH_API foi configurado para ter dados suficientes antes.
    # start_date_of_interest = pd.to_datetime("2025-04-01")
    # df_final_new_data = df_processed_combined[df_processed_combined['Data'] >= start_date_of_interest].copy()
    
    # Ou, simplesmente pegar tudo que foi processado e o usuário decide depois
    df_final_new_data = df_processed_combined.copy()
    
    # Remover linhas onde 'Eólica' (target) é NaN, se houver
    # Isso pode acontecer se a produção para os dias mais recentes da API ainda não estiver disponível.
    df_final_new_data.dropna(subset=['Eólica'], inplace=True)


    print(f"\n--- Final Processed New Data (first 5 rows) ---")
    print(df_final_new_data.head())
    
    # (Restante do código de print e save...)

    # Para a consistência de colunas, se df_historical foi carregado:
    if df_historical is not None:
        expected_cols = df_historical.columns.tolist()
        # Adicionar colunas que podem ter sido criadas e não estão no histórico antigo
        # (ex: se você adicionou novas features no apply_feature_engineering)
        current_cols = df_final_new_data.columns.tolist()
        for col in current_cols:
            if col not in expected_cols:
                expected_cols.append(col) # Mantém novas features

        df_final_new_data = df_final_new_data.reindex(columns=expected_cols)
    else: # Se não há histórico, a ordem atual é a que fica
        print("No historical file for column reference, using current column order.")


    if df_final_new_data.isnull().sum().sum() > 0:
        print("\nWARNING: Some NaNs detected in the final new data after column reordering:")
        print(df_final_new_data.isnull().sum()[df_final_new_data.isnull().sum() > 0])

    output_filename = PROCESSED_DATA_PATH / f"api_data_featured_{datetime.now().strftime('%Y%m%d')}.csv"
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    df_final_new_data.to_csv(output_filename, index=False)
    print(f"\nSuccessfully processed and saved new data to: {output_filename}")

if __name__ == "__main__":
    # Create dummy data/processed folders and files if they don't exist for testing
    # In a real scenario, DataPreparation.ipynb would create agg_data_ml.csv
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    # Create a dummy ReparticaoProducao.csv if it doesn't exist
    if not PRODUCAO_RAW_FILE.exists():
        print(f"Dummy {PRODUCAO_RAW_FILE} not found. Creating a minimal one for script to run.")
        dummy_prod_data = {
            "Data e Hora": pd.to_datetime([datetime.now() - timedelta(days=i) for i in range(30)]).strftime('%Y-%m-%d %H:%M:%S'),
            "Eólica": np.random.randint(10000, 200000, 30)
        }
        # Add dummy header rows to match original skip
        with open(PRODUCAO_RAW_FILE, 'w') as f:
            f.write("Dummy Header 1\n")
            f.write("Dummy Header 2\n")
            pd.DataFrame(dummy_prod_data).to_csv(f, sep=';', index=False)

    # Create a dummy agg_data_ml.csv if it doesn't exist
    if not HISTORICAL_PROCESSED_FILE.exists():
        print(f"Dummy {HISTORICAL_PROCESSED_FILE} not found. Creating a minimal one for script to run.")
        # Create a minimal DataFrame with expected columns
        # This is complex due to many engineered features. For a true test, run DataPreparation.ipynb.
        # For this dummy, we'll just make it have the base columns.
        # The script will still try to make features, but lags will be mostly NaN then bfilled.
        num_dummy_historical_rows = 60
        dummy_dates = [datetime.now() - timedelta(days=i) for i in range(DAYS_TO_FETCH_API + 1, DAYS_TO_FETCH_API + 1 + num_dummy_historical_rows)]
        dummy_hist_data = {
            'Data': pd.to_datetime(dummy_dates),
            'Eólica': np.random.randint(50000, 150000, num_dummy_historical_rows),
            'Intensidade_Media': np.random.rand(num_dummy_historical_rows) * 5 + 1,
            'Temperatura_Media': np.random.rand(num_dummy_historical_rows) * 10 + 5,
            'Direcao_Media': np.random.randint(0, 360, num_dummy_historical_rows)
        }
        df_dummy_hist = pd.DataFrame(dummy_hist_data)
        # Apply feature engineering to this dummy historical data to get all columns
        df_dummy_hist_featured = apply_feature_engineering(df_dummy_hist)
        df_dummy_hist_featured = handle_final_nans(df_dummy_hist_featured)
        df_dummy_hist_featured.to_csv(HISTORICAL_PROCESSED_FILE, index=False)
        print(f"Created dummy {HISTORICAL_PROCESSED_FILE} with shape {df_dummy_hist_featured.shape} and columns: {df_dummy_hist_featured.columns.tolist()}")


    main()