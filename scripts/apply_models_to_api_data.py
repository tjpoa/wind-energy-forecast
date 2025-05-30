import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path
from datetime import datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import load_model 


BASE_DATA_PATH = Path("../data")
PROCESSED_DATA_PATH = BASE_DATA_PATH / "processed"
MODELS_PATH = Path("../models")


api_data_files = sorted(PROCESSED_DATA_PATH.glob("api_data_featured_*.csv"))
if not api_data_files:
    print("Nenhum arquivo 'api_data_featured_*.csv' encontrado em ../data/processed/")
    print("Por favor, execute o script 'process_api_data.py' primeiro.")
    exit()
LATEST_API_DATA_FILE = api_data_files[-1]
print(f"Usando o arquivo de dados da API mais recente: {LATEST_API_DATA_FILE}")


BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK = "ANN_Tuned" # Ex: 'ANN_Tuned', 'GradientBoosting_Tuned', etc.
BEST_MODEL_LOG_NAME_FROM_NOTEBOOK = "ANN_Tuned" # Ex: 'XGBoost_Tuned', 'ANN_Tuned', etc.


# --- Funções Auxiliares ---

def load_new_data(filepath: Path) -> pd.DataFrame:
    """Carrega os novos dados processados pela API."""
    print(f"Carregando novos dados de: {filepath}")
    df = pd.read_csv(filepath)
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values('Data').reset_index(drop=True)
    df.dropna(subset=['Eólica'], inplace=True)
    return df

def load_trained_model_and_scalers(model_name: str, target_type: str, models_dir: Path):
    """
    Carrega um modelo treinado e seus scalers associados, se aplicável (para ANNs).
    target_type deve ser 'original' ou 'log'.
    """
    print(f"Carregando melhor modelo para alvo {target_type}: {model_name}")
    model_instance = None
    scaler_x_instance = None
    scaler_y_instance = None

    if 'ANN' in model_name:
        model_path = models_dir / f"best_model_{target_type}_target_{model_name}.keras"
        scaler_x_path = models_dir / f"scaler_X_{target_type}_ann.joblib" 
        scaler_y_path = models_dir / f"scaler_y_{target_type}_ann.joblib" 

        if model_path.exists():
            model_instance = load_model(model_path)
        else:
            print(f"AVISO: Arquivo do modelo ANN não encontrado: {model_path}")

        if scaler_x_path.exists():
            scaler_x_instance = joblib.load(scaler_x_path)
        else:
            print(f"AVISO: Arquivo do scaler X não encontrado: {scaler_x_path} (Necessário para ANN)")

        if scaler_y_path.exists():
            scaler_y_instance = joblib.load(scaler_y_path)
        else:
            print(f"AVISO: Arquivo do scaler Y não encontrado: {scaler_y_path} (Necessário para ANN)")
    else: # Modelos Sklearn
        model_path = models_dir / f"best_model_{target_type}_target_{model_name}.joblib"
        if model_path.exists():
            model_instance = joblib.load(model_path)
        else:
            print(f"AVISO: Arquivo do modelo não encontrado: {model_path}")

    return model_instance, scaler_x_instance, scaler_y_instance

def prepare_data_for_prediction(df_new: pd.DataFrame, scaler_x=None):
    """Prepara as features X dos novos dados, aplicando scaling se necessário."""
    X_new = df_new.drop(columns=['Data', 'Eólica'])

    # Vamos carregar o agg_data_ml.csv para obter a ordem correta das colunas das features
    df_historical_for_cols = pd.read_csv(PROCESSED_DATA_PATH / "agg_data_ml.csv")
    feature_columns_ordered = df_historical_for_cols.drop(columns=['Data', 'Eólica']).columns.tolist()

    for col in feature_columns_ordered:
        if col not in X_new.columns:
            X_new[col] = 0 # ou np.nan, dependendo da estratégia de preenchimento do modelo
    X_new = X_new[feature_columns_ordered]


    if scaler_x:
        print("Aplicando X scaling...")
        X_new_scaled = scaler_x.transform(X_new)
        return X_new_scaled, X_new.columns 
    return X_new, X_new.columns

def make_predictions(model, X_data, scaler_y=None, is_log_target=False):
    """Faz previsões e reverte transformações se necessário."""
    preds_transformed = model.predict(X_data)
    if scaler_y: 
        preds_unscaled_transformed = scaler_y.inverse_transform(preds_transformed).flatten()
    else: 
        preds_unscaled_transformed = preds_transformed.flatten()

    if is_log_target:
        preds_final = np.expm1(preds_unscaled_transformed)
        preds_final = np.nan_to_num(preds_final, nan=0.0, posinf=y_test_safe.max(), neginf=0.0) # y_test_safe do notebook
        preds_final[preds_final < 0] = 0 
    else:
        preds_final = preds_unscaled_transformed
        preds_final[preds_final < 0] = 0

    return preds_final

def evaluate_predictions(y_true, y_pred, model_label:str):
    """Calcula e imprime métricas de avaliação."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_true_safe = y_true.copy()
    y_true_safe[y_true_safe == 0] = 1e-6 # Para evitar divisão por zero no MAPE
    y_pred_safe = y_pred.copy()
    y_pred_safe[y_pred_safe == 0] = 1e-6 # Para consistência
    mape = mean_absolute_percentage_error(y_true_safe, y_pred_safe) * 100

    print(f"\nMétricas para {model_label}:")
    print(f"  R2:   {r2:.6f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape}

# --- Script Principal ---
def main():
    # 1. Carregar novos dados
    df_new = load_new_data(LATEST_API_DATA_FILE)
    if df_new.empty:
        print("Nenhum dado novo para processar após remover NaNs no alvo.")
        return

    y_true_new = df_new['Eólica']

    print("\n--- Processando Melhor Modelo (Alvo Original) ---")
    model_orig, scaler_X_orig, scaler_y_orig = load_trained_model_and_scalers(
        BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK, 'original', MODELS_PATH
    )
    if not model_orig:
        print(f"Não foi possível carregar o modelo original {BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK}.")
        preds_orig_final = np.full_like(y_true_new, np.nan) # Previsões NaN se o modelo falhar
    else:
        if 'ANN' in BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK and (not scaler_X_orig or not scaler_y_orig):
            print("Scalers X_orig ou y_orig não carregados. Previsões ANN não podem ser feitas corretamente.")
            preds_orig_final = np.full_like(y_true_new, np.nan)
        else:
            X_new_prepared_orig, feature_cols = prepare_data_for_prediction(df_new.copy(), scaler_X_orig)
            preds_orig_final = make_predictions(model_orig, X_new_prepared_orig, scaler_y_orig, is_log_target=False)

    print("\n--- Processando Melhor Modelo (Alvo Log-Transformado) ---")
    model_log, scaler_X_log, scaler_y_log = load_trained_model_and_scalers(
        BEST_MODEL_LOG_NAME_FROM_NOTEBOOK, 'log', MODELS_PATH
    )
    if not model_log:
        print(f"Não foi possível carregar o modelo log {BEST_MODEL_LOG_NAME_FROM_NOTEBOOK}.")
        preds_log_final = np.full_like(y_true_new, np.nan)
    else:

        actual_scaler_X_for_log = scaler_X_log if scaler_X_log else scaler_X_orig

        if 'ANN' in BEST_MODEL_LOG_NAME_FROM_NOTEBOOK and (not actual_scaler_X_for_log or not scaler_y_log):
            print("Scalers X_log ou y_log não carregados. Previsões ANN (log) não podem ser feitas corretamente.")
            preds_log_final = np.full_like(y_true_new, np.nan)
        else:
            X_new_prepared_log, _ = prepare_data_for_prediction(df_new.copy(), actual_scaler_X_for_log)
            preds_log_final = make_predictions(model_log, X_new_prepared_log, scaler_y_log, is_log_target=True)

    metrics_results = {}
    if not np.isnan(preds_orig_final).all(): # Avaliar apenas se as previsões não forem todas NaN
        metrics_results[f"Melhor_Original_{BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK}"] = evaluate_predictions(
            y_true_new, preds_orig_final, f"Melhor Modelo (Alvo Original - {BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK})"
        )
    if not np.isnan(preds_log_final).all():
        metrics_results[f"Melhor_Log_{BEST_MODEL_LOG_NAME_FROM_NOTEBOOK}"] = evaluate_predictions(
            y_true_new, preds_log_final, f"Melhor Modelo (Alvo Log - {BEST_MODEL_LOG_NAME_FROM_NOTEBOOK})"
        )

    df_metrics_summary = pd.DataFrame(metrics_results).T
    print("\n--- Resumo Consolidado das Métricas nos Novos Dados ---")
    print(df_metrics_summary)

    # DataFrame de comparação
    comparison_df_new = pd.DataFrame({
        'Data': df_new['Data'],
        'Eolica_Real': y_true_new
    })
    if not np.isnan(preds_orig_final).all():
        comparison_df_new[f'Pred_Melhor_Orig_{BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK}'] = preds_orig_final
    if not np.isnan(preds_log_final).all():
        comparison_df_new[f'Pred_Melhor_Log_{BEST_MODEL_LOG_NAME_FROM_NOTEBOOK}'] = preds_log_final

    print("\n--- DataFrame de Comparação Final (Primeiras e Últimas linhas) ---")
    print(comparison_df_new.head())
    print(comparison_df_new.tail())

    plt.figure(figsize=(15, 7))
    plot_limit = min(100, len(comparison_df_new))
    plt.plot(comparison_df_new['Data'][:plot_limit], comparison_df_new['Eolica_Real'][:plot_limit], label='Real', marker='o', linestyle='-', alpha=0.7)
    if f'Pred_Melhor_Orig_{BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK}' in comparison_df_new.columns:
        plt.plot(comparison_df_new['Data'][:plot_limit], comparison_df_new[f'Pred_Melhor_Orig_{BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK}'][:plot_limit],
                 label=f'Pred_Melhor_Orig ({BEST_MODEL_ORIG_NAME_FROM_NOTEBOOK})', marker='x', linestyle='--', alpha=0.7)
    if f'Pred_Melhor_Log_{BEST_MODEL_LOG_NAME_FROM_NOTEBOOK}' in comparison_df_new.columns:
        plt.plot(comparison_df_new['Data'][:plot_limit], comparison_df_new[f'Pred_Melhor_Log_{BEST_MODEL_LOG_NAME_FROM_NOTEBOOK}'][:plot_limit],
                 label=f'Pred_Melhor_Log ({BEST_MODEL_LOG_NAME_FROM_NOTEBOOK})', marker='s', linestyle=':', alpha=0.7)

    plt.title(f'Produção Eólica nos Novos Dados da API (Primeiros {plot_limit} Pontos)')
    plt.xlabel('Data')
    plt.ylabel('Energia Eólica (kW)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    output_comparison_filename = PROCESSED_DATA_PATH / f"api_data_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
    comparison_df_new.to_csv(output_comparison_filename, index=False)
    print(f"\nDataFrame de comparação salvo em: {output_comparison_filename}")


if __name__ == "__main__":
    os.makedirs(MODELS_PATH, exist_ok=True)

    y_test_safe_max_ref = 1e9 
    if Path(PROCESSED_DATA_PATH / "agg_data_ml.csv").exists():
        df_temp_hist = pd.read_csv(PROCESSED_DATA_PATH / "agg_data_ml.csv")
        y_test_safe_max_ref = df_temp_hist['Eólica'].max() * 1.5 
    y_test_safe = pd.Series([y_test_safe_max_ref]) 

    main()