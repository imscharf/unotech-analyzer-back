# treinarmodelo.py

import pandas as pd
import os
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np 

# ====================================================================
# --- Configuração de Caminhos e Parâmetros (DO NOTEBOOK) ---
# ====================================================================

DATA_FILE_PATH = 'data/Cadastro.csv' 
MODELS_DIR = 'Models'
# NOVO NOME para o modelo treinado com resampling e features de tempo
NOME_ARQUIVO_MODELO = 'modelo_consumo_resampled.pkl' 
CAMINHO_MODELO_SALVO = os.path.join(MODELS_DIR, NOME_ARQUIVO_MODELO)

# Configurações do modelo do Notebook
NUM_LAGS = 1 
# Horizontes serão calculados em 'minutos' (1 passo = 1 minuto)
HORIZONTES = [1, 3, 5] 
VARIAVEL_MEDIDA = 'elapsed' 
FEATURE_TARGET_NAME = 'consumo_minuto' # Nome da série temporal reamostrada


def criar_pasta_modelos(caminho):
    """Cria a pasta de modelos se ela não existir."""
    if not os.path.exists(caminho):
        os.makedirs(caminho)
        print(f"✅ Pasta '{caminho}' criada com sucesso.")


def treinar_e_salvar_modelo_resampling(caminho_arquivo_csv, caminho_modelo_saida):
    """
    Carrega dados brutos, aplica resampling (minuto), cria features de tempo/lag,
    treina um modelo de Regressão Linear e o salva.
    """
    print("="*50)
    print("= INICIANDO TREINAMENTO COM RESAMPLING E FEATURES DE TEMPO =")
    print(f"Lendo dados de: {caminho_arquivo_csv}")
    print(f"Série Temporal Criada: Média de '{VARIAVEL_MEDIDA}' por Minuto.")
    print(f"Features: Hora, Minuto e {NUM_LAGS} Lag.")
    print("="*50)

    criar_pasta_modelos(MODELS_DIR)

    try:
        # 1. Carregar Dados e Pré-processamento
        df_bruto = pd.read_csv(caminho_arquivo_csv, encoding='latin-1') 
        df_bruto.columns = df_bruto.columns.str.strip() 

        if VARIAVEL_MEDIDA not in df_bruto.columns or 'timeStamp' not in df_bruto.columns:
            print(f"\n❌ ERRO: Coluna '{VARIAVEL_MEDIDA}' ou 'timeStamp' não encontrada no CSV.")
            return

        # 2. Resampling (Criação da Série Temporal)
        # Converter timeStamp (milissegundos) para datetime e definir como índice
        df_bruto['datetime'] = pd.to_datetime(df_bruto['timeStamp'], unit='ms')
        df_bruto = df_bruto.set_index('datetime')
        
        # Reamostragem para a média por minuto (lógica do seu notebook)
        df_ts = df_bruto[VARIAVEL_MEDIDA].resample('min').mean().to_frame(name=FEATURE_TARGET_NAME)
        df_ts.dropna(inplace=True) 

        if len(df_ts) < 2 * max(HORIZONTES):
             print(f"\n❌ ERRO: Após resampling, apenas {len(df_ts)} minutos de dados úteis.")
             print("    É necessário mais tempo de amostragem.")
             return

        # 3. Engenharia de Features e Targets
        FEATURES = []
        
        # 3a. Features de Tempo (Hora e Minuto)
        df_ts['hora_do_dia'] = df_ts.index.hour
        df_ts['minuto_da_hora'] = df_ts.index.minute
        FEATURES.extend(['hora_do_dia', 'minuto_da_hora'])
        
        # 3b. Feature de Lag
        lag_col_name = f'lag_{NUM_LAGS}'
        df_ts[lag_col_name] = df_ts[FEATURE_TARGET_NAME].shift(NUM_LAGS) 
        FEATURES.append(lag_col_name)

        # 3c. Targets
        TARGETS = []
        for h in HORIZONTES:
            target_col_name = f'Target_{h}passos'
            df_ts[target_col_name] = df_ts[FEATURE_TARGET_NAME].shift(-h) 
            TARGETS.append(target_col_name)
        
        # Limpar NaN's resultantes do lag e target
        df_ts.dropna(inplace=True) 
        
        # 4. Preparar Dados
        X = df_ts[FEATURES]
        y = df_ts[TARGETS]
        
        # Usar test_size=0.2 (como no seu notebook)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        if len(X_train) == 0:
             print("\n❌ ERRO: O conjunto de treinamento ficou vazio.")
             return

        print(f"Dados prontos. Treino: {len(X_train)} | Teste: {len(X_test)}")
        print(f"Features utilizadas: {FEATURES}")

        # 5. Treinar Modelo (Regressão Linear)
        modelo = LinearRegression(n_jobs=-1)
        modelo.fit(X_train, y_train)
        print("Modelo de Regressão Linear (Resampled) treinado com sucesso.")

        # 6. Avaliar
        y_pred = modelo.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
        r2 = r2_score(y_test, y_pred, multioutput='raw_values')
        
        print("\n--- Métricas de Regressão de Teste ---")
        for i, h in enumerate(HORIZONTES):
             print(f"Horizonte {h} minutos: MSE={mse[i]:.2f}, R2={r2[i]:.4f}")
        
        # 7. Salvar Modelo
        joblib.dump(modelo, caminho_modelo_saida)
        print(f"\n✅ Modelo treinado salvo em: {caminho_modelo_saida}")

    except FileNotFoundError:
        print(f"\n❌ ERRO: O arquivo '{caminho_arquivo_csv}' não foi encontrado.")
    except Exception as e:
        print(f"\n❌ Ocorreu um erro durante o treinamento: {e}")
    print("="*50)


if __name__ == "__main__":
    treinar_e_salvar_modelo_resampling(DATA_FILE_PATH, CAMINHO_MODELO_SALVO)