# API/apiprojeto.py

import os
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from io import StringIO
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# ====================================================================
# --- Configuração da API e Carregamento do Modelo ---
# ====================================================================

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://unotech-analyzer.vercel.app/"
]

app = FastAPI(
    title="API de Predição de Consumo por Minuto (Resampled)",
    description="API para prever o consumo médio futuro em janelas de 1, 3 e 5 minutos, usando a média por minuto como base."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Ou ["*"] para permitir qualquer origem (não recomendado p/ produção)
    allow_credentials=True,
    allow_methods=["*"],   # Permite GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)

# Caminho e variáveis ajustados para o novo modelo Resampled
MODEL_PATH = os.path.join('Models', 'modelo_consumo_resampled.pkl') # CORRIGIDO: Usando os.path.join
NUM_LAGS = 1 
VARIAVEL_MEDIDA = 'elapsed' 
FEATURE_TARGET_NAME = 'consumo_minuto' # Nome da série temporal reamostrada


try:
    modelo_ia = joblib.load(MODEL_PATH)
    print(f"✅ Modelo Resampled carregado com sucesso de: {MODEL_PATH}")
except Exception:
    modelo_ia = None
    print(f"❌ ERRO: Modelo não carregado. Execute treinarmodelo.py primeiro.")


# ====================================================================
# --- Definição do Esquema de Retorno da Predição ---
# ====================================================================

class PredicaoJanela(BaseModel):
    janela_minutos: int 
    consumo_medio_previsto: float
    tendencia: str 

class ResultadoPredicao(BaseModel):
    # O último valor é a média do último minuto completo
    consumo_ultimo_minuto_medio: float 
    previsoes: List[PredicaoJanela]


# ====================================================================
# --- ENDPOINTS ---
# ====================================================================

@app.get("/", summary="Verifica o Status da API")
def read_root():
    """Endpoint básico para checagem de saúde da API."""
    status_modelo = "Modelo de consumo por minuto carregado e pronto." if modelo_ia else "Modelo ausente ou com erro."
    return {
        "status": "API em execução! 🚀",
        "servico": "Predição de Consumo (Resampled)",
        "modelo_status": status_modelo
    }

@app.post("/predict", response_model=ResultadoPredicao, summary=f"Recebe CSV e prevê o consumo médio para 1, 3 e 5 minutos futuros.")
async def predict_latency(file: UploadFile = File(...)):
    """
    Recebe um novo arquivo CSV, calcula a média do consumo do último minuto completo
    e usa as features de tempo/lag para prever os próximos minutos.
    """
    if modelo_ia is None:
        raise HTTPException(status_code=503, detail="Modelo de IA não foi carregado corretamente.")

    # 1. Leitura e Pré-processamento do CSV
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        df_bruto = pd.read_csv(StringIO(content_str), encoding='latin-1')
        df_bruto.columns = df_bruto.columns.str.strip()
        
        # Converte timeStamp (milissegundos) para datetime
        df_bruto['datetime'] = pd.to_datetime(df_bruto['timeStamp'], unit='ms')
        df_bruto = df_bruto.set_index('datetime')
        
        if VARIAVEL_MEDIDA not in df_bruto.columns:
            raise HTTPException(status_code=422, detail=f"O CSV deve conter a coluna '{VARIAVEL_MEDIDA}'.")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao ler ou processar o arquivo CSV: {e}")

    # 2. Resampling para obter o Lag e a Hora
    # Reamostragem: Média do consumo (elapsed) por minuto
    df_ts = df_bruto[VARIAVEL_MEDIDA].resample('min').mean().to_frame(name=FEATURE_TARGET_NAME)
    df_ts.dropna(inplace=True)

    if len(df_ts) < NUM_LAGS:
        raise HTTPException(status_code=422, detail=f"CSV precisa de pelo menos {NUM_LAGS} minuto(s) completo(s) de dados úteis após o resampling.")

    # Pega o último ponto conhecido da série temporal (que será o LAG)
    ultimo_ponto = df_ts.iloc[-1]
    ultimo_consumo_medio = ultimo_ponto[FEATURE_TARGET_NAME]
    
    # 3. Engenharia de Features para a Predição
    
    # A hora/minuto para a PREDIÇÃO (é o próximo minuto após o último ponto conhecido)
    prox_timestamp = ultimo_ponto.name + pd.Timedelta(minutes=1)
    
    hora_do_dia = prox_timestamp.hour
    minuto_da_hora = prox_timestamp.minute
    
    # O lag é o último valor conhecido
    lag_1 = ultimo_consumo_medio
    
    # 4. Construção do Vetor de Entrada X (deve ter 3 features)
    X_input = pd.DataFrame({
        'hora_do_dia': [hora_do_dia],
        'minuto_da_hora': [minuto_da_hora],
        f'lag_{NUM_LAGS}': [lag_1]
    })
    
    # 5. Predição
    try:
        # predicoes = [Pred_1passo, Pred_3passos, Pred_5passos]
        predicoes = modelo_ia.predict(X_input)[0] 
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno durante a predição: {e}")

    # 6. Construção da Resposta Formatada
    horizontes_passos = [1, 3, 5]
    resultados = []
    
    for i, h in enumerate(horizontes_passos):
        pred_val = predicoes[i]
        
        tendencia = ""
        if pred_val > ultimo_consumo_medio * 1.05:
            tendencia = "Aumento Forte"
        elif pred_val > ultimo_consumo_medio:
            tendencia = "Leve Aumento"
        elif pred_val < ultimo_consumo_medio * 0.95:
            tendencia = "Queda Forte"
        elif pred_val < ultimo_consumo_medio:
            tendencia = "Leve Queda"
        else:
            tendencia = "Estável"

        resultados.append(
            PredicaoJanela(
                janela_minutos=h,
                consumo_medio_previsto=round(pred_val, 2),
                tendencia=tendencia
            )
        )

    return ResultadoPredicao(
        consumo_ultimo_minuto_medio=round(ultimo_consumo_medio, 2),
        previsoes=resultados
    )