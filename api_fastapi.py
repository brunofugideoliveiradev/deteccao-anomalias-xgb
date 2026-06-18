"""
API REST para Detecção de Fraudes usando FastAPI.
Versão Integrada com Modelo Real.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
from contextlib import asynccontextmanager

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variáveis globais
modelo = None
scaler = None
historico_predicoes = []

# =========================================================================
# MODELOS DE DADOS
# =========================================================================
class TransacaoInput(BaseModel):
    """
    Dados de entrada. Para simplificar, aceitamos um array de features 
    ou um dicionário. Em produção, seria um schema rígido com as 49 features.
    """
    features: List[float] = Field(..., description="Array com as features da transação (49 valores)")

class PredicaoOutput(BaseModel):
    transacao_id: str
    probabilidade_fraude: float
    status: str
    threshold_utilizado: float
    timestamp: str
    modelo_utilizado: str

class HealthCheck(BaseModel):
    status: str
    modelo_carregado: bool
    versao: str
    timestamp: str
    total_predicoes: int

# =========================================================================
# LIFESPAN (Carregar Modelo Real)
# =========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global modelo, scaler, threshold_otimo, nome_modelo_real
    
    # Carregar configuração real gerada pelo main.py
    config_path = "results/dados_streamlit.pkl"
    modelo_path = "models/pipeline_Random_Forest.pkl"
        
    if os.path.exists(config_path):
            config = joblib.load(config_path)
            nome_modelo_real = config['melhor_modelo']
            threshold_otimo = config['threshold_otimo']
            
            # Mapear nome para arquivo (simplificado)
            # Em produção, salvaríamos o caminho exato no config
            modelo_path = f"models/pipeline_{nome_modelo_real.replace(' ', '_')}.pkl"
            
            if os.path.exists(modelo_path):
                modelo = joblib.load(modelo_path)
                logger.info(f"✅ Carregado modelo real: {nome_modelo_real} (Threshold: {threshold_otimo})")
            
    
    yield
    logger.info("API encerrada")

# =========================================================================
# API
# =========================================================================
app = FastAPI(
    title="🔍 API de Detecção de Fraudes (Real)",
    description="Microsserviço para detecção de fraudes",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/", response_model=dict, tags=["Root"])
async def root():
    return {
        "message": "API de Detecção de Fraudes - Versão Real",
        "version": "2.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthCheck, tags=["Monitoramento"])
async def health_check():
    return HealthCheck(
        status="healthy" if modelo is not None else "degraded",
        modelo_carregado=modelo is not None,
        versao="2.0.0",
        timestamp=datetime.now().isoformat(),
        total_predicoes=len(historico_predicoes)
    )

@app.post("/predict", response_model=PredicaoOutput, tags=["Predição"])
async def prever_fraude(transacao: TransacaoInput, background_tasks: BackgroundTasks):
    logger.info(f"Recebendo transação com {len(transacao.features)} features")
    
    try:
        if modelo is not None:
            # Usar modelo real
            features_array = np.array([transacao.features])
            
            # O pipeline já faz o scaling internamente se estiver configurado
            probabilidade = modelo.predict_proba(features_array)[0][1]
            modelo_nome = "Random Forest Pipeline (Real)"
        else:
            # Fallback simulado
            probabilidade = 0.1 if len(transacao.features) > 10 else 0.9
            modelo_nome = "Simulação (Modelo não carregado)"
        
        threshold = threshold_otimo
        status = "BLOCKED" if probabilidade >= threshold else "APPROVED"
        transacao_id = f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        output = PredicaoOutput(
            transacao_id=transacao_id,
            probabilidade_fraude=round(float(probabilidade), 4),
            status=status,
            threshold_utilizado=threshold,
            timestamp=datetime.now().isoformat(),
            modelo_utilizado=modelo_nome
        )
        
        background_tasks.add_task(lambda: historico_predicoes.append(output.dict()))
        logger.info(f"Transação {transacao_id}: {status} (prob: {probabilidade:.4f})")
        
        return output
        
    except Exception as e:
        logger.error(f"Erro ao processar: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info(" Iniciando API...")
    uvicorn.run("api_fastapi:app", host="0.0.0.0", port=8000, reload=True)