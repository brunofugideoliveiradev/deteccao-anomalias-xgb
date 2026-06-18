"""
Motor de Decisão Profissional.
Implementa Threshold Dinâmico baseado em custo e Sistema Híbrido (Regras + ML).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from src.utils import logger


def calcular_threshold_dinamico(y_true, y_prob, custo_falso_positivo=100.0, custo_falso_negativo=1000.0):
    """
    Calcula o threshold ótimo que minimiza o custo financeiro total.
    
    Args:
        y_true: Valores reais (0 ou 1)
        y_prob: Probabilidades preditas pelo modelo
        custo_falso_positivo: Custo de bloquear um cliente bom (ex: R$ 100)
        custo_falso_negativo: Custo de deixar passar uma fraude (ex: R$ 1000)
        
    Returns:
        Tuple com (threshold_otimo, custo_minimo, matriz_de_custo)
    """
    logger.info(f"Calculando threshold dinâmico (Custo FP: R${custo_falso_positivo}, Custo FN: R${custo_falso_negativo})...")
    
    thresholds = np.arange(0.05, 0.95, 0.01)
    custos = []
    
    for threshold in thresholds:
        y_pred_temp = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_temp).ravel()
        
        custo_total = (fp * custo_falso_positivo) + (fn * custo_falso_negativo)
        custos.append(custo_total)
    
    threshold_otimo = thresholds[np.argmin(custos)]
    custo_minimo = min(custos)
    
    logger.info(f"Threshold ótimo encontrado: {threshold_otimo:.2f}")
    logger.info(f"Custo financeiro mínimo estimado: R$ {custo_minimo:,.2f}")
    
    return threshold_otimo, custo_minimo


def sistema_hibrido(df_X, modelo, threshold_ml=0.5, 
                    limite_valor_alto=None, limite_madrugada=None):
    # Se não passar, calcula dinamicamente baseado no dataset
    if limite_valor_alto is None:
        limite_valor_alto = df_X['Amount'].quantile(0.99) # Top 1% valores
    """
    Sistema Híbrido: Combina Regras de Negócio (Hard Rules) com Machine Learning.
    
    Regras implementadas:
    1. Se o valor for extremamente alto (> 5000), bloqueia automaticamente.
    2. Se for madrugada (Hour < 5) E o valor for alto (> 2000), bloqueia.
    3. Caso contrário, usa o modelo de ML com o threshold definido.
    
    Args:
        df_X: DataFrame com as features originais (não escalonadas, para as regras funcionarem)
        modelo: O modelo ou pipeline treinado
        threshold_ml: Threshold para a decisão do ML
        
    Returns:
        Tuple com (y_pred_hibrido, y_prob_hibrido)
    """
    logger.info("Aplicando Sistema Híbrido (Regras de Negócio + ML)...")
    
    # 1. Decisão baseada puramente no ML
    y_prob_ml = modelo.predict_proba(df_X)[:, 1]
    y_pred_ml = (y_prob_ml >= threshold_ml).astype(int)
    
    # 2. Aplicação das Regras de Negócio
    y_pred_final = y_pred_ml.copy()
    
    # Regra 1: Valor extremamente alto
    if 'Amount' in df_X.columns:
        regra_valor_alto = df_X['Amount'] > 5000
        y_pred_final[regra_valor_alto] = 1
        logger.info(f"Regra 1 (Valor > 5000): {regra_valor_alto.sum()} transações bloqueadas automaticamente.")
        
        # Regra 2: Madrugada com valor alto
        if 'Hour' in df_X.columns:
            regra_madrugada = (df_X['Hour'] < 5) & (df_X['Amount'] > 2000)
            # Aplica apenas onde a regra é verdadeira e o ML não havia bloqueado (para contar o impacto da regra)
            impacto_regra2 = regra_madrugada & (y_pred_ml == 0)
            y_pred_final[regra_madrugada] = 1
            logger.info(f"Regra 2 (Madrugada + Valor > 2000): {impacto_regra2.sum()} fraudes adicionais detectadas pela regra.")
    
    logger.info(f"Sistema Híbrido concluído. Total de alertas: {y_pred_final.sum()}")
    
    return y_pred_final, y_prob_ml