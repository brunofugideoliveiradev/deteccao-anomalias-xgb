"""
Modelos Adicionais Não-Supervisionados.
Versão SIMPLIFICADA - removidos LOF e One-Class SVM (redundantes).
"""
from src.utils import logger
import numpy as np


def treinar_lof(X_train_normal, X_test, contamination=0.0017):
    """
    LOF REMOVIDO - Redundante com Isolation Forest.
    Esta função foi mantida apenas para compatibilidade.
    """
    logger.warning("⚠️ LOF foi removido do projeto (redundante com Isolation Forest)")
    logger.info("Use Isolation Forest para detecção não-supervisionada")
    
    # Retornar valores dummy para não quebrar o código
    y_pred = np.zeros(len(X_test))
    y_prob = np.zeros(len(X_test))
    
    return None, y_pred, y_prob


def treinar_one_class_svm(X_train_normal, X_test, nu=0.0017):
    """
    One-Class SVM REMOVIDO - Redundante com Isolation Forest.
    Esta função foi mantida apenas para compatibilidade.
    """
    logger.warning("⚠️ One-Class SVM foi removido do projeto (redundante)")
    logger.info("Use Isolation Forest para detecção não-supervisionada")
    
    # Retornar valores dummy para não quebrar o código
    y_pred = np.zeros(len(X_test))
    y_prob = np.zeros(len(X_test))
    
    return None, y_pred, y_prob