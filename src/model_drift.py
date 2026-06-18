"""
Detecção de Model Drift.
Monitora quando o modelo perde performance ao longo do tempo.
"""
import numpy as np
from scipy import stats
from src.utils import logger


class ModelDriftDetector:
    """
    Detector de drift para monitoramento de modelos.
    """
    
    def __init__(self, X_reference, threshold=0.05):
        """
        Inicializa o detector.
        
        Args:
            X_reference: Dados de referência (treino original)
            threshold: Valor de p-value para detectar drift (default: 0.05)
        """
        self.X_reference = X_reference
        self.threshold = threshold
        self.historico_drift = []
        
        logger.info(f"Detector de drift inicializado (threshold: {threshold})")
    
    def detectar_drift_ks_test(self, X_atual):
        """
        Detecta drift usando teste Kolmogorov-Smirnov.
        
        Returns:
            bool: True se drift detectado, False caso contrário
            dict: Detalhes do drift por feature
        """
        logger.info("Analisando drift com KS Test...")
        
        drift_detected = False
        detalhes = {}
        
        for col in X_atual.columns:
            if col in self.X_reference.columns:
                stat, p_value = stats.ks_2samp(
                    self.X_reference[col],
                    X_atual[col]
                )
                
                drift = p_value < self.threshold
                if drift:
                    drift_detected = True
                
                detalhes[col] = {
                    'drift': drift,
                    'p_value': p_value,
                    'statistic': stat
                }
        
        self.historico_drift.append({
            'drift_detected': drift_detected,
            'features_com_drift': sum(1 for v in detalhes.values() if v['drift'])
        })
        
        return drift_detected, detalhes
    
    def detectar_drift_psi(self, X_atual, num_bins=10):
        """
        Detecta drift usando Population Stability Index (PSI).
        
        Returns:
            float: PSI total
            dict: PSI por feature
        """
        logger.info("Analisando drift com PSI...")
        
        psi_total = 0
        psi_detalhes = {}
        
        for col in X_atual.columns:
            if col in self.X_reference.columns:
                psi = self._calcular_psi(
                    self.X_reference[col],
                    X_atual[col],
                    num_bins
                )
                
                psi_total += psi
                psi_detalhes[col] = psi
        
        return psi_total, psi_detalhes
    
    def _calcular_psi(self, actual, expected, num_bins=10):
        """
        Calcula PSI entre duas distribuições.
        """
        # Criar bins
        breakpoints = np.linspace(
            min(expected.min(), actual.min()),
            max(expected.max(), actual.max()),
            num_bins + 1
        )
        
        # Calcular percentuais
        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
        
        # Evitar divisão por zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        # Calcular PSI
        psi = np.sum((actual_percents - expected_percents) * 
                    np.log(actual_percents / expected_percents))
        
        return psi
    
    def gerar_relatorio_drift(self):
        """
        Gera relatório de histórico de drift.
        """
        logger.info("\n" + "="*60)
        logger.info("RELATÓRIO DE DRIFT")
        logger.info("="*60)
        
        if not self.historico_drift:
            logger.info("Nenhum drift detectado até o momento.")
            return
        
        total_analises = len(self.historico_drift)
        vezes_com_drift = sum(1 for h in self.historico_drift if h['drift_detected'])
        
        logger.info(f"Total de análises: {total_analises}")
        logger.info(f"Vezes com drift: {vezes_com_drift}")
        logger.info(f"Taxa de drift: {vezes_com_drift/total_analises*100:.1f}%")
        
        if vezes_com_drift > total_analises * 0.3:
            logger.warning("⚠️ ALERTA: Drift frequente detectado!")
            logger.warning("Considere retreinar o modelo.")
        
        logger.info("="*60 + "\n")