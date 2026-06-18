"""
Módulo de Engenharia de Features para Detecção de Fraudes.
Versão OTIMIZADA com features de interação, polinomiais e risco composto.
"""
import pandas as pd
import numpy as np
from src.utils import logger


def criar_features_temporais(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features baseadas no tempo da transação."""
    logger.info("Criando features temporais...")
    df = df.copy()
    
    # Converter segundos para horas
    df['Hour'] = (df['Time'] // 3600) % 24
    
    # Dia da semana
    df['DayOfWeek'] = (df['Time'] // 86400) % 7
    
    # Período do dia
    def periodo_dia(hour):
        if 0 <= hour < 6:
            return 'madrugada'
        elif 6 <= hour < 12:
            return 'manha'
        elif 12 <= hour < 18:
            return 'tarde'
        else:
            return 'noite'
    
    df['PeriodoDia'] = df['Hour'].apply(periodo_dia)
    
    # É fim de semana?
    df['EhFimDeSemana'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # É madrugada?
    df['EhMadrugada'] = (df['Hour'] < 6).astype(int)
    
    # É noite?
    df['EhNoite'] = ((df['Hour'] >= 18) | (df['Hour'] < 6)).astype(int)
    
    logger.info("Features temporais criadas")
    return df


def criar_features_frequencia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features de frequência usando operações VETORIZADAS (rápido!).
    """
    logger.info("Criando features de frequência (otimizado)...")
    df = df.copy()
    
    # Ordenar por tempo
    df = df.sort_values('Time').reset_index(drop=True)
    
    # Usar searchsorted para encontrar índices de janelas temporais
    times = df['Time'].values
    
    # Transações na última hora (3600 segundos)
    start_times = times - 3600
    indices_inicio = np.searchsorted(times, start_times, side='left')
    indices_fim = np.arange(len(times))
    df['TransacoesUltimaHora'] = indices_fim - indices_inicio + 1
    
    # Transações nas últimas 24 horas (86400 segundos)
    start_times_24h = times - 86400
    indices_inicio_24h = np.searchsorted(times, start_times_24h, side='left')
    df['TransacoesUltimoDia'] = indices_fim - indices_inicio_24h + 1
    
    # Velocidade de transações
    df['VelocidadeTransacoes'] = df['TransacoesUltimaHora']
    
    logger.info("Features de frequência criadas (vetorizado - rápido!)")
    return df


def criar_features_valor(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features baseadas no valor da transação."""
    logger.info("Criando features de valor...")
    df = df.copy()
    
    # Média móvel do valor (últimas 10 transações)
    df['MediaMovelAmount'] = df['Amount'].rolling(window=10, min_periods=1).mean()
    
    # Desvio padrão móvel
    df['DesvioPadraoAmount'] = df['Amount'].rolling(window=10, min_periods=1).std().fillna(0)
    
    # Valor vs média
    df['ValorVsMedia'] = (df['Amount'] - df['MediaMovelAmount']) / (df['DesvioPadraoAmount'] + 1e-8)
    
    # Percentis
    percentil_95 = df['Amount'].quantile(0.95)
    percentil_5 = df['Amount'].quantile(0.05)
    
    df['ValorMuitoAlto'] = (df['Amount'] > percentil_95).astype(int)
    df['ValorMuitoBaixo'] = (df['Amount'] < percentil_5).astype(int)
    
    # Log do valor
    df['LogAmount'] = np.log1p(df['Amount'])
    
    logger.info("Features de valor criadas")
    return df


def criar_features_distancia_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features baseadas na distância temporal."""
    logger.info("Criando features de distância temporal...")
    df = df.copy()
    
    # Tempo desde a última transação
    df['TempoDesdeUltima'] = df['Time'].diff().fillna(0)
    
    # Tempo desde a primeira transação
    df['TempoDesdePrimeira'] = df['Time'] - df['Time'].min()
    
    # Transação muito rápida?
    df['TransacaoRapida'] = (df['TempoDesdeUltima'] < 10).astype(int)
    
    logger.info("Features de distância temporal criadas")
    return df


def criar_features_interacao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features de interação entre variáveis.
    """
    logger.info("Criando features de interação...")
    df = df.copy()
    
    # Amount × Hora (fraudes altas à noite)
    df['Amount_Hour'] = df['Amount'] * df['Hour']
    
    # Amount × Fim de Semana
    df['Amount_Weekend'] = df['Amount'] * df['EhFimDeSemana']
    
    # Amount × Madrugada
    df['Amount_Madrugada'] = df['Amount'] * df['EhMadrugada']
    
    # Frequência × Valor
    df['Freq_Value'] = df['TransacoesUltimaHora'] * df['Amount']
    
    # Valor × Rapidez
    df['Value_Speed'] = df['Amount'] * df['TransacaoRapida']
    
    # Hora × Fim de Semana
    df['Hour_Weekend'] = df['Hour'] * df['EhFimDeSemana']
    
    logger.info("Features de interação criadas")
    return df


def criar_features_polinomiais(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features polinomiais e cíclicas.
    """
    logger.info("Criando features polinomiais...")
    df = df.copy()
    
    # Amount polinomial
    df['Amount_Squared'] = df['Amount'] ** 2
    df['Amount_Sqrt'] = np.sqrt(df['Amount'])
    df['Amount_Cube'] = df['Amount'] ** 3
    
    # Hour cíclico (sin/cos para capturar periodicidade)
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    
    # DayOfWeek cíclico
    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    # Log de features de frequência
    df['Log_TransacoesUltimaHora'] = np.log1p(df['TransacoesUltimaHora'])
    df['Log_TransacoesUltimoDia'] = np.log1p(df['TransacoesUltimoDia'])
    
    logger.info("Features polinomiais criadas")
    return df


def criar_features_risco_composto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria score de risco combinado.
    """
    logger.info("Criando features de risco composto...")
    df = df.copy()
    
    # Score de risco ponderado
    df['RiskScore'] = (
        df['EhMadrugada'] * 0.15 +
        df['ValorMuitoAlto'] * 0.25 +
        (df['TransacoesUltimaHora'] > 5).astype(int) * 0.20 +
        df['TransacaoRapida'] * 0.20 +
        df['EhFimDeSemana'] * 0.10 +
        (df['ValorVsMedia'] > 2).astype(int) * 0.10
    )
    
    # Score de risco alto
    df['HighRiskScore'] = (
        (df['EhMadrugada'] & df['ValorMuitoAlto']).astype(int) +
        (df['TransacoesUltimaHora'] > 10).astype(int) +
        (df['TransacaoRapida'] & df['ValorMuitoAlto']).astype(int)
    )
    
    # Flag de risco crítico
    df['CriticalRisk'] = (
        (df['EhMadrugada'] & df['ValorMuitoAlto'] & (df['TransacoesUltimaHora'] > 3))
    ).astype(int)
    
    logger.info("Features de risco composto criadas")
    return df


def criar_todas_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas as engenharias de features."""
    logger.info("="*60)
    logger.info("INICIANDO ENGENHARIA DE FEATURES (OTIMIZADA)")
    logger.info("="*60)
    
    df = criar_features_temporais(df)
    df = criar_features_frequencia(df)
    df = criar_features_valor(df)
    df = criar_features_distancia_temporal(df)
    df = criar_features_interacao(df)  # NOVO!
    df = criar_features_polinomiais(df)  # NOVO!
    df = criar_features_risco_composto(df)  # NOVO!
    
    # One-Hot Encoding
    df = pd.get_dummies(df, columns=['PeriodoDia'], prefix='Periodo')
    
    logger.info(f"Total de features após engenharia: {df.shape[1]}")
    logger.info("="*60)
    
    return df