import pandas as pd
from src.utils import logger

def carregar_dataset(url: str) -> pd.DataFrame:
    try:
        logger.info(f"Carregando dataset de: {url}")
        df = pd.read_csv(url)
        if 'Class' not in df.columns:
            raise ValueError("Coluna 'Class' não encontrada!")
        logger.info(f"Sucesso: {df.shape[0]} linhas, {df.shape[1]} colunas")
        return df
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        raise