import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def criar_diretorio(caminho: str) -> None:
    if not os.path.exists(caminho):
        os.makedirs(caminho)
        logger.info(f"Diretório criado: {caminho}")
        