import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.utils import logger

def preparar_dados(df: pd.DataFrame):
    logger.info("Preparando features e target...")
    
    # IMPORTANTE: Aplicar engenharia de features ANTES de remover colunas
    from src.feature_engineering import criar_todas_features
    df = criar_todas_features(df)
    
    # Agora sim, remover colunas que não serão usadas
    # Removemos 'Time' (já extraímos features dele) e 'Class' (target)
    X = df.drop(['Class', 'Time'], axis=1)
    y = df['Class']
    
    # Escalonar Amount (e as novas features numéricas se necessário)
    scaler = StandardScaler()
    colunas_escalonar = ['Amount', 'MediaMovelAmount', 'DesvioPadraoAmount', 
                         'TempoDesdeUltima', 'TempoDesdePrimeira']
    
    for col in colunas_escalonar:
        if col in X.columns:
            X[col] = scaler.fit_transform(X[[col]])
    
    return X, y

def dividir_e_balancear(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
    logger.info("Dividindo dados (estratificado)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info("Aplicando SMOTE apenas no conjunto de TREINO...")
    smote = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    logger.info(f"Shape após SMOTE: {X_train_bal.shape}")
    return X_train_bal, X_test, y_train_bal, y_test