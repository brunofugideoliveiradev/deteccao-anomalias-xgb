"""
Módulo de Pipelines do Scikit-learn.
Versão OTIMIZADA - sem CatBoost (redundante).
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from src.utils import logger


def criar_pipeline_random_forest():
    """Cria Pipeline para Random Forest."""
    logger.info("Criando Pipeline Random Forest...")
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=1, oob_score=True
        ))
    ])

def criar_pipeline_xgboost():
    """Cria Pipeline para XGBoost."""
    logger.info("Criando Pipeline XGBoost...")
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6, min_child_weight=3,
            gamma=0.1, subsample=0.8, colsample_bytree=0.8, eval_metric='aucpr',
            random_state=42, n_jobs=1
        ))
    ])


def criar_pipeline_lightgbm():
    """Cria Pipeline para LightGBM."""
    logger.info("Criando Pipeline LightGBM...")
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=1, verbose=-1
        ))
    ])


def criar_pipeline_stacking():
    """Cria Pipeline para Stacking Ensemble OTIMIZADO."""
    logger.info("Criando Pipeline Stacking Ensemble (rápido)...")
    
    # Apenas 2 modelos base
    estimators = [
        ('rf', RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight='balanced',
            random_state=42, n_jobs=-1
        )),
        ('xgb', XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            eval_metric='aucpr', random_state=42, n_jobs=-1
        ))
    ]
    
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
        cv=3, n_jobs=-1, passthrough=False
    )
    
    return Pipeline([
        ('scaler', StandardScaler()),
        ('stacking', stacking)
    ])


def criar_pipeline_voting():
    """Cria Pipeline para Voting Classifier."""
    logger.info("Criando Pipeline Voting Classifier...")
    
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42, n_jobs=1)),
        ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, eval_metric='aucpr', random_state=42, n_jobs=1)),
        ('lgbm', LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, is_unbalance=True, random_state=42, n_jobs=1, verbose=-1))
    ]
    
    voting = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=1
    )
    
    return Pipeline([('scaler', StandardScaler()), ('voting', voting)])


def treinar_pipeline(pipeline, X_train, y_train, X_test, nome="Pipeline"):
    """Treina pipeline e retorna predições."""
    logger.info(f"Treinando {nome}...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    logger.info(f"{nome} treinado. Fraudes detectadas: {y_pred.sum()}")
    return pipeline, y_pred, y_prob