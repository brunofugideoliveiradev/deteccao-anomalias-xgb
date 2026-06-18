"""
Módulo de modelos de Machine Learning.
Versão CORRIGIDA: removido use_label_encoder, GridSearch otimizado.
"""
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from src.utils import logger


def treinar_isolation_forest(X_train, X_test):
    """
    Isolation Forest (não-supervisionado, sem pipeline).
    """
    logger.info("Treinando Isolation Forest (otimizado)...")
    
    model = IsolationForest(
        contamination=0.01,
        n_estimators=200,
        max_samples='auto',
        max_features=1.0,
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train)
    y_pred_raw = model.predict(X_test)
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    y_prob = -model.score_samples(X_test)
    
    # Normalizar probabilidades entre 0 e 1
    y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-10)
    
    logger.info(f"Isolation Forest treinado. Fraudes detectadas: {y_pred.sum()}")
    return model, y_pred, y_prob


def treinar_random_forest(X_train, y_train, X_test, tuning=False):
    """
    Random Forest com hyperparâmetros otimizados.
    Se tuning=True, usa GridSearch (SEM paralelismo para evitar problemas).
    """
    if tuning:
        logger.info("Treinando Random Forest com GridSearch (tuning)...")
        
        # Grid simplificado para ser mais rápido e estável
        param_grid = {
            'n_estimators': [200],
            'max_depth': [15, 18],
            'min_samples_split': [5],
            'min_samples_leaf': [2],
            'class_weight': ['balanced']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=1, oob_score=True)
        
        # GridSearch SEM paralelismo (n_jobs=1) para evitar vazamento de memória
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,
            scoring='f1',
            n_jobs=1,  # CORRIGIDO: SEM paralelismo
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        model = grid_search.best_estimator_
        logger.info(f"Melhores parâmetros: {grid_search.best_params_}")
        logger.info(f"Melhor F1-Score (CV): {grid_search.best_score_:.4f}")
    else:
        logger.info("Treinando Random Forest (otimizado)...")
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=1,  # CORRIGIDO: SEM paralelismo
            oob_score=True
        )
        
        model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    logger.info(f"Random Forest treinado. Fraudes detectadas: {y_pred.sum()}")
    return model, y_pred, y_prob


def treinar_xgboost(X_train, y_train, X_test, tuning=False):
    """
    XGBoost com hyperparâmetros otimizados.
    CORRIGIDO: removido use_label_encoder (obsoleto).
    """
    if tuning:
        logger.info("Treinando XGBoost com GridSearch (tuning)...")
        
        # Grid simplificado
        param_grid = {
            'n_estimators': [200],
            'max_depth': [6, 7],
            'learning_rate': [0.05],
            'min_child_weight': [3],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
        
        # CORRIGIDO: SEM use_label_encoder
        xgb = XGBClassifier(
            eval_metric='aucpr',
            n_jobs=1,  # CORRIGIDO: SEM paralelismo
            random_state=42
        )
        
        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=3,
            scoring='f1',
            n_jobs=1,  # CORRIGIDO: SEM paralelismo
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        model = grid_search.best_estimator_
        logger.info(f"Melhores parâmetros: {grid_search.best_params_}")
        logger.info(f"Melhor F1-Score (CV): {grid_search.best_score_:.4f}")
    else:
        logger.info("Treinando XGBoost (otimizado)...")
        
        # CORRIGIDO: SEM use_label_encoder
        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            eval_metric='aucpr',
            random_state=42,
            n_jobs=1  # CORRIGIDO: SEM paralelismo
        )
        
        model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    logger.info(f"XGBoost treinado. Fraudes detectadas: {y_pred.sum()}")
    return model, y_pred, y_prob


def treinar_lightgbm(X_train, y_train, X_test):
    """
    LightGBM - Modelo rápido e eficiente.
    """
    logger.info("Treinando LightGBM...")
    
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=1,  # CORRIGIDO: SEM paralelismo
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    logger.info(f"LightGBM treinado. Fraudes detectadas: {y_pred.sum()}")
    return model, y_pred, y_prob



def criar_stacking_ensemble():
    """
    Stacking Ensemble OTIMIZADO com meta-modelo XGBoost.
    CORRIGIDO: SEM use_label_encoder, SEM paralelismo excessivo.
    """
    logger.info("Criando Stacking Ensemble (otimizado com XGBoost)...")
    
    # Modelos base
    estimators = [
        ('rf', RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=1  # CORRIGIDO
        )),
        ('xgb', XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=10,
            eval_metric='aucpr',
            random_state=42,
            n_jobs=1  # CORRIGIDO: SEM use_label_encoder
        )),
        ('lgbm', LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            is_unbalance=True,
            random_state=42,
            n_jobs=1,  # CORRIGIDO
            verbose=-1
        ))
    ]
        # Meta-modelo XGBoost
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            scale_pos_weight=10,
            eval_metric='aucpr',
            random_state=42,
            n_jobs=1  # CORRIGIDO: SEM use_label_encoder
        ),
        cv=5,
        n_jobs=1,  # CORRIGIDO: SEM paralelismo
        passthrough=False
    )
    
    logger.info("Stacking Ensemble otimizado criado")
    return stacking


def treinar_ensemble(ensemble_model, X_train, y_train, X_test, nome="Ensemble"):
    """
    Treina modelo ensemble.
    """
    logger.info(f"Treinando {nome}...")
    
    ensemble_model.fit(X_train, y_train)
    y_pred = ensemble_model.predict(X_test)
    y_prob = ensemble_model.predict_proba(X_test)[:, 1]
    
    logger.info(f"{nome} treinado. Fraudes detectadas: {y_pred.sum()}")
    return ensemble_model, y_pred, y_prob