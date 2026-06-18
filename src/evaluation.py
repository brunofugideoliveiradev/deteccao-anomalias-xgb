"""
Módulo de avaliação de modelos.
CORRIGIDO: Backend do matplotlib configurado para evitar problemas com Tkinter.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # CORRIGIDO: Usar backend não-interativo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    average_precision_score, 
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
import shap
from src.utils import logger, criar_diretorio


def avaliar_modelo(y_true, y_pred, y_prob, nome_modelo):
    """Avaliação completa com visualizações profissionais."""
    logger.info(f"\n--- Avaliando: {nome_modelo} ---")
    
    criar_diretorio('results/figures')
    
    # 1. Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                cbar_kws={'label': 'Quantidade'},
                linewidths=0.5, linecolor='gray',
                ax=ax)
    
    ax.set_xlabel('Previsto', fontsize=12, fontweight='bold')
    ax.set_ylabel('Real', fontsize=12, fontweight='bold')
    ax.set_title(f'Matriz de Confusão - {nome_modelo}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    plt.suptitle(f'Recall: {recall:.2%} | Precision: {precision:.2%}', 
                 fontsize=10, y=1.02)
    
    plt.savefig(f'results/figures/cm_{nome_modelo.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()  # CORRIGIDO: Fechar figura para liberar memória
    
    # 2. Métricas Completas
    report = classification_report(y_true, y_pred, output_dict=True)
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"MÉTRICAS - {nome_modelo}")
    logger.info(f"{'='*50}")
    logger.info(f"Precision: {report['1']['precision']:.4f}")
    logger.info(f"Recall: {report['1']['recall']:.4f}")
    logger.info(f"F1-Score: {report['1']['f1-score']:.4f}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info(f"PR-AUC: {pr_auc:.4f}")
    logger.info(f"{'='*50}\n")
    
    # 3. Plotar Curva ROC
    plot_curva_roc(y_true, y_prob, nome_modelo)
    
    # 4. Plotar Curva Precision-Recall
    plot_curva_precision_recall(y_true, y_prob, nome_modelo)
    
    return report['1'], pr_auc, roc_auc


def plot_curva_roc(y_true, y_prob, nome_modelo):
    """Plota a curva ROC."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Aleatório')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos', fontsize=12, fontweight='bold')
    plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12, fontweight='bold')
    plt.title(f'Curva ROC - {nome_modelo}', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.savefig(f'results/figures/roc_{nome_modelo.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()  # CORRIGIDO


def plot_curva_precision_recall(y_true, y_prob, nome_modelo):
    """Plota a curva Precision-Recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title(f'Curva Precision-Recall - {nome_modelo}', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.savefig(f'results/figures/pr_{nome_modelo.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()  # CORRIGIDO


def gerar_shap_xgboost(model, X_test, feature_names, num_samples=100):
    """Gera explicações SHAP."""
    logger.info("Gerando explicações SHAP para XGBoost...")
    criar_diretorio('results/figures')
    
    X_sample = X_test.iloc[:num_samples]
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Gráfico de Importância Global
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                     plot_type="bar", show=False, color='steelblue')
    plt.title("Importância Global das Features (XGBoost)", 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Impacto Médio (|SHAP value|)', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.savefig('results/figures/shap_xgb_global.png', 
                dpi=300, bbox_inches='tight')
    plt.close()  # CORRIGIDO
    
    # Gráfico de Beeswarm
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                     show=False, max_display=15)
    plt.title("Distribuição do Impacto das Features (SHAP)", 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig('results/figures/shap_xgb_beeswarm.png', 
                dpi=300, bbox_inches='tight')
    plt.close()  # CORRIGIDO
    
    logger.info("Gráficos SHAP salvos em results/figures/")