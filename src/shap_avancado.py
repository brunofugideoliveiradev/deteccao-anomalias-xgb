"""
SHAP Avançado - Explicabilidade detalhada de modelos.
Inclui dependence plots, interaction values e waterfall plots.
"""
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils import logger, criar_diretorio


def gerar_shap_dependence_plot(modelo, X_test, feature_names, top_features=5):
    """
    Gera SHAP dependence plots para as top features.
    Mostra como cada feature impacta a previsão.
    """
    logger.info("Gerando SHAP dependence plots...")
    criar_diretorio('results/figures')
    
    # Calcular SHAP values
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_test)
    
    # Selecionar top features
    if hasattr(modelo, 'feature_importances_'):
        importances = modelo.feature_importances_
        top_indices = np.argsort(importances)[-top_features:][::-1]
    else:
        top_indices = list(range(min(top_features, X_test.shape[1])))
    
    # Criar figura com múltiplos subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(top_indices[:6]):
        if i >= len(axes):
            break
        
        feature_name = feature_names[idx]
        feature_values = X_test.iloc[:, idx] if hasattr(X_test, 'iloc') else X_test[:, idx]
        shap_feature = shap_values[:, idx]
        
        axes[i].scatter(feature_values, shap_feature, alpha=0.3, s=10, c=shap_values[:, idx], 
                       cmap='RdBu', edgecolors='none')
        axes[i].set_xlabel(feature_name, fontsize=10)
        axes[i].set_ylabel('SHAP Value', fontsize=10)
        axes[i].set_title(f'Impacto de {feature_name}', fontsize=11, fontweight='bold')
        axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/shap_dependence_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("✅ SHAP dependence plots salvos!")


def gerar_shap_interaction_values(modelo, X_test, feature_names, num_samples=100):
    """
    Gera matriz de interação SHAP entre features.
    Mostra quais features interagem entre si.
    """
    logger.info("Gerando SHAP interaction values...")
    criar_diretorio('results/figures')
    
    # Usar subset para performance
    X_sample = X_test.iloc[:num_samples] if hasattr(X_test, 'iloc') else X_test[:num_samples]
    
    explainer = shap.TreeExplainer(modelo)
    shap_interaction = explainer.shap_interaction_values(X_sample)
    
    # Calcular magnitude média das interações
    mean_abs_interaction = np.abs(shap_interaction).mean(axis=0)
    
    # Selecionar top 10 features
    if len(feature_names) > 10:
        importances = modelo.feature_importances_ if hasattr(modelo, 'feature_importances_') else np.ones(len(feature_names))
        top_indices = np.argsort(importances)[-10:][::-1]
    else:
        top_indices = list(range(len(feature_names)))
    
    # Matriz de interação
    interaction_matrix = mean_abs_interaction[np.ix_(top_indices, top_indices)]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(interaction_matrix, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(range(len(top_indices)))
    ax.set_yticks(range(len(top_indices)))
    ax.set_xticklabels([feature_names[i][:15] for i in top_indices], rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels([feature_names[i][:15] for i in top_indices], fontsize=9)
    
    ax.set_title('Matriz de Interação SHAP (Top 10 Features)', fontsize=14, fontweight='bold', pad=20)
    plt.colorbar(im, ax=ax, label='Magnitude da Interação')
    
    plt.tight_layout()
    plt.savefig('results/figures/shap_interaction_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("✅ Matriz de interação SHAP salva!")


def gerar_shap_waterfall_plot(modelo, X_test, feature_names, sample_idx=0):
    """
    Gera waterfall plot para uma previsão específica.
    Mostra contribuição de cada feature para uma decisão.
    """
    logger.info(f"Gerando SHAP waterfall plot para amostra {sample_idx}...")
    criar_diretorio('results/figures')
    
    explainer = shap.TreeExplainer(modelo)
    
    # Selecionar amostra
    if hasattr(X_test, 'iloc'):
        sample = X_test.iloc[sample_idx:sample_idx+1]
    else:
        sample = X_test[sample_idx:sample_idx+1]
    
    shap_values = explainer.shap_values(sample)
    
    # Criar waterfall plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Ordenar features por magnitude
    if hasattr(modelo, 'feature_importances_'):
        importances = np.abs(shap_values[0])
        sorted_indices = np.argsort(importances)[::-1][:15]
    else:
        sorted_indices = list(range(min(15, len(feature_names))))
    
    # Plotar barras
    y_pos = np.arange(len(sorted_indices))
    values = shap_values[0][sorted_indices]
    feature_labels = [feature_names[i][:20] for i in sorted_indices]
    
    colors = ['#00ff88' if v > 0 else '#ff6b6b' for v in values]
    
    ax.barh(y_pos, values, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_labels, fontsize=9)
    ax.set_xlabel('Contribuição SHAP', fontsize=12, fontweight='bold')
    ax.set_title(f'Waterfall Plot - Amostra {sample_idx}', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('results/figures/shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("✅ Waterfall plot salvo!")


def gerar_todos_shap_avancados(modelo, X_test, feature_names):
    """
    Gera todos os gráficos SHAP avançados.
    """
    logger.info("\n" + "="*80)
    logger.info("🔬 GERANDO SHAP AVANÇADO")
    logger.info("="*80)
    
    try:
        gerar_shap_dependence_plot(modelo, X_test, feature_names, top_features=6)
        gerar_shap_interaction_values(modelo, X_test, feature_names, num_samples=100)
        gerar_shap_waterfall_plot(modelo, X_test, feature_names, sample_idx=0)
        
        logger.info("✅ Todos os gráficos SHAP avançados gerados!")
        
    except Exception as e:
        logger.error(f"Erro ao gerar SHAP avançado: {e}")
        import traceback
        traceback.print_exc()