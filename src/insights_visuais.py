"""
Geração de Cards de Insights Visuais Individuais.
Estilo profissional e moderno.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import pandas as pd
import numpy as np
from src.utils import logger, criar_diretorio


def criar_cards_insights(resultados: dict, df_original: pd.DataFrame):
    """
    Cria cards individuais de insights profissionais.
    """
    logger.info("\n" + "="*80)
    logger.info(" GERANDO CARDS DE INSIGHTS VISUAIS")
    logger.info("="*80)
    
    criar_diretorio('results/figures')
    
    # Encontrar melhor modelo
    melhor_modelo = max(resultados.items(), key=lambda x: x[1]['F1'])
    nome_melhor = melhor_modelo[0]
    metricas_melhor = melhor_modelo[1]
    
    # Calcular métricas
    total_transacoes = len(df_original)
    total_fraudes = df_original['Class'].sum()
    valor_medio_fraude = df_original[df_original['Class']==1]['Amount'].mean()
    
    fraudes_detectadas = int(total_fraudes * metricas_melhor['Recall'])
    fraudes_nao_detectadas = total_fraudes - fraudes_detectadas
    prejuizo_evitado = fraudes_detectadas * valor_medio_fraude
    prejuizo_nao_evitado = fraudes_nao_detectadas * valor_medio_fraude
    
    total_alertas = int(fraudes_detectadas / metricas_melhor['Precision'])
    falsos_positivos = total_alertas - fraudes_detectadas
    
    eficiencia = (prejuizo_evitado / (prejuizo_evitado + prejuizo_nao_evitado)) * 100
    
    # =========================================================================
    # CARD 1: PREJUÍZO EVITADO
    # =========================================================================
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8), facecolor='#0a0e27')
    ax1.set_facecolor('#1a1f3a')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    rect1 = FancyBboxPatch((0.05, 0.05), 0.90, 0.90, boxstyle="round,pad=0.02", 
                           linewidth=3, edgecolor='#00ff88', facecolor='#1a1f3a')
    ax1.add_patch(rect1)
    
    ax1.text(0.5, 0.85, 'PREJUIZO EVITADO', ha='center', va='top', 
             fontsize=16, fontweight='bold', color='#00ff88', family='monospace')
    
    ax1.text(0.5, 0.60, f'R$ {prejuizo_evitado:,.2f}', ha='center', va='top', 
             fontsize=42, fontweight='bold', color='#00ff88')
    
    ax1.text(0.5, 0.40, f'Fraudes detectadas: {fraudes_detectadas:,}', 
             ha='center', va='top', fontsize=12, color='#cccccc')
    
    ax1.text(0.5, 0.32, f'Taxa de detecção: {metricas_melhor["Recall"]*100:.1f}%', 
             ha='center', va='top', fontsize=12, color='#cccccc')
    
    plt.savefig('results/figures/insight_01_prejuizo_evitado.png', 
                dpi=300, facecolor='#0a0e27', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # CARD 2: PREJUÍZO NÃO EVITADO
    # =========================================================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8), facecolor='#0a0e27')
    ax2.set_facecolor('#1a1f3a')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    rect2 = FancyBboxPatch((0.05, 0.05), 0.90, 0.90, boxstyle="round,pad=0.02", 
                           linewidth=3, edgecolor='#ff6b6b', facecolor='#1a1f3a')
    ax2.add_patch(rect2)
    
    ax2.text(0.5, 0.85, 'PREJUIZO NAO EVITADO', ha='center', va='top', 
             fontsize=16, fontweight='bold', color='#ff6b6b', family='monospace')
    
    ax2.text(0.5, 0.60, f'R$ {prejuizo_nao_evitado:,.2f}', ha='center', va='top', 
             fontsize=42, fontweight='bold', color='#ff6b6b')
    
    ax2.text(0.5, 0.40, f'Fraudes não detectadas: {fraudes_nao_detectadas:,}', 
             ha='center', va='top', fontsize=12, color='#cccccc')
    
    ax2.text(0.5, 0.32, f'Taxa de erro: {(1-metricas_melhor["Recall"])*100:.1f}%', 
             ha='center', va='top', fontsize=12, color='#cccccc')
    
    plt.savefig('results/figures/insight_02_prejuizo_nao_evitado.png', 
                dpi=300, facecolor='#0a0e27', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # CARD 3: FALSOS POSITIVOS
    # =========================================================================
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8), facecolor='#0a0e27')
    ax3.set_facecolor('#1a1f3a')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    rect3 = FancyBboxPatch((0.05, 0.05), 0.90, 0.90, boxstyle="round,pad=0.02", 
                           linewidth=3, edgecolor='#ffd93d', facecolor='#1a1f3a')
    ax3.add_patch(rect3)
    
    ax3.text(0.5, 0.85, 'FALSOS POSITIVOS', ha='center', va='top', 
             fontsize=16, fontweight='bold', color='#ffd93d', family='monospace')
    
    ax3.text(0.5, 0.60, f'{falsos_positivos:,}', ha='center', va='top', 
             fontsize=42, fontweight='bold', color='#ffd93d')
    
    ax3.text(0.5, 0.40, f'Precisão: {metricas_melhor["Precision"]*100:.1f}%', 
             ha='center', va='top', fontsize=12, color='#cccccc')
    
    ax3.text(0.5, 0.32, f'Total de alertas: {total_alertas:,}', 
             ha='center', va='top', fontsize=12, color='#cccccc')
    
    plt.savefig('results/figures/insight_03_falsos_positivos.png', 
                dpi=300, facecolor='#0a0e27', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # CARD 4: EFICIÊNCIA DO MODELO
    # =========================================================================
    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 8), facecolor='#0a0e27')
    ax4.set_facecolor('#1a1f3a')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    rect4 = FancyBboxPatch((0.05, 0.05), 0.90, 0.90, boxstyle="round,pad=0.02", 
                           linewidth=3, edgecolor='#6bcbff', facecolor='#1a1f3a')
    ax4.add_patch(rect4)
    
    ax4.text(0.5, 0.85, 'EFICIENCIA DO MODELO', ha='center', va='top', 
             fontsize=16, fontweight='bold', color='#6bcbff', family='monospace')
    
    ax4.text(0.5, 0.60, f'{eficiencia:.1f}%', ha='center', va='top', 
             fontsize=42, fontweight='bold', color='#6bcbff')
    
    ax4.text(0.5, 0.42, 'de eficiência', ha='center', va='top', 
             fontsize=12, color='#cccccc')
    
    ax4.text(0.5, 0.32, f'F1-Score: {metricas_melhor["F1"]*100:.1f}%', 
             ha='center', va='top', fontsize=12, color='#cccccc')
    
    ax4.text(0.5, 0.22, f'Modelo: {nome_melhor}', 
             ha='center', va='top', fontsize=11, color='#888888')
    
    plt.savefig('results/figures/insight_04_eficiencia.png', 
                dpi=300, facecolor='#0a0e27', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # CARD 5: RESUMO EXECUTIVO COMPLETO
    # =========================================================================
    fig5 = plt.figure(figsize=(16, 20), facecolor='#0a0e27')
    
    # Título
    fig5.text(0.5, 0.98, 'RESUMO EXECUTIVO - DETECÇÃO DE FRAUDES', 
              ha='center', va='top', fontsize=20, fontweight='bold', 
              color='#00ff88', family='monospace')
    
    # Melhor Modelo
    ax5_1 = fig5.add_axes([0.05, 0.82, 0.42, 0.14])
    ax5_1.set_facecolor('#1a1f3a')
    ax5_1.set_xlim(0, 1)
    ax5_1.set_ylim(0, 1)
    ax5_1.axis('off')
    
    rect5_1 = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.01", 
                              linewidth=2, edgecolor='#00ff88', facecolor='#1a1f3a')
    ax5_1.add_patch(rect5_1)
    
    ax5_1.text(0.5, 0.90, 'MELHOR MODELO', ha='center', va='top', 
               fontsize=14, fontweight='bold', color='#00ff88', family='monospace')
    ax5_1.text(0.5, 0.75, nome_melhor.upper(), ha='center', va='top', 
               fontsize=20, fontweight='bold', color='#ffffff')
    
    metrics = [
        f"F1-Score: {metricas_melhor['F1']*100:.1f}%",
        f"Recall: {metricas_melhor['Recall']*100:.1f}%",
        f"Precision: {metricas_melhor['Precision']*100:.1f}%",
        f"ROC-AUC: {metricas_melhor['ROC-AUC']*100:.1f}%"
    ]
    
    for i, text in enumerate(metrics):
        ax5_1.text(0.5, 0.60 - i*0.11, text, ha='center', va='top', 
                   fontsize=11, color='#cccccc', family='monospace')
    
    # Dataset
    ax5_2 = fig5.add_axes([0.53, 0.82, 0.42, 0.14])
    ax5_2.set_facecolor('#1a1f3a')
    ax5_2.set_xlim(0, 1)
    ax5_2.set_ylim(0, 1)
    ax5_2.axis('off')
    
    rect5_2 = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.01", 
                              linewidth=2, edgecolor='#ff6b6b', facecolor='#1a1f3a')
    ax5_2.add_patch(rect5_2)
    
    ax5_2.text(0.5, 0.90, 'DATASET', ha='center', va='top', 
               fontsize=14, fontweight='bold', color='#ff6b6b', family='monospace')
    
    dataset_info = [
        f"Total de Transações: {total_transacoes:,}",
        f"Total de Fraudes: {total_fraudes:,}",
        f"Taxa de Fraude: {total_fraudes/total_transacoes*100:.3f}%",
        f"Problema: Classe extremamente",
        f"desbalanceada"
    ]
    
    for i, text in enumerate(dataset_info):
        ax5_2.text(0.5, 0.72 - i*0.11, text, ha='center', va='top', 
                   fontsize=11, color='#cccccc')
    
    # Impacto Financeiro
    ax5_3 = fig5.add_axes([0.05, 0.64, 0.90, 0.14])
    ax5_3.set_facecolor('#1a1f3a')
    ax5_3.set_xlim(0, 1)
    ax5_3.set_ylim(0, 1)
    ax5_3.axis('off')
    
    rect5_3 = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.01", 
                              linewidth=2, edgecolor='#ffd93d', facecolor='#1a1f3a')
    ax5_3.add_patch(rect5_3)
    
    ax5_3.text(0.5, 0.90, 'IMPACTO FINANCEIRO ESTIMADO', ha='center', va='top', 
               fontsize=14, fontweight='bold', color='#ffd93d', family='monospace')
    
    ax5_3.text(0.5, 0.72, f'Prejuízo Evitado:', ha='center', va='top', 
               fontsize=12, color='#888888')
    ax5_3.text(0.5, 0.52, f'R$ {prejuizo_evitado:,.2f}', ha='center', va='top', 
               fontsize=32, fontweight='bold', color='#00ff88')
    
    ax5_3.text(0.5, 0.35, f'Fraudes Detectadas: {fraudes_detectadas:,} de {total_fraudes:,}', 
               ha='center', va='top', fontsize=11, color='#cccccc')
    
    # Ranking de Modelos
    ax5_4 = fig5.add_axes([0.05, 0.05, 0.90, 0.55])
    ax5_4.set_facecolor('#1a1f3a')
    ax5_4.set_xlim(0, 1)
    ax5_4.set_ylim(0, 1)
    ax5_4.axis('off')
    
    rect5_4 = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.01", 
                              linewidth=2, edgecolor='#6bcbff', facecolor='#1a1f3a')
    ax5_4.add_patch(rect5_4)
    
    ax5_4.text(0.5, 0.96, 'RANKING DE MODELOS (por F1-Score)', ha='center', va='top', 
               fontsize=14, fontweight='bold', color='#6bcbff', family='monospace')
    
    modelos_ordenados = sorted(resultados.items(), key=lambda x: x[1]['F1'], reverse=True)
    
    # Cabeçalho
    ax5_4.text(0.05, 0.88, 'Pos', ha='left', va='top', fontsize=10, fontweight='bold', color='#888888')
    ax5_4.text(0.15, 0.88, 'Modelo', ha='left', va='top', fontsize=10, fontweight='bold', color='#888888')
    ax5_4.text(0.50, 0.88, 'F1-Score', ha='center', va='top', fontsize=10, fontweight='bold', color='#888888')
    ax5_4.text(0.70, 0.88, 'Recall', ha='center', va='top', fontsize=10, fontweight='bold', color='#888888')
    ax5_4.text(0.87, 0.88, 'Precision', ha='center', va='top', fontsize=10, fontweight='bold', color='#888888')
    
    for i, (nome, metricas) in enumerate(modelos_ordenados[:6]):
        y = 0.78 - i*0.12
        
        medalha = '1o' if i == 0 else '2o' if i == 1 else '3o' if i == 2 else f'{i+1}o'
        ax5_4.text(0.05, y, medalha, ha='left', va='top', fontsize=11, color='#ffffff')
        
        cor = '#00ff88' if i == 0 else '#ffffff'
        ax5_4.text(0.15, y, nome, ha='left', va='top', fontsize=10, color=cor, 
                   fontweight='bold' if i==0 else 'normal')
        
        ax5_4.text(0.50, y, f'{metricas["F1"]*100:.1f}%', ha='center', va='top', 
                   fontsize=10, color='#cccccc')
        ax5_4.text(0.70, y, f'{metricas["Recall"]*100:.1f}%', ha='center', va='top', 
                   fontsize=10, color='#cccccc')
        ax5_4.text(0.87, y, f'{metricas["Precision"]*100:.1f}%', ha='center', va='top', 
                   fontsize=10, color='#cccccc')
        
        # Barra de progresso
        bar_width = metricas['F1'] * 0.30
        bar = Rectangle((0.50 - bar_width/2, y-0.04), bar_width, 0.03, 
                       facecolor='#00ff88' if i==0 else '#6bcbff', alpha=0.6)
        ax5_4.add_patch(bar)
    
    plt.savefig('results/figures/insight_05_resumo_executivo.png', 
                dpi=300, facecolor='#0a0e27', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    logger.info(" Cards de insights visuais gerados com sucesso!")
    logger.info("="*80 + "\n")