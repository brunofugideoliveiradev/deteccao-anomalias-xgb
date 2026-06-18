"""
Visualizações Interativas com Plotly.
Gera gráficos dinâmicos e interativos para dashboards web.
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score
from src.utils import logger, criar_diretorio


def criar_grafico_comparacao_modelos_interativo(resultados: dict):
    """
    Cria gráfico de barras interativo comparando todos os modelos.
    """
    logger.info("Criando gráfico comparativo interativo...")
    
    df = pd.DataFrame(resultados).T
    df = df.reset_index().rename(columns={'index': 'Modelo'})
    
    fig = go.Figure()
    
    # Barras para F1-Score
    fig.add_trace(go.Bar(
        x=df['Modelo'],
        y=df['F1'] * 100,
        name='F1-Score',
        marker_color='#00ff88',
        text=df['F1'] * 100,
        texttemplate='%{text:.1f}%',
        textposition='auto'
    ))
    
    # Barras para Recall
    fig.add_trace(go.Bar(
        x=df['Modelo'],
        y=df['Recall'] * 100,
        name='Recall',
        marker_color='#6bcbff',
        text=df['Recall'] * 100,
        texttemplate='%{text:.1f}%',
        textposition='auto'
    ))
    
    # Barras para Precision
    fig.add_trace(go.Bar(
        x=df['Modelo'],
        y=df['Precision'] * 100,
        name='Precision',
        marker_color='#ffd93d',
        text=df['Precision'] * 100,
        texttemplate='%{text:.1f}%',
        textposition='auto'
    ))
    
    fig.update_layout(
        title='📊 Comparação de Modelos - Métricas de Performance',
        xaxis_title='Modelo',
        yaxis_title='Score (%)',
        barmode='group',
        height=600,
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    criar_diretorio('results/figures')
    fig.write_html('results/figures/comparacao_modelos_interativa.html')
    fig.write_image('results/figures/comparacao_modelos_interativa.png', width=1200, height=600)
    
    logger.info("Gráfico comparativo salvo!")
    return fig


def criar_curva_roc_interativa(y_true, y_prob, nome_modelo):
    """
    Cria curva ROC interativa.
    """
    logger.info(f"Criando curva ROC interativa para {nome_modelo}...")
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'{nome_modelo} (AUC = {roc_auc:.4f})',
        line=dict(color='#00ff88', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 136, 0.2)'
    ))
    
    # Linha diagonal (aleatório)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Aleatório',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'📈 Curva ROC - {nome_modelo}',
        xaxis_title='Taxa de Falsos Positivos',
        yaxis_title='Taxa de Verdadeiros Positivos',
        height=600,
        template='plotly_dark',
        showlegend=True
    )
    
    criar_diretorio('results/figures')
    fig.write_html(f'results/figures/roc_{nome_modelo.replace(" ", "_")}_interativa.html')
    
    logger.info("Curva ROC salva!")
    return fig


def criar_matriz_confusao_interativa(y_true, y_pred, nome_modelo):
    """
    Cria matriz de confusão interativa com heatmap.
    """
    logger.info(f"Criando matriz de confusão interativa para {nome_modelo}...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Normal (0)', 'Fraude (1)'],
        y=['Normal (0)', 'Fraude (1)'],
        colorscale='Blues',
        showscale=True,
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20}
    ))
    
    fig.update_layout(
        title=f'🎯 Matriz de Confusão - {nome_modelo}',
        xaxis_title='Previsto',
        yaxis_title='Real',
        height=600,
        template='plotly_dark'
    )
    
    criar_diretorio('results/figures')
    fig.write_html(f'results/figures/matriz_confusao_{nome_modelo.replace(" ", "_")}_interativa.html')
    
    logger.info("Matriz de confusão salva!")
    return fig


def criar_dashboard_interativo_completo(resultados, df_original, modelo_melhor=None, X_test=None, y_test=None):
    """
    Cria dashboard completo com múltiplos gráficos interativos.
    """
    logger.info("Criando dashboard interativo completo...")
    criar_diretorio('results/figures')
    
    # Gráfico 1: Comparação de Modelos
    fig_comparacao = criar_grafico_comparacao_modelos_interativo(resultados)
    
    # Gráfico 2: Distribuição de Valores
    fig_distribuicao = px.histogram(
        df_original,
        x='Amount',
        color='Class',
        nbins=100,
        title='📊 Distribuição de Valores - Normais vs Fraudes',
        labels={'Amount': 'Valor da Transação (R$)', 'Class': 'Classe'},
        color_discrete_map={0: '#00ff88', 1: '#ff6b6b'},
        template='plotly_dark',
        opacity=0.7
    )
    fig_distribuicao.update_layout(height=600)
    fig_distribuicao.write_html('results/figures/distribuicao_valores_interativa.html')
    
    # Gráfico 3: Fraudes por Hora
    if 'Hour' in df_original.columns:
        fraudes_por_hora = df_original[df_original['Class'] == 1]['Hour'].value_counts().sort_index()
        
        fig_hora = px.bar(
            x=fraudes_por_hora.index,
            y=fraudes_por_hora.values,
            title='⏰ Fraudes por Hora do Dia',
            labels={'x': 'Hora', 'y': 'Quantidade de Fraudes'},
            color=fraudes_por_hora.values,
            color_continuous_scale='Reds',
            template='plotly_dark'
        )
        fig_hora.update_layout(height=600)
        fig_hora.write_html('results/figures/fraudes_por_hora_interativa.html')
    
    logger.info("Dashboard interativo completo salvo!")
    return fig_comparacao