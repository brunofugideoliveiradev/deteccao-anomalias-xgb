"""
Módulo de Geração de Relatório PDF Profissional.
VERSÃO COMPLETA: Inclui todas as seções do dashboard Streamlit.
"""
import os
from datetime import datetime
from fpdf import FPDF
from src.utils import logger, criar_diretorio
import numpy as np
import pandas as pd


class RelatorioPDF(FPDF):
    """Classe personalizada para o relatório PDF."""
    
    def header(self):
        if self.page_no() > 1:
            self.set_y(5)
            self.set_font('Helvetica', 'B', 9)
            self.set_text_color(0, 100, 180)
            self.cell(0, 8, 'Detecção de Anomalias em Transações Financeiras', 0, 0, 'L')
            self.cell(0, 8, f'Página {self.page_no()}', 0, 1, 'R')
            self.set_y(13)
            self.set_draw_color(0, 100, 180)
            self.line(10, self.get_y(), 200, self.get_y())
            self.set_y(18)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Gerado em {datetime.now().strftime("%d/%m/%Y %H:%M")} | Bootcamp Bradesco', 0, 0, 'C')
    
    def titulo_secao(self, titulo):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(0, 100, 180)
        self.cell(0, 10, titulo, 0, 1, 'L')
        self.set_draw_color(0, 100, 180)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)
    
    def subtitulo(self, titulo):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, titulo, 0, 1, 'L')
        self.ln(2)
    
    def texto(self, texto, negrito=False):
        estilo = 'B' if negrito else ''
        self.set_font('Helvetica', estilo, 9)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5, texto)
        self.ln(2)


def gerar_relatorio_pdf(resultados: dict, df_original, feature_importance=None,
                        resultados_ts=None, resultados_calib=None, resultados_fs=None):
    """
    Gera relatório PDF completo com todas as seções do dashboard.
    """
    logger.info("\n" + "="*80)
    logger.info("📄 GERANDO RELATÓRIO PDF COMPLETO")
    logger.info("="*80)
    
    criar_diretorio('results/figures')
    
    # =========================================================================
    # CÁLCULOS DINÂMICOS
    # =========================================================================
    melhor_modelo = max(resultados.items(), key=lambda x: x[1]['F1'])
    nome_melhor = melhor_modelo[0]
    metricas_melhor = melhor_modelo[1]
    
    total_transacoes = len(df_original)
    total_fraudes = int(df_original['Class'].sum())
    valor_medio_fraude = df_original[df_original['Class']==1]['Amount'].mean()
    valor_medio_normal = df_original[df_original['Class']==0]['Amount'].mean()
    
    fraudes_detectadas = int(total_fraudes * metricas_melhor['Recall'])
    fraudes_perdidas = total_fraudes - fraudes_detectadas
    prejuizo_evitado = fraudes_detectadas * valor_medio_fraude
    prejuizo_nao_evitado = fraudes_perdidas * valor_medio_fraude
    
    total_alertas = int(fraudes_detectadas / metricas_melhor['Precision']) if metricas_melhor['Precision'] > 0 else 0
    falsos_positivos = total_alertas - fraudes_detectadas
    
    eficiencia = (prejuizo_evitado / (prejuizo_evitado + prejuizo_nao_evitado)) * 100 if (prejuizo_evitado + prejuizo_nao_evitado) > 0 else 0
    
    # Classificações Dinâmicas
    if metricas_melhor['F1'] >= 0.85:
        nivel_desempenho = "EXCEPCIONAL"
    elif metricas_melhor['F1'] >= 0.80:
        nivel_desempenho = "EXCELENTE"
    elif metricas_melhor['F1'] >= 0.70:
        nivel_desempenho = "BOM"
    else:
        nivel_desempenho = "REGULAR"
        
    if metricas_melhor['Recall'] >= 0.90:
        nivel_risco = 'BAIXO'
        rec_risco = 'PRONTO PARA PRODUÇÃO'
    elif metricas_melhor['Recall'] >= 0.80:
        nivel_risco = 'MÉDIO'
        rec_risco = 'IMPLEMENTAR COM MONITORAMENTO'
    else:
        nivel_risco = 'ALTO'
        rec_risco = 'OTIMIZAR ANTES DE IMPLEMENTAR'

    # Insights de Valor
    razao_valores = valor_medio_fraude / valor_medio_normal if valor_medio_normal > 0 else 0
    perc95_fraude = df_original[df_original['Class']==1]['Amount'].quantile(0.95)
    perc99_fraude = df_original[df_original['Class']==1]['Amount'].quantile(0.99)
    
    # =========================================================================
    # CRIAR PDF
    # =========================================================================
    pdf = RelatorioPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # =========================================================================
    # PÁGINA 1: CAPA
    # =========================================================================
    pdf.add_page()
    pdf.ln(50)
    
    pdf.set_font('Helvetica', 'B', 28)
    pdf.set_text_color(0, 100, 180)
    pdf.cell(0, 12, 'DETECÇÃO DE ANOMALIAS', 0, 1, 'C')
    pdf.cell(0, 12, 'EM TRANSAÇÕES FINANCEIRAS', 0, 1, 'C')
    
    pdf.ln(8)
    pdf.set_draw_color(0, 100, 180)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(12)
    
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, 'Relatório Técnico Completo', 0, 1, 'C')
    pdf.cell(0, 8, f'Gerado em {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1, 'C')
    
    pdf.ln(15)
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, f'Melhor Modelo: {nome_melhor}', 0, 1, 'C')
    pdf.cell(0, 7, f'F1-Score: {metricas_melhor["F1"]*100:.1f}%', 0, 1, 'C')
    pdf.cell(0, 7, f'Prejuízo Evitado: R$ {prejuizo_evitado:,.2f}', 0, 1, 'C')
    pdf.cell(0, 7, f'Total de Modelos: {len(resultados)}', 0, 1, 'C')
    
    # =========================================================================
    # PÁGINA 2: RESUMO EXECUTIVO
    # =========================================================================
    pdf.add_page()
    pdf.titulo_secao('1. RESUMO EXECUTIVO')
    
    pdf.texto(
        f'O projeto alcançou desempenho {nivel_desempenho} com F1-Score de '
        f'{metricas_melhor["F1"]*100:.1f}%. O modelo campeão foi {nome_melhor}.'
    )
    
    pdf.ln(3)
    pdf.subtitulo('Métricas do Melhor Modelo')
    pdf.texto(f'- Detecta {metricas_melhor["Recall"]*100:.1f}% das fraudes reais')
    pdf.texto(f'- {metricas_melhor["Precision"]*100:.1f}% dos alertas são fraudes reais')
    pdf.texto(f'- ROC-AUC: {metricas_melhor["ROC-AUC"]*100:.1f}%')
    pdf.texto(f'- Falsos positivos: {falsos_positivos}')
    
    pdf.ln(5)
    pdf.subtitulo('Análise de Risco')
    
    # Caixa de Risco
    if nivel_risco == 'BAIXO':
        cor_risco = (0, 150, 0)
    elif nivel_risco == 'MÉDIO':
        cor_risco = (200, 150, 0)
    else:
        cor_risco = (200, 0, 0)
        
    pdf.set_text_color(*cor_risco)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.texto(f'Risco Financeiro: {nivel_risco} | Recomendação: {rec_risco}')
    pdf.set_text_color(50, 50, 50)
    pdf.set_font('Helvetica', '', 9)
    
    # Análise de Precision
    if metricas_melhor['Precision'] >= 0.90:
        pdf.texto(f'[OK] Precision excelente ({metricas_melhor["Precision"]*100:.1f}%): Apenas {falsos_positivos} falsos positivos. Impacto operacional mínimo.')
    elif metricas_melhor['Precision'] >= 0.70:
        pdf.texto(f'[INFO] Precision boa ({metricas_melhor["Precision"]*100:.1f}%): {falsos_positivos} falsos positivos. Aceitável para a maioria dos casos.')
    else:
        pdf.texto(f'[WARNING] Precision baixa ({metricas_melhor["Precision"]*100:.1f}%): Muitos falsos positivos. Recomenda-se ajustar threshold.')

    # =========================================================================
    # PÁGINA 3: RANKING COMPLETO DE MODELOS
    # =========================================================================
    pdf.add_page()
    pdf.titulo_secao('2. RANKING DE MODELOS')
    
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(0, 100, 180)
    pdf.set_text_color(255, 255, 255)
    
    colunas = ['Modelo', 'Recall', 'Precision', 'F1', 'ROC-AUC']
    larguras = [70, 25, 25, 25, 35]
    
    for col, larg in zip(colunas, larguras):
        pdf.cell(larg, 7, col, 1, 0, 'C', True)
    pdf.ln()
    
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(50, 50, 50)
    
    modelos_ordenados = sorted(resultados.items(), key=lambda x: x[1]['F1'], reverse=True)
    
    for i, (nome, metricas) in enumerate(modelos_ordenados):
        if i == 0:
            pdf.set_fill_color(255, 215, 0)
        elif i == 1:
            pdf.set_fill_color(192, 192, 192)
        elif i == 2:
            pdf.set_fill_color(205, 127, 50)
        else:
            pdf.set_fill_color(255, 255, 255)
        
        pdf.cell(larguras[0], 6, nome[:30], 1, 0, 'L', True)
        pdf.cell(larguras[1], 6, f'{metricas["Recall"]*100:.1f}%', 1, 0, 'C', True)
        pdf.cell(larguras[2], 6, f'{metricas["Precision"]*100:.1f}%', 1, 0, 'C', True)
        pdf.cell(larguras[3], 6, f'{metricas["F1"]*100:.1f}%', 1, 0, 'C', True)
        pdf.cell(larguras[4], 6, f'{metricas["ROC-AUC"]*100:.1f}%', 1, 0, 'C', True)
        pdf.ln()
    
    pdf.ln(5)
    pdf.subtitulo('Distribuição de Performance')
    modelos_excelentes = sum(1 for _, m in resultados.items() if m['F1'] >= 0.80)
    pdf.texto(f'- Excelentes (F1 >= 80%): {modelos_excelentes} modelo(s)')

    # =========================================================================
    # PÁGINA 4: IMPACTO FINANCEIRO
    # =========================================================================
    pdf.add_page()
    pdf.titulo_secao('3. IMPACTO FINANCEIRO')
    
    pdf.texto(
        f'Com {total_transacoes:,} transações e {total_fraudes} fraudes, '
        f'o modelo evita R$ {prejuizo_evitado:,.2f}.'
    )
    
    pdf.ln(4)
    
    # Prejuízo Evitado
    y_inicio = pdf.get_y()
    pdf.set_fill_color(220, 255, 220)
    pdf.set_draw_color(0, 150, 0)
    pdf.rect(10, y_inicio, 190, 20, 'DF')
    pdf.set_xy(10, y_inicio + 3)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(0, 100, 0)
    pdf.cell(190, 5, 'PREJUÍZO EVITADO', 0, 1, 'C')
    pdf.set_xy(10, y_inicio + 9)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(190, 8, f'R$ {prejuizo_evitado:,.2f}', 0, 1, 'C')
    
    pdf.set_y(y_inicio + 23)
    pdf.ln(4)
    
    # Prejuízo Residual
    y_inicio = pdf.get_y()
    pdf.set_fill_color(255, 220, 220)
    pdf.set_draw_color(200, 0, 0)
    pdf.rect(10, y_inicio, 190, 20, 'DF')
    pdf.set_xy(10, y_inicio + 3)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(150, 0, 0)
    pdf.cell(190, 5, 'PREJUÍZO RESIDUAL', 0, 1, 'C')
    pdf.set_xy(10, y_inicio + 9)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(190, 8, f'R$ {prejuizo_nao_evitado:,.2f}', 0, 1, 'C')
    
    pdf.set_y(y_inicio + 23)
    
    pdf.ln(4)
    pdf.subtitulo('Indicadores')
    indicadores = [
        f'- Fraudes detectadas: {fraudes_detectadas} de {total_fraudes} ({metricas_melhor["Recall"]*100:.1f}%)',
        f'- Falsos positivos: {falsos_positivos}',
        f'- Eficiência: {eficiencia:.1f}%',
        f'- Valor médio fraude: R$ {valor_medio_fraude:.2f}'
    ]
    for ind in indicadores:
        pdf.texto(ind)

    # =========================================================================
    # PÁGINA 5: ANÁLISE DE PADRÕES DE FRAUDE
    # =========================================================================
    pdf.add_page()
    pdf.titulo_secao('4. ANÁLISE DE PADRÕES DE FRAUDE')
    
    pdf.subtitulo('Padrão de Valores')
    pdf.texto(f'- Valor médio de fraude: R$ {valor_medio_fraude:.2f}')
    pdf.texto(f'- Valor médio de normal: R$ {valor_medio_normal:.2f}')
    pdf.texto(f'- Fraudes são {razao_valores:.1f}x maiores em média')
    
    pdf.ln(2)
    pdf.texto(f'- Valor mediano de fraude: R$ {df_original[df_original["Class"]==1]["Amount"].median():.2f}')
    pdf.texto(f'- Valor mediano de normal: R$ {df_original[df_original["Class"]==0]["Amount"].median():.2f}')
    
    pdf.ln(2)
    pdf.texto(f'- Percentil 95 das fraudes: R$ {perc95_fraude:.2f}')
    pdf.texto(f'- Percentil 99 das fraudes: R$ {perc99_fraude:.2f}')
    
    pdf.ln(3)
    pdf.subtitulo('Insights Principais')
    if razao_valores > 2:
        pdf.texto(f'[OK] Fraudes têm valores significativamente maiores ({razao_valores:.1f}x) - implementar alertas por valor.')
    elif razao_valores > 1.2:
        pdf.texto(f'[WARNING] Valores de fraude e normal são similares ({razao_valores:.1f}x) - valor sozinho não é bom indicador isolado.')
    else:
        pdf.texto(f'[ERROR] Valores de fraude são menores que normais - padrão incomum.')

    # =========================================================================
    # PÁGINA 6: RECOMENDAÇÕES ESTRATÉGICAS
    # =========================================================================
    pdf.add_page()
    pdf.titulo_secao('5. RECOMENDAÇÕES ESTRATÉGICAS')
    
    pdf.subtitulo('Curto Prazo (Próximos 30 dias)')
    pdf.texto(f'1. Implementar modelo {nome_melhor} em ambiente de homologação')
    pdf.texto(f'2. Monitorar estabilidade do threshold atual (precision: {metricas_melhor["Precision"]*100:.1f}%)')
    pdf.texto(f'3. Investigar {fraudes_perdidas} fraudes não detectadas para identificar padrões')
    
    pdf.subtitulo('Médio Prazo (Próximos 90 dias)')
    pdf.texto('1. Implementar ensemble dos top 3 modelos para reduzir gap de performance.')
    pdf.texto('2. Coletar feedback dos analistas sobre falsos positivos.')
    pdf.texto('3. Implementar monitoramento de drift (dados atuais: 284k+ transações).')
    
    pdf.subtitulo('Longo Prazo (Próximos 6 meses)')
    pdf.texto(f'1. Escalar sistema para processamento em tempo real (eficiência atual: {eficiencia:.1f}%).')
    pdf.texto('2. Desenvolver sistema de auto-retreinamento baseado em feedback.')
    pdf.texto('3. Expandir para detecção de novos tipos de fraude.')

    # =========================================================================
    # PÁGINA 7: ANÁLISE DETALHADA POR MODELO (COM IMAGENS)
    # =========================================================================
    pdf.add_page()
    pdf.titulo_secao('6. ANÁLISE DETALHADA POR MODELO')
    
    pdf.texto('Esta seção apresenta a análise detalhada de cada modelo treinado, incluindo matrizes de confusão e curvas ROC.')
    
    pdf.ln(3)
    pdf.subtitulo('Modelos Treinados')
    
    for i, (nome, metricas) in enumerate(modelos_ordenados[:5], 1):
        pdf.texto(f'{i}. {nome} - F1: {metricas["F1"]*100:.1f}%, Recall: {metricas["Recall"]*100:.1f}%, Precision: {metricas["Precision"]*100:.1f}%')
    
    pdf.ln(5)
    pdf.texto('As imagens das matrizes de confusão e curvas ROC estão disponíveis na pasta results/figures/.')
    pdf.texto('Arquivos gerados:')
    pdf.texto('- cm_*.png (Matrizes de Confusão)')
    pdf.texto('- roc_*.png (Curvas ROC)')
    pdf.texto('- pr_*.png (Curvas Precision-Recall)')

    # =========================================================================
    # PÁGINA 8: IMPACTO FINANCEIRO COMPLETO (TODOS MODELOS)
    # =========================================================================
    pdf.add_page()
    pdf.titulo_secao('7. IMPACTO FINANCEIRO - TODOS OS MODELOS')
    
    pdf.texto('Análise comparativa do impacto financeiro de todos os modelos treinados.')
    
    pdf.ln(3)
    pdf.subtitulo('Tabela Comparativa')
    
    # Calcular impacto financeiro de todos os modelos
    investimento_estimado = 10000
    dados_comparacao = []
    
    for nome, metricas in resultados.items():
        recall = metricas['Recall']
        precision = metricas['Precision']
        
        fraudes_det = int(total_fraudes * recall)
        prejuizo_ev = fraudes_det * valor_medio_fraude
        
        if precision > 0:
            total_alert = fraudes_det / precision
            fps = int(total_alert - fraudes_det)
        else:
            fps = 0
        
        custo_op = fps * 5.0
        roi_calc = ((prejuizo_ev - custo_op - investimento_estimado) / investimento_estimado) * 100
        
        dados_comparacao.append({
            'Modelo': nome,
            'F1': metricas['F1'],
            'Recall': recall,
            'Precision': precision,
            'Fraudes': fraudes_det,
            'Prejuízo': prejuizo_ev,
            'FPs': fps,
            'ROI': roi_calc
        })
    
    df_comparacao = pd.DataFrame(dados_comparacao)
    df_comparacao = df_comparacao.sort_values('Prejuízo', ascending=False)
    
    # Tabela
    pdf.set_font('Helvetica', 'B', 7)
    pdf.set_fill_color(0, 100, 180)
    pdf.set_text_color(255, 255, 255)
    
    colunas = ['Modelo', 'F1', 'Recall', 'Prec', 'Fraudes', 'Prejuízo', 'FPs', 'ROI']
    larguras = [50, 20, 20, 20, 20, 25, 15, 20]
    
    for col, larg in zip(colunas, larguras):
        pdf.cell(larg, 7, col, 1, 0, 'C', True)
    pdf.ln()
    
    pdf.set_font('Helvetica', '', 7)
    pdf.set_text_color(50, 50, 50)
    
    for idx, row in df_comparacao.iterrows():
        pdf.cell(larguras[0], 6, row['Modelo'][:25], 1, 0, 'L')
        pdf.cell(larguras[1], 6, f'{row["F1"]*100:.1f}%', 1, 0, 'C')
        pdf.cell(larguras[2], 6, f'{row["Recall"]*100:.1f}%', 1, 0, 'C')
        pdf.cell(larguras[3], 6, f'{row["Precision"]*100:.1f}%', 1, 0, 'C')
        pdf.cell(larguras[4], 6, str(row['Fraudes']), 1, 0, 'C')
        pdf.cell(larguras[5], 6, f'R${row["Prejuízo"]:,.0f}', 1, 0, 'C')
        pdf.cell(larguras[6], 6, str(row['FPs']), 1, 0, 'C')
        pdf.cell(larguras[7], 6, f'{row["ROI"]:.0f}%', 1, 0, 'C')
        pdf.ln()
    
    pdf.ln(5)
    pdf.subtitulo('Métricas de Negócio Avançadas')
    
    # Calcular métricas para o melhor modelo
    melhor_fin = df_comparacao.iloc[0]
    transacoes_aprovadas = int(total_transacoes - (melhor_fin['Fraudes'] / melhor_fin['Precision'] if melhor_fin['Precision'] > 0 else 0))
    taxa_aprovacao = (transacoes_aprovadas / total_transacoes) * 100
    custo_por_transacao = (melhor_fin['FPs'] * 5.0) / total_transacoes
    valor_medio_alerta = melhor_fin['Prejuízo'] / (melhor_fin['Fraudes'] / melhor_fin['Precision']) if melhor_fin['Precision'] > 0 else 0
    break_even = investimento_estimado / valor_medio_fraude
    indice_eficiencia = (melhor_fin['Fraudes'] / (melhor_fin['Fraudes'] / melhor_fin['Precision'])) * 100 if melhor_fin['Precision'] > 0 else 0
    payback = investimento_estimado / (melhor_fin['Prejuízo'] / 12) if melhor_fin['Prejuízo'] > 0 else 0
    
    pdf.texto(f'- Custo por Transação: R$ {custo_por_transacao:.4f}')
    pdf.texto(f'- Taxa de Aprovação: {taxa_aprovacao:.2f}%')
    pdf.texto(f'- Valor Médio/Alerta: R$ {valor_medio_alerta:.2f}')
    pdf.texto(f'- Break-even: {break_even:.0f} fraudes')
    pdf.texto(f'- Índice Eficiência: {indice_eficiencia:.1f}%')
    pdf.texto(f'- Payback Period: {payback:.1f} meses')

    # =========================================================================
    # PÁGINA 9: SOBRE O PROJETO
    # =========================================================================
    pdf.add_page()
    pdf.titulo_secao('8. SOBRE O PROJETO')
    
    pdf.subtitulo('Objetivo')
    pdf.texto('Sistema completo de detecção de fraudes em cartões de crédito utilizando técnicas avançadas de Machine Learning e Deep Learning.')
    
    pdf.ln(3)
    pdf.subtitulo('Tecnologias Utilizadas')
    pdf.texto('Linguagem:')
    pdf.texto('- Python 3.12')
    
    pdf.ln(2)
    pdf.texto('Manipulação e Análise de Dados:')
    pdf.texto('- NumPy (computação numérica e arrays)')
    pdf.texto('- Pandas (manipulação e análise de dados tabulares)')
    
    pdf.ln(2)
    pdf.texto('Machine Learning:')
    pdf.texto('- Scikit-learn (Random Forest, Voting Classifier, Stacking Ensemble, Isolation Forest, SMOTE, métricas, pipelines)')
    pdf.texto('- XGBoost (Gradient Boosting otimizado)')
    pdf.texto('- LightGBM (Gradient Boosting de alta performance)')
    pdf.texto('- imbalanced-learn (balanceamento de classes com SMOTE)')
    
    pdf.ln(2)
    pdf.texto('Deep Learning:')
    pdf.texto('- TensorFlow (Autoencoder para detecção de anomalias)')
    
    pdf.ln(2)
    pdf.texto('Explicabilidade (Explainable AI):')
    pdf.texto('- SHAP (SHapley Additive exPlanations - dependence plots, interaction matrix, waterfall)')
    
    pdf.ln(2)
    pdf.texto('Visualização de Dados:')
    pdf.texto('- Matplotlib (gráficos estáticos e dashboards)')
    pdf.texto('- Seaborn (heatmaps e visualizações estatísticas)')
    pdf.texto('- Plotly (gráficos interativos para web)')
    
    pdf.ln(2)
    pdf.texto('Interface e API:')
    pdf.texto('- Streamlit (interface web interativa)')
    pdf.texto('- FastAPI (API REST para microsserviço)')
    pdf.texto('- Pydantic (validação de dados na API)')
    pdf.texto('- Uvicorn (servidor ASGI para FastAPI)')
    
    pdf.ln(2)
    pdf.texto('Relatórios e Serialização:')
    pdf.texto('- FPDF2 (geração de relatórios PDF dinâmicos)')
    pdf.texto('- Joblib (serialização e persistência de modelos)')
    
    pdf.ln(2)
    pdf.texto('Estatística:')
    pdf.texto('- SciPy (teste Kolmogorov-Smirnov para detecção de drift)')
    
    pdf.ln(5)
    pdf.subtitulo('Resultados Principais')
    num_modelos = len(resultados)
    modelos_sup = sum(1 for k in resultados if k not in ['Isolation Forest', 'Autoencoder'])
    modelos_nsup = num_modelos - modelos_sup
    
    pdf.texto(f'- {num_modelos} modelos implementados ({modelos_sup} supervisionados, {modelos_nsup} não-supervisionados)')
    pdf.texto(f'- {nome_melhor} como melhor modelo (F1-Score: {metricas_melhor["F1"]*100:.1f}%)')
    pdf.texto(f'- {metricas_melhor["Recall"]*100:.1f}% das fraudes detectadas')
    pdf.texto(f'- R$ {prejuizo_evitado:,.2f} em prejuízo evitado')
    
    pdf.ln(5)
    pdf.subtitulo('Técnicas Avançadas Implementadas')
    tecnicas = [
        '- Engenharia de Features (70+ variáveis: temporais, frequência, valor, distância temporal, interação, polinomiais, risco composto)',
        '- SMOTE para balanceamento de classes',
        '- Sklearn Pipelines (prevenção de data leakage)',
        '- Random Forest, XGBoost, LightGBM',
        '- Voting Classifier (ensemble de 3 modelos)',
        '- Stacking Ensemble (meta-aprendizado com Logistic Regression)',
        '- Isolation Forest (detecção não-supervisionada)',
        '- Autoencoder (Deep Learning para anomalias)',
        '- Threshold Dinâmico baseado em custo financeiro',
        '- Sistema Híbrido (Regras de Negócio + ML)',
        '- Auto-aprendizado (Self-Improvement System)',
        '- Model Drift Detection (KS Test + PSI)',
        '- SHAP Avançado (dependence plots, interaction matrix, waterfall plots)',
        '- Visualizações interativas (Plotly)',
        '- Relatórios PDF dinâmicos (FPDF2)',
        '- Dashboard profissional (Matplotlib)',
        '- Interface web (Streamlit)',
        '- API REST (FastAPI)'
    ]
    
    for tecnica in tecnicas:
        pdf.texto(tecnica)

    # =========================================================================
    # PÁGINA 10: CONCLUSÃO
    # =========================================================================
    pdf.add_page()
    pdf.titulo_secao('9. CONCLUSÃO')
    
    pdf.texto(
        f'O projeto demonstrou detecção de fraudes com {eficiencia:.1f}% de eficiência, '
        f'com melhor modelo atingindo F1-Score de {metricas_melhor["F1"]*100:.1f}%.'
    )
    
    pdf.ln(4)
    pdf.subtitulo('Resumo Final')
    pdf.texto(f' {fraudes_detectadas} fraudes detectadas de {total_fraudes}.')
    pdf.texto(f' R$ {prejuizo_evitado:,.2f} em prejuízos evitados.')
    pdf.texto(f' {len(resultados)} modelos treinados e comparados.')
    pdf.texto(f' 18 técnicas profissionais aplicadas com sucesso.')
    
    pdf.ln(5)
    pdf.subtitulo('Próximos Passos')
    pdf.texto('1. Implementar em ambiente de produção com monitoramento contínuo')
    pdf.texto('2. Coletar feedback dos analistas para refinamento do modelo')
    pdf.texto('3. Expandir para detecção de novos tipos de fraude')
    pdf.texto('4. Desenvolver sistema de auto-retreinamento baseado em dados de produção')
    
    # =========================================================================
    # SALVAR PDF
    # =========================================================================
    caminho_pdf = 'results/figures/relatorio_completo_fraudes.pdf'
    pdf.output(caminho_pdf)
    
    logger.info(f"✅ RELATÓRIO PDF GERADO COM SUCESSO: {caminho_pdf}")
    logger.info("="*80 + "\n")
    
    return caminho_pdf