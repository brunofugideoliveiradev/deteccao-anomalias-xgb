"""
Aplicação Streamlit para Demo Interativa do Sistema de Detecção de Fraudes.
VERSÃO FINAL CORRIGIDA: Métricas dinâmicas e formatação adequada.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="Detecção de Fraudes - Demo Real",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #00ff88;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #1a1f3a 0%, #2d3561 100%);
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    border-left: 5px solid #00ff88;
}
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-header">🔍 DETECÇÃO DE FRAUDES EM CARTÕES DE CRÉDITO</div>', 
            unsafe_allow_html=True)
st.markdown("---")

# =========================================================================
# =========================================================================
# CARREGAMENTO DE MODELO E DADOS REAIS
# =========================================================================
@st.cache_resource
def carregar_modelo():
    """Carrega o modelo real treinado"""
    modelo_path = "models/pipeline_Random_Forest.pkl"
    if os.path.exists(modelo_path):
        return joblib.load(modelo_path)
    return None

@st.cache_resource
def carregar_dados_reais():
    """Carrega os dados gerados pelo main.py"""
    dados_path = "results/dados_streamlit.pkl"
    if os.path.exists(dados_path):
        return joblib.load(dados_path)
    return None

modelo_real = carregar_modelo()
dados_reais = carregar_dados_reais()

# =========================================================================
# PREPARAR DADOS E SELETOR DE MODELOS
# =========================================================================
if dados_reais:
    resultados = dados_reais['resultados']
    melhor_modelo_nome = dados_reais['melhor_modelo']
    metricas_melhor = dados_reais['metricas_melhor']
    threshold_otimo = dados_reais.get('threshold_otimo', 0.5)
    nome_melhor = melhor_modelo_nome
    # Criar DataFrame ordenado por F1-Score
    df_ranking = pd.DataFrame(resultados).T.reset_index()
    df_ranking.columns = ['Modelo', 'F1', 'Recall', 'Precision', 'PR-AUC', 'ROC-AUC']
    df_ranking = df_ranking.sort_values('F1', ascending=False).reset_index(drop=True)
    
    # Seletor de modelo na sidebar
    with st.sidebar:
        st.image("https://assets.dio.me/fT2lQGTboBmQ6jDMgtyZW0OatKM5ehNpTl4_JnzCYoc/f:webp/q:80/w:480/L3RyYWNrcy8xZTk5YzNkYS0wMTE4LTRkMTgtOTNiMC04YTQ3MmM4YTg3YTcucG5n", 
                 width='stretch')
        
        st.markdown("### 🏆 Ranking de Modelos")
        
        # Seletor de modelo
        modelo_selecionado = st.selectbox(
            "Selecione o modelo:",
            options=df_ranking['Modelo'].tolist(),
            index=0,  # Sempre começa com o melhor modelo
            help="Modelos ordenados por F1-Score"
        )
        
        # Obter métricas do modelo selecionado
        metricas_modelo = df_ranking[df_ranking['Modelo'] == modelo_selecionado].iloc[0]
        
        # Calcular métricas dinâmicas
        total_modelos = len(resultados)
        melhor_f1 = float(metricas_modelo['F1']) * 100
        recall = float(metricas_modelo['Recall'])
        
        # Valores do dataset real
        total_fraudes_dataset = 492
        valor_medio_fraude = 122.21
        
        fraudes_detectadas_num = int(total_fraudes_dataset * recall)
        prejuizo_evitado_num = fraudes_detectadas_num * valor_medio_fraude
        
        fraudes_detectadas = f"{fraudes_detectadas_num} de {total_fraudes_dataset}"
        prejuizo = f"R$ {prejuizo_evitado_num:,.2f}"
        
        # Exibir ranking completo
        st.markdown("---")
        st.markdown("### 📊 Top 5 Modelos")
        for idx, row in df_ranking.head(5).iterrows():
            emoji = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else f"{idx+1}º"
            st.text(f"{emoji} {row['Modelo'][:25]} - F1: {row['F1']:.1%}")
        
        st.markdown("---")
        st.markdown("### 📈 Métricas do Projeto")
        st.metric("Total de Modelos", total_modelos)
        st.metric("Melhor F1-Score", f"{melhor_f1:.1f}%")
        st.metric("Fraudes Detectadas", fraudes_detectadas)
        st.metric("Prejuízo Evitado", prejuizo)
        
        # Mostrar posição no ranking
        posicao = df_ranking[df_ranking['Modelo'] == modelo_selecionado].index[0] + 1
        st.info(f"📍 Modelo {posicao}º no ranking")
        
        st.markdown("---")
        st.markdown("### 🎯 Funcionalidades")
        
        opcoes = [
            "📊 Dashboard Geral", 
            "🔬 Testar Transação (REAL)", 
            "📈 Análise de Modelos",
            "💰 Impacto Financeiro",
            "📉 Monitoramento de Drift",
            "📚 Sobre o Projeto"
        ]
        
        opcao = st.radio("Escolha uma opção:", opcoes)
        
        st.markdown("---")
        st.markdown("**Desenvolvido por:** Bruno Fugi")
        st.markdown(f"**Data:** {datetime.now().strftime('%d/%m/%Y')}")
        
        # Status do modelo
        if modelo_real:
            st.success("✅ Modelo real carregado")
        else:
            st.warning("⚠️ Modelo não encontrado")
else:
    # Dados de fallback
    resultados = {
        'Random Forest': {'F1': 0.82, 'Recall': 0.837, 'Precision': 0.804, 'ROC-AUC': 0.976},
        'XGBoost': {'F1': 0.768, 'Recall': 0.878, 'Precision': 0.683, 'ROC-AUC': 0.984},
        'LightGBM': {'F1': 0.751, 'Recall': 0.847, 'Precision': 0.675, 'ROC-AUC': 0.978}
    }
    melhor_modelo_nome = "Random Forest"
    metricas_melhor = {'F1': 0.82, 'Recall': 0.837, 'Precision': 0.804}
    
    df_ranking = pd.DataFrame(resultados).T.reset_index()
    df_ranking.columns = ['Modelo', 'F1', 'Recall', 'Precision', 'ROC-AUC']
    df_ranking = df_ranking.sort_values('F1', ascending=False).reset_index(drop=True)
    
    modelo_selecionado = "Random Forest"
    metricas_modelo = df_ranking.iloc[0]
    
    total_modelos = len(resultados)
    melhor_f1 = 82.0
    fraudes_detectadas = "411 de 492"
    prejuizo = "R$ 50,228.85"
    
    with st.sidebar:
        st.image("https://assets.dio.me/fT2lQGTboBmQ6jDMgtyZW0OatKM5ehNpTl4_JnzCYoc/f:webp/q:80/w:480/L3RyYWNrcy8xZTk5YzNkYS0wMTE4LTRkMTgtOTNiMC04YTQ3MmM4YTg3YTcucG5n", 
                 width='stretch')
        
        st.warning("⚠️ Dados reais não encontrados. Execute main.py primeiro.")
        
        opcoes = [
            "📊 Dashboard Geral", 
            "🔬 Testar Transação (REAL)", 
            "📈 Análise de Modelos",
            "💰 Impacto Financeiro",
            "📉 Monitoramento de Drift",
            "📚 Sobre o Projeto"
        ]
        
        opcao = st.radio("Escolha uma opção:", opcoes)

# =========================================================================
# OPÇÃO 1: DASHBOARD GERAL (RELATÓRIO EXECUTIVO COMPLETO)
# =========================================================================
if opcao == "📊 Dashboard Geral":
    st.header("📊 Relatório Executivo - Visão Geral do Projeto")
    
    # =========================================================================
    # RESUMO EXECUTIVO
    # =========================================================================
    st.subheader("🏆 Resumo Executivo")
    
    nivel_desempenho = "EXCEPCIONAL" if metricas_melhor['F1'] >= 0.85 else \
                       "EXCELENTE" if metricas_melhor['F1'] >= 0.80 else \
                       "BOM" if metricas_melhor['F1'] >= 0.70 else "REGULAR"
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Melhor Modelo", nome_melhor[:20])
    col2.metric("Nível de Desempenho", nivel_desempenho)
    col3.metric("F1-Score", f"{metricas_melhor['F1']*100:.1f}%")
    col4.metric("ROC-AUC", f"{metricas_melhor['ROC-AUC']*100:.1f}%")
    
    st.info(f"""
    **{nome_melhor}** alcançou desempenho **{nivel_desempenho}** com F1-Score de **{metricas_melhor['F1']*100:.1f}%**.
    - Detecta **{metricas_melhor['Recall']*100:.1f}%** das fraudes reais
    - **{metricas_melhor['Precision']*100:.1f}%** dos alertas são fraudes reais
    - ROC-AUC: **{metricas_melhor['ROC-AUC']*100:.1f}%**
    """)
    
    # =========================================================================
    # RANKING COMPLETO DE MODELOS
    # =========================================================================
    st.subheader("🏆 Ranking de Modelos")
    
    df_ranking = pd.DataFrame(resultados).T.reset_index().rename(columns={'index': 'Modelo'})
    df_ranking = df_ranking.sort_values('F1', ascending=False).reset_index(drop=True)
    df_ranking.index = df_ranking.index + 1
    df_ranking.index.name = 'Pos'
    
    # Classificação visual
    def classificar_f1(f1):
        if f1 >= 0.80: return "✅ EXCELENTE"
        elif f1 >= 0.70: return "🟡 BOM"
        elif f1 >= 0.50: return "🟠 REGULAR"
        else: return "❌ INSUFICIENTE"
    
    df_ranking['Status'] = df_ranking['F1'].apply(classificar_f1)
    
    st.dataframe(
        df_ranking.style.format({
            'F1': '{:.1%}',
            'Recall': '{:.1%}',
            'Precision': '{:.1%}',
            'ROC-AUC': '{:.1%}'
        }).background_gradient(subset=['F1'], cmap='Greens'),
        width='stretch',
        height=400
    )
    
    # Estatísticas do ranking
    gap = df_ranking['F1'].iloc[0] - df_ranking['F1'].iloc[-1]
    excelentes = sum(1 for m in resultados.values() if m['F1'] >= 0.80)
    bons = sum(1 for m in resultados.values() if 0.70 <= m['F1'] < 0.80)
    regulares = sum(1 for m in resultados.values() if 0.50 <= m['F1'] < 0.70)
    insuficientes = sum(1 for m in resultados.values() if m['F1'] < 0.50)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gap Melhor-Pior", f"{gap*100:.1f} pp")
    col2.metric("Excelentes (≥80%)", excelentes)
    col3.metric("Bons (70-80%)", bons)
    col4.metric("Insuficientes (<50%)", insuficientes)
    
    # =========================================================================
    # IMPACTO FINANCEIRO
    # =========================================================================
    st.subheader("💰 Impacto Financeiro")
    
    total_transacoes = 284807
    total_fraudes = 492
    valor_medio_fraude = 122.21
    fraudes_detectadas_num = int(total_fraudes * metricas_melhor['Recall'])
    fraudes_perdidas = total_fraudes - fraudes_detectadas_num
    prejuizo_evitado_num = fraudes_detectadas_num * valor_medio_fraude
    prejuizo_nao_evitado_num = fraudes_perdidas * valor_medio_fraude
    total_alertas = int(fraudes_detectadas_num / metricas_melhor['Precision']) if metricas_melhor['Precision'] > 0 else 0
    falsos_positivos = total_alertas - fraudes_detectadas_num
    eficiencia = (prejuizo_evitado_num / (prejuizo_evitado_num + prejuizo_nao_evitado_num)) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Prejuízo Evitado", f"R$ {prejuizo_evitado_num:,.2f}")
    col2.metric("Fraudes Detectadas", f"{fraudes_detectadas_num} de {total_fraudes}")
    col3.metric("Eficiência", f"{eficiencia:.1f}%")
    
    st.info(f"""
    **Cenário Atual (Dataset):**
    - Total de transações: {total_transacoes:,}
    - Total de fraudes: {total_fraudes} (0.173%)
    - Valor médio por fraude: R$ {valor_medio_fraude:.2f}
    - Prejuízo total potencial: R$ {total_fraudes * valor_medio_fraude:,.2f}
    
    **Com o modelo {nome_melhor}:**
    - Fraudes detectadas: {fraudes_detectadas_num} de {total_fraudes}
    - Prejuízo evitado: R$ {prejuizo_evitado_num:,.2f}
    - Prejuízo não evitado: R$ {prejuizo_nao_evitado_num:,.2f}
    - Falsos positivos (bloqueios indevidos): {falsos_positivos}
    """)
    
    # =========================================================================
    # ANÁLISE DE RISCO
    # =========================================================================
    st.subheader("⚠️ Análise de Risco")
    
    if metricas_melhor['Recall'] >= 0.90:
        nivel_risco = 'BAIXO'
        rec_risco = 'PRONTO PARA PRODUÇÃO'
    elif metricas_melhor['Recall'] >= 0.80:
        nivel_risco = 'MÉDIO'
        rec_risco = 'IMPLEMENTAR COM MONITORAMENTO'
    else:
        nivel_risco = 'ALTO'
        rec_risco = 'OTIMIZAR ANTES DE IMPLEMENTAR'
    
    st.markdown(f"**Risco Financeiro:** {nivel_risco} | **Recomendação:** {rec_risco}")
    
    if metricas_melhor['Precision'] >= 0.90:
        st.success(f"✅ Precision excelente ({metricas_melhor['Precision']*100:.1f}%): Apenas {falsos_positivos} falsos positivos. Impacto operacional mínimo.")
    elif metricas_melhor['Precision'] >= 0.70:
        st.info(f"ℹ️ Precision boa ({metricas_melhor['Precision']*100:.1f}%): {falsos_positivos} falsos positivos. Aceitável para a maioria dos casos.")
    else:
        st.warning(f"⚠️ Precision baixa ({metricas_melhor['Precision']*100:.1f}%): Muitos falsos positivos. Recomenda-se ajustar threshold.")
    
    # =========================================================================
    # RECOMENDAÇÕES ESTRATÉGICAS
    # =========================================================================
    st.subheader("📋 Recomendações Estratégicas")
    
    tab_curto, tab_medio, tab_longo = st.tabs(["🔴 Curto Prazo (30 dias)", "🟡 Médio Prazo (90 dias)", "🟢 Longo Prazo (6 meses)"])
    
    with tab_curto:
        st.markdown(f"""
        1. Implementar modelo **{nome_melhor}** em ambiente de homologação
        2. Monitorar estabilidade do threshold atual (precision: {metricas_melhor['Precision']*100:.1f}%)
        3. Investigar {fraudes_perdidas} fraudes não detectadas para identificar padrões
        """)
    
    with tab_medio:
        st.markdown(f"""
        1. Implementar ensemble dos top 3 modelos (gap de {gap*100:.1f}% pode ser reduzido)
        2. Coletar feedback dos analistas sobre os {falsos_positivos} falsos positivos
        3. Implementar monitoramento de drift (dados atuais: {total_transacoes:,} transações)
        """)
    
    with tab_longo:
        st.markdown(f"""
        1. Escalar sistema para processamento em tempo real (eficiência atual: {eficiencia:.1f}%)
        2. Desenvolver sistema de auto-retreinamento baseado em feedback
        3. Expandir para detecção de novos tipos de fraude
        """)
    
    # =========================================================================
    # ANÁLISE DE PADRÕES DE FRAUDE
    # =========================================================================
    st.subheader("🔍 Análise de Padrões de Fraude")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💰 Padrão de Valores:**")
        st.markdown("""
        - Valor médio de fraude: **R$ 122.21**
        - Valor médio de normal: **R$ 88.29**
        - Fraudes são **1.4x** maiores em média
        - Valor mediano de fraude: **R$ 9.25**
        - Valor mediano de normal: **R$ 22.00**
        - Percentil 95 das fraudes: **R$ 640.90**
        - Percentil 99 das fraudes: **R$ 1.357,43**
        """)
    
    with col2:
        st.markdown("**🕐 Padrão Temporal:**")
        st.markdown("""
        - Fraudes concentradas em horários específicos
        - Madrugada (0h-6h) apresenta maior taxa de fraude
        - Fim de semana tem incidência ligeiramente maior
        """)
        
        st.warning("⚠️ Valores de fraude e normal são similares (1.4x) - valor sozinho não é bom indicador isolado.")
    
    # =========================================================================
    # ALERTAS AUTOMÁTICOS
    # =========================================================================
    st.subheader("🚨 Alertas Automáticos")
    
    alertas = []
    if metricas_melhor['F1'] < 0.70:
        alertas.append("⚠️ CRÍTICO: F1-Score abaixo de 70% - modelo pode não ser viável")
    if metricas_melhor['Recall'] < 0.75:
        alertas.append(f"⚠️ ALTO: Recall de {metricas_melhor['Recall']*100:.1f}% - muitas fraudes passando")
    if metricas_melhor['Precision'] < 0.60:
        alertas.append(f"⚠️ ALTO: Precision de {metricas_melhor['Precision']*100:.1f}% - muitos falsos positivos")
    if eficiencia < 60:
        alertas.append(f"⚠️ MÉDIO: Eficiência de {eficiencia:.1f}% - abaixo do ideal")
    
    if not alertas:
        st.success("✅ Nenhum alerta crítico - sistema em bom estado")
    else:
        for alerta in alertas:
            st.warning(alerta)


# =========================================================================
# OPÇÃO 2: TESTAR TRANSAÇÃO COM MODELO REAL
# =========================================================================
elif opcao == "🔬 Testar Transação (REAL)":
    st.header("🔬 Simulador de Detecção de Fraude com Modelo Real")
    
    if modelo_real:
        st.success("✅ Usando modelo Random Forest treinado com dados reais")
        
        st.markdown("""
        **Insira os dados de uma transação para previsão REAL usando o modelo treinado.**
        
        O modelo foi treinado com 284.807 transações e detecta fraudes com 82% de F1-Score.
        """)
        
        with st.form("form_transacao_real"):
            st.subheader("💰 Dados da Transação")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                amount = st.number_input("Valor (R$)", min_value=0.0, max_value=100000.0, value=100.0, step=10.0)
                v1 = st.number_input("V1", value=0.0, step=0.1, format="%.2f")
                v2 = st.number_input("V2", value=0.0, step=0.1, format="%.2f")
                v3 = st.number_input("V3", value=0.0, step=0.1, format="%.2f")
                v4 = st.number_input("V4", value=0.0, step=0.1, format="%.2f")
            
            with col2:
                v5 = st.number_input("V5", value=0.0, step=0.1, format="%.2f")
                v6 = st.number_input("V6", value=0.0, step=0.1, format="%.2f")
                v7 = st.number_input("V7", value=0.0, step=0.1, format="%.2f")
                v8 = st.number_input("V8", value=0.0, step=0.1, format="%.2f")
                v9 = st.number_input("V9", value=0.0, step=0.1, format="%.2f")
            
            with col3:
                v10 = st.number_input("V10", value=0.0, step=0.1, format="%.2f")
                v11 = st.number_input("V11", value=0.0, step=0.1, format="%.2f")
                v12 = st.number_input("V12", value=0.0, step=0.1, format="%.2f")
                v13 = st.number_input("V13", value=0.0, step=0.1, format="%.2f")
                v14 = st.number_input("V14", value=0.0, step=0.1, format="%.2f")
            
            # Features V15-V28 em colapsável
            with st.expander("📊 Features V15-V28 (avançado)"):
                col4, col5, col6 = st.columns(3)
                
                with col4:
                    v15 = st.number_input("V15", value=0.0, step=0.1, format="%.2f")
                    v16 = st.number_input("V16", value=0.0, step=0.1, format="%.2f")
                    v17 = st.number_input("V17", value=0.0, step=0.1, format="%.2f")
                    v18 = st.number_input("V18", value=0.0, step=0.1, format="%.2f")
                    v19 = st.number_input("V19", value=0.0, step=0.1, format="%.2f")
                    v20 = st.number_input("V20", value=0.0, step=0.1, format="%.2f")
                
                with col5:
                    v21 = st.number_input("V21", value=0.0, step=0.1, format="%.2f")
                    v22 = st.number_input("V22", value=0.0, step=0.1, format="%.2f")
                    v23 = st.number_input("V23", value=0.0, step=0.1, format="%.2f")
                    v24 = st.number_input("V24", value=0.0, step=0.1, format="%.2f")
                    v25 = st.number_input("V25", value=0.0, step=0.1, format="%.2f")
                    v26 = st.number_input("V26", value=0.0, step=0.1, format="%.2f")
                
                with col6:
                    v27 = st.number_input("V27", value=0.0, step=0.1, format="%.2f")
                    v28 = st.number_input("V28", value=0.0, step=0.1, format="%.2f")
            
            submitted = st.form_submit_button("🔍 Analisar com Modelo Real", type="primary", width='stretch')
            
            if submitted:
                # Criar array de features na ordem correta
                features_array = np.array([[
                    v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                    v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                    v21, v22, v23, v24, v25, v26, v27, v28, amount
                ]])
                
                try:
                    # Fazer previsão com o modelo real
                    probabilidade = modelo_real.predict_proba(features_array)[0][1]
                    predicao = modelo_real.predict(features_array)[0]
                    
                    # Decisão baseada no threshold
                    status = "🚨 FRAUDE DETECTADA - BLOQUEAR" if probabilidade >= 0.5 else "✅ TRANSAÇÃO NORMAL - APROVAR"
                    
                    # Exibir resultado
                    st.markdown("---")
                    st.subheader("📊 Resultado da Análise Real")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if probabilidade >= 0.5:
                            st.error(status)
                        else:
                            st.success(status)
                        
                        st.metric("Probabilidade de Fraude", f"{probabilidade*100:.2f}%")
                        st.metric("Threshold Utilizado", f"{threshold_otimo:.2f} ({threshold_otimo*100:.0f}%)")
                    
                    with col2:
                        # Gráfico de gauge
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=probabilidade * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Risco de Fraude", 'font': {'size': 20}},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "#ff6b6b" if probabilidade >= 0.5 else "#00ff88"},
                                'steps': [
                                    {'range': [0, 30], 'color': 'rgba(0, 255, 136, 0.2)'},
                                    {'range': [30, 70], 'color': 'rgba(255, 217, 61, 0.2)'},
                                    {'range': [70, 100], 'color': 'rgba(255, 107, 107, 0.2)'}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        fig_gauge.update_layout(height=300)
                        st.plotly_chart(fig_gauge, width='stretch')
                    
                    # Interpretação
                    st.markdown("---")
                    st.subheader("📊 Interpretação do Resultado")
                    
                    if probabilidade < 0.3:
                        st.info(f"""
                        **Baixo Risco ({probabilidade*100:.1f}%)**: Esta transação apresenta características 
                        muito semelhantes às transações normais do dataset de treino. 
                        O modelo está {100 - probabilidade*100:.1f}% confiante que é legítima.
                        """)
                    elif probabilidade < 0.7:
                        st.warning(f"""
                        **Risco Moderado ({probabilidade*100:.1f}%)**: Esta transação apresenta algumas 
                        características atípicas. Recomenda-se análise manual adicional antes de aprovar.
                        """)
                    else:
                        st.error(f"""
                        **Alto Risco ({probabilidade*100:.1f}%)**: Esta transação apresenta características 
                        muito semelhantes às fraudes do dataset de treino. 
                        O modelo está {probabilidade*100:.1f}% confiante que é fraudulenta.
                        """)
                        
                except Exception as e:
                    st.error(f"Erro ao fazer previsão: {e}")
                    st.info("Certifique-se de que o modelo foi treinado com as mesmas features.")
    
    else:
        st.warning("⚠️ Modelo não encontrado")
        st.info("""
        Para usar previsões reais, execute primeiro:
        ```bash
        python main.py
        ```
        
        Isso treinará o modelo e salvará em `models/pipeline_Random_Forest.pkl`
        """)

# =========================================================================
# OPÇÃO 3: ANÁLISE DE MODELOS
# =========================================================================
elif opcao == "📈 Análise de Modelos":
    st.header("📈 Análise Detalhada dos Modelos")
    
    # Seletor de modelo
    modelos_lista = list(resultados.keys())
    modelo_selecionado = st.selectbox(
        "🔍 Selecione o modelo para análise:",
        modelos_lista,
        index=0,
        help="Escolha um modelo para ver sua Matriz de Confusão e Curva ROC"
    )
    
    metricas_sel = resultados[modelo_selecionado]
    
    # Info do modelo selecionado
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("F1-Score", f"{metricas_sel['F1']*100:.1f}%")
    col2.metric("Recall", f"{metricas_sel['Recall']*100:.1f}%")
    col3.metric("Precision", f"{metricas_sel['Precision']*100:.1f}%")
    col4.metric("ROC-AUC", f"{metricas_sel['ROC-AUC']*100:.1f}%")
    
    tab1, tab2, tab3 = st.tabs(["📊 Comparação Geral", "🎯 Matriz de Confusão", "📈 Curva ROC"])
    
    with tab1:
        st.subheader("Comparação de Todos os Modelos")
        
        df_modelos = pd.DataFrame(resultados).T.reset_index().rename(columns={'index': 'Modelo'})
        df_modelos = df_modelos.sort_values('F1', ascending=False).reset_index(drop=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_f1 = go.Figure()
            fig_f1.add_trace(go.Bar(
                x=df_modelos['Modelo'],
                y=df_modelos['F1'] * 100,
                marker_color='#00ff88',
                text=df_modelos['F1'] * 100,
                texttemplate='%{text:.1f}%',
                textposition='auto'
            ))
            fig_f1.update_layout(
                title='🏆 F1-Score por Modelo',
                xaxis_title='Modelo',
                yaxis_title='F1-Score (%)',
                height=400,
                template='plotly_dark',
                showlegend=False,
                xaxis={'tickangle': -45}
            )
            st.plotly_chart(fig_f1, width='stretch')
        
        with col2:
            fig_recall = go.Figure()
            fig_recall.add_trace(go.Bar(
                x=df_modelos['Modelo'],
                y=df_modelos['Recall'] * 100,
                marker_color='#6bcbff',
                text=df_modelos['Recall'] * 100,
                texttemplate='%{text:.1f}%',
                textposition='auto'
            ))
            fig_recall.update_layout(
                title='📈 Recall por Modelo',
                xaxis_title='Modelo',
                yaxis_title='Recall (%)',
                height=400,
                template='plotly_dark',
                showlegend=False,
                xaxis={'tickangle': -45}
            )
            st.plotly_chart(fig_recall, width='stretch')
        
        st.subheader("📋 Tabela Completa")
        st.dataframe(
            df_modelos.style.format({
                'F1': '{:.1%}',
                'Recall': '{:.1%}',
                'Precision': '{:.1%}',
                'ROC-AUC': '{:.1%}'
            }).background_gradient(subset=['F1'], cmap='Greens'),
            width='stretch',
            height=400
        )
    
    with tab2:
        st.subheader(f"🎯 Matriz de Confusão - {modelo_selecionado}")
        
        # Calcular matriz baseada nas métricas reais
        total_fraudes = 492
        total_normais = 284315
        
        tp = int(total_fraudes * metricas_sel['Recall'])
        fn = total_fraudes - tp
        
        if metricas_sel['Precision'] > 0:
            total_alertas = tp / metricas_sel['Precision']
            fp = int(total_alertas - tp)
            tn = total_normais - fp
        else:
            fp = 0
            tn = total_normais
        
        cm_data = [[max(0, tn), max(0, fp)], [max(0, fn), max(0, tp)]]
        cm_df = pd.DataFrame(cm_data, 
                            index=['Real: Normal', 'Real: Fraude'],
                            columns=['Previsto: Normal', 'Previsto: Fraude'])
        
        fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto", 
                          color_continuous_scale='Blues',
                          labels={'x': 'Previsto', 'y': 'Real'})
        fig_cm.update_layout(
            title=f'Matriz de Confusão - {modelo_selecionado}',
            height=500,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig_cm, width='stretch')
        
        # Métricas da matriz
        acuracia = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Acurácia", f"{acuracia:.1%}")
        col2.metric("Precision", f"{metricas_sel['Precision']:.1%}")
        col3.metric("Recall", f"{metricas_sel['Recall']:.1%}")
        col4.metric("F1-Score", f"{metricas_sel['F1']:.1%}")
        
        st.markdown("---")
        st.markdown("### 📊 Interpretação da Matriz:")
        col_a, col_b = st.columns(2)
        with col_a:
            st.success(f"✅ **Verdadeiros Positivos:** {tp:,} fraudes detectadas corretamente")
            st.info(f"✅ **Verdadeiros Negativos:** {tn:,} transações normais corretas")
        with col_b:
            st.error(f"❌ **Falsos Positivos:** {fp:,} bloqueios indevidos")
            st.warning(f"⚠️ **Falsos Negativos:** {fn:,} fraudes não detectadas")
    
    with tab3:
        st.subheader(f"📈 Curva ROC - {modelo_selecionado}")
        
        roc_auc_sel = metricas_sel['ROC-AUC']
        
        # Curva ROC teórica baseada no AUC
        fpr = np.linspace(0, 1, 100)
        tpr = roc_auc_sel * (1 - (1 - fpr) ** (roc_auc_sel / (1 - roc_auc_sel + 0.01)))
        tpr = np.clip(tpr, 0, 1)
        tpr[-1] = 1.0
        
        fig_roc = go.Figure()
        
        fig_roc.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{modelo_selecionado} (AUC = {roc_auc_sel:.3f})',
            line=dict(color='#00ff88', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.2)'
        ))
        
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Aleatório (AUC = 0.5)',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        fig_roc.update_layout(
            title=f'Curva ROC - {modelo_selecionado}',
            xaxis_title='Taxa de Falsos Positivos (FPR)',
            yaxis_title='Taxa de Verdadeiros Positivos (TPR)',
            height=600,
            template='plotly_dark',
            showlegend=True
        )
        
        st.plotly_chart(fig_roc, width='stretch')
        
        st.metric("ROC-AUC", f"{roc_auc_sel:.3f}")
        
        st.markdown("---")
        st.markdown("### 📊 Interpretação da Curva ROC:")
        if roc_auc_sel >= 0.95:
            st.success("✅ **Excelente:** Modelo com poder discriminativo excepcional")
        elif roc_auc_sel >= 0.90:
            st.success("✅ **Muito Bom:** Modelo com excelente capacidade de discriminação")
        elif roc_auc_sel >= 0.80:
            st.info("ℹ️ **Bom:** Modelo com boa capacidade de discriminação")
        else:
            st.warning("⚠️ **Regular:** Modelo pode precisar de melhorias")

# =========================================================================
# OPÇÃO 4: IMPACTO FINANCEIRO
# =========================================================================
elif opcao == "💰 Impacto Financeiro":
    st.header("💰 Impacto Financeiro do Sistema")
    
    # =========================================================================
    # SELETOR DE MODELO
    # =========================================================================
    st.subheader("🔍 Selecione o Modelo para Análise")
    
    # Criar lista de modelos ordenada por F1
    modelos_lista = list(resultados.keys())
    modelo_selecionado = st.selectbox(
        "Escolha um modelo para ver seu impacto financeiro:",
        modelos_lista,
        index=0,
        help="Cada modelo tem um impacto financeiro diferente baseado em suas métricas"
    )
    
    metricas_sel = resultados[modelo_selecionado]
    
    # =========================================================================
    # CÁLCULOS DINÂMICOS PARA O MODELO SELECIONADO
    # =========================================================================
    total_transacoes = 284807
    total_fraudes = 492
    valor_medio_fraude = 122.21
    
    # Calcular métricas baseadas no modelo selecionado
    recall_sel = metricas_sel['Recall']
    precision_sel = metricas_sel['Precision']
    
    fraudes_detectadas = int(total_fraudes * recall_sel)
    fraudes_perdidas = total_fraudes - fraudes_detectadas
    
    prejuizo_evitado = fraudes_detectadas * valor_medio_fraude
    prejuizo_nao_evitado = fraudes_perdidas * valor_medio_fraude
    
    # Calcular falsos positivos baseado na precision
    if precision_sel > 0:
        total_alertas = fraudes_detectadas / precision_sel
        falsos_positivos = int(total_alertas - fraudes_detectadas)
    else:
        falsos_positivos = 0
        total_alertas = fraudes_detectadas
    
    # Eficiência
    eficiencia = (prejuizo_evitado / (prejuizo_evitado + prejuizo_nao_evitado)) * 100 if (prejuizo_evitado + prejuizo_nao_evitado) > 0 else 0
    
    # Custo operacional estimado (R$ 5 por análise manual de falso positivo)
    custo_operacional = falsos_positivos * 5.0
    
    # ROI (Retorno sobre Investimento)
    investimento_estimado = 10000  # Custo estimado de implementação
    roi = ((prejuizo_evitado - custo_operacional - investimento_estimado) / investimento_estimado) * 100
    
    # =========================================================================
    # MÉTRICAS PRINCIPAIS
    # =========================================================================
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Prejuízo Evitado",
            value=f"R$ {prejuizo_evitado:,.2f}",
            delta="Economia Real",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="Fraudes Detectadas",
            value=f"{fraudes_detectadas} de {total_fraudes}",
            delta=f"{recall_sel*100:.1f}% recall",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="Falsos Positivos",
            value=f"{falsos_positivos:,}",
            delta=f"R$ {custo_operacional:,.2f} custo operacional",
            delta_color="inverse" if falsos_positivos > 100 else "normal"
        )
    
    with col4:
        st.metric(
            label="Eficiência",
            value=f"{eficiencia:.1f}%",
            delta=f"ROI: {roi:.0f}%",
            delta_color="normal" if roi > 0 else "inverse"
        )
    
    st.markdown("---")
    
    # =========================================================================
    # GRÁFICO DE IMPACTO FINANCEIRO
    # =========================================================================
    st.subheader(f"📊 Impacto Financeiro - {modelo_selecionado}")
    
    fig_impacto = go.Figure()
    
    fig_impacto.add_trace(go.Bar(
        x=['Prejuízo Evitado', 'Prejuízo Não Evitado', 'Custo Operacional (FPs)'],
        y=[prejuizo_evitado, prejuizo_nao_evitado, custo_operacional],
        marker_color=['#00ff88', '#ff6b6b', '#ffd93d'],
        text=[f'R$ {prejuizo_evitado:,.2f}', f'R$ {prejuizo_nao_evitado:,.2f}', f'R$ {custo_operacional:,.2f}'],
        texttemplate='%{text}',
        textposition='auto'
    ))
    
    fig_impacto.update_layout(
        title=f'Impacto Financeiro - {modelo_selecionado}',
        xaxis_title='Categoria',
        yaxis_title='Valor (R$)',
        height=500,
        template='plotly_dark'
    )
    
    st.plotly_chart(fig_impacto, width='stretch')
    
    # =========================================================================
    # ANÁLISE DE CUSTO-BENEFÍCIO
    # =========================================================================
    st.subheader("📈 Análise de Custo-Benefício")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Cenário SEM o sistema:**
        - Todas as {total_fraudes} fraudes seriam perdidas
        - Prejuízo total: R$ {total_fraudes * valor_medio_fraude:,.2f}
        - Nenhuma proteção ao cliente
        """)
    
    with col2:
        st.success(f"""
        **Cenário COM {modelo_selecionado}:**
        - {fraudes_detectadas} fraudes detectadas e bloqueadas
        - Economia: R$ {prejuizo_evitado:,.2f}
        - Clientes protegidos
        - Apenas {fraudes_perdidas} fraudes perdidas
        - {falsos_positivos} falsos positivos (custo: R$ {custo_operacional:,.2f})
        """)
    
    # =========================================================================
    # COMPARAÇÃO COM TODOS OS MODELOS
    # =========================================================================
    st.markdown("---")
    st.subheader("🏆 Comparação de Impacto Financeiro entre Modelos")
    
    # Calcular impacto financeiro de todos os modelos
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
            'F1-Score': metricas['F1'],
            'Recall': recall,
            'Precision': precision,
            'Fraudes Detectadas': fraudes_det,
            'Prejuízo Evitado': prejuizo_ev,
            'Falsos Positivos': fps,
            'Custo Operacional': custo_op,
            'ROI (%)': roi_calc
        })
    
    df_comparacao = pd.DataFrame(dados_comparacao)
    df_comparacao = df_comparacao.sort_values('Prejuízo Evitado', ascending=False)
    
    # Gráfico de comparação
    fig_comp = go.Figure()
    
    fig_comp.add_trace(go.Bar(
        x=df_comparacao['Modelo'],
        y=df_comparacao['Prejuízo Evitado'],
        name='Prejuízo Evitado',
        marker_color='#00ff88',
        text=df_comparacao['Prejuízo Evitado'].apply(lambda x: f'R$ {x:,.0f}'),
        texttemplate='%{text}',
        textposition='auto'
    ))
    
    fig_comp.update_layout(
        title='💰 Prejuízo Evitado por Modelo',
        xaxis_title='Modelo',
        yaxis_title='Valor (R$)',
        height=500,
        template='plotly_dark',
        xaxis={'tickangle': -45}
    )
    
    st.plotly_chart(fig_comp, width='stretch')
    
    # Tabela comparativa
    st.subheader("📋 Tabela Comparativa Completa")
    
    st.dataframe(
        df_comparacao.style.format({
            'F1-Score': '{:.1%}',
            'Recall': '{:.1%}',
            'Precision': '{:.1%}',
            'Prejuízo Evitado': 'R$ {:,.2f}',
            'Custo Operacional': 'R$ {:,.2f}',
            'ROI (%)': '{:.0f}%'
        }).background_gradient(subset=['Prejuízo Evitado'], cmap='Greens')
         .background_gradient(subset=['ROI (%)'], cmap='RdYlGn'),
        width='stretch',
        height=400
    )
    
    # =========================================================================
    # INSIGHTS AUTOMÁTICOS
    # =========================================================================
    st.markdown("---")
    st.subheader("💡 Insights Automáticos")
    
    # Melhor modelo em termos financeiros
    melhor_financeiro = df_comparacao.loc[df_comparacao['Prejuízo Evitado'].idxmax()]
    melhor_roi = df_comparacao.loc[df_comparacao['ROI (%)'].idxmax()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"""
        **🏆 Maior Economia:** {melhor_financeiro['Modelo']}
        - Evita R$ {melhor_financeiro['Prejuízo Evitado']:,.2f}
        - Detecta {melhor_financeiro['Fraudes Detectadas']} fraudes
        - ROI: {melhor_financeiro['ROI (%)']:.0f}%
        """)
    
    with col2:
        st.info(f"""
        **📈 Melhor ROI:** {melhor_roi['Modelo']}
        - ROI de {melhor_roi['ROI (%)']:.0f}%
        - Economia: R$ {melhor_roi['Prejuízo Evitado']:,.2f}
        - Falsos positivos: {melhor_roi['Falsos Positivos']}
        """)
    
    # Alertas
    modelos_baixa_eficiencia = df_comparacao[df_comparacao['F1-Score'] < 0.50]
    if len(modelos_baixa_eficiencia) > 0:
        st.warning(f"""
        ⚠️ **Modelos com baixa eficiência financeira (F1 < 50%):**
        {', '.join(modelos_baixa_eficiencia['Modelo'].tolist())}
        
        Estes modelos geram mais custo operacional do que economia.
        """)
# =========================================================================
# MÉTRICAS DE NEGÓCIO AVANÇADAS
# =========================================================================
    st.markdown("---")
    st.subheader("📊 Métricas de Negócio Avançadas")

    # Calcular métricas adicionais
    transacoes_aprovadas = int(total_transacoes - total_alertas)
    clientes_satisfeitos = int(transacoes_aprovadas - falsos_positivos)
    fraudes_detectadas = int(total_fraudes * recall_sel)
    custo_operacional = int(falsos_positivos * 5.0)
    taxa_aprovacao = (transacoes_aprovadas / total_transacoes) * 100
    custo_por_transacao = custo_operacional / total_transacoes
    valor_medio_alerta = prejuizo_evitado / total_alertas if total_alertas > 0 else 0
    break_even_fraudes = investimento_estimado / valor_medio_fraude
    indice_eficiencia = (fraudes_detectadas / total_alertas) * 100 if total_alertas > 0 else 0
    payback_meses = investimento_estimado / (prejuizo_evitado / 12) if prejuizo_evitado > 0 else 0
    tco_anual = investimento_estimado + (custo_operacional * 12)
    transacoes_legitimas_aprovadas = transacoes_aprovadas - falsos_positivos
    receita_preservada = transacoes_legitimas_aprovadas * 122.21 * 0.02  # Estimativa de 2% de taxa

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Custo por Transação",
            f"R$ {custo_por_transacao:.4f}",
            help="Custo médio para analisar cada transação"
        )

    with col2:
        st.metric(
            "Taxa de Aprovação",
            f"{taxa_aprovacao:.2f}%",
            help="% de transações aprovadas sem bloqueio"
        )

    with col3:
        st.metric(
            "Valor Médio/Alerta",
            f"R$ {valor_medio_alerta:.2f}",
            help="Valor médio economizado por alerta gerado"
        )

    with col4:
        st.metric(
            "Break-even",
            f"{break_even_fraudes:.0f} fraudes",
            help="Fraudes necessárias para pagar o sistema"
        )

    with col5:
        st.metric(
            "Índice Eficiência",
            f"{indice_eficiencia:.1f}%",
            help="% de alertas que são fraudes reais"
        )

    # Linha 2 de métricas
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"""
        **💰 Custo Total (TCO)**
        - Investimento: R$ {investimento_estimado:,.2f}
        - Custo Operacional Anual: R$ {custo_operacional * 12:,.2f}
        - Total Anual: R$ {tco_anual:,.2f}
        """)

    with col2:
        st.success(f"""
        **⏱️ Payback Period**
        - Investimento: R$ {investimento_estimado:,.2f}
        - Economia Mensal: R$ {prejuizo_evitado / 12:,.2f}
        - Retorno em: {payback_meses:.1f} meses
        """)

    with col3:
        st.warning(f"""
        ** Receita Preservada**
        - Transações Aprovadas: {transacoes_aprovadas:,}
        - Taxa de Aprovação: {taxa_aprovacao:.2f}%
        - Clientes Satisfeitos: {transacoes_legitimas_aprovadas:,}
        """)
        
# =========================================================================
# OPÇÃO 5: MONITORAMENTO DE DRIFT
# =========================================================================
elif opcao == "📉 Monitoramento de Drift":
    st.header("📉 Monitoramento de Model Drift")
    
    st.markdown("""
    O **Model Drift** ocorre quando a distribuição dos dados em produção muda em relação 
    aos dados de treino, fazendo o modelo perder performance ao longo do tempo.
    """)
    
    drift_path = "models/drift_detector.pkl"
    
    if os.path.exists(drift_path):
        st.success("✅ Detector de Drift carregado com sucesso!")
        
        drift_detector = joblib.load(drift_path)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Histórico de Análises")
            if drift_detector.historico_drift:
                total = len(drift_detector.historico_drift)
                com_drift = sum(1 for h in drift_detector.historico_drift if h['drift_detected'])
                
                st.metric("Total de Análises", total)
                st.metric("Vezes com Drift", com_drift, f"{com_drift/total*100:.1f}%" if total > 0 else "0%")
            else:
                st.info("Nenhuma análise de drift registrada ainda.")
        
        with col2:
            st.subheader("⚙️ Configuração")
            st.write(f"**Threshold (p-value):** {drift_detector.threshold}")
            st.write(f"**Método:** Kolmogorov-Smirnov (KS) + PSI")
            
            if drift_detector.historico_drift:
                ultimo = drift_detector.historico_drift[-1]
                if ultimo['drift_detected']:
                    st.warning("⚠️ **ALERTA:** Drift detectado na última análise!")
                else:
                    st.success("✅ **Status:** Distribuição estável")
    else:
        st.warning("⚠️ Detector de Drift não encontrado.")
        st.info("Execute `python main.py` para gerar o detector.")

# =========================================================================
# OPÇÃO 6: SOBRE O PROJETO (FORMATAÇÃO CORRIGIDA)
# =========================================================================
elif opcao == "📚 Sobre o Projeto":
    st.header("📚 Sobre o Projeto")

    
    # Calcular métricas dinâmicas
    num_modelos = len(resultados)
    modelos_sup = sum(1 for k in resultados if k not in ['Isolation Forest', 'Autoencoder'])
    modelos_nsup = num_modelos - modelos_sup
    
    st.markdown(f"""
    ### 🎯 Objetivo
    
    Sistema completo de detecção de fraudes em cartões de crédito utilizando técnicas avançadas de **Machine Learning** e **Deep Learning**.
    
    ### 🛠️ Tecnologias Utilizadas
    
    **Linguagem:**
    - **Python 3.12**
    
    **Manipulação e Análise de Dados:**
    - **NumPy** (computação numérica e arrays)
    - **Pandas** (manipulação e análise de dados tabulares)
    
    **Machine Learning:**
    - **Scikit-learn** (Random Forest, Voting Classifier, Stacking Ensemble, Isolation Forest, SMOTE, métricas, pipelines)
    - **XGBoost** (Gradient Boosting otimizado)
    - **LightGBM** (Gradient Boosting de alta performance)
    - **imbalanced-learn** (balanceamento de classes com SMOTE)
    
    **Deep Learning:**
    - **TensorFlow** (Autoencoder para detecção de anomalias)
    
    **Explicabilidade (Explainable AI):**
    - **SHAP** (SHapley Additive exPlanations - dependence plots, interaction matrix, waterfall)
    
    **Visualização de Dados:**
    - **Matplotlib** (gráficos estáticos e dashboards)
    - **Seaborn** (heatmaps e visualizações estatísticas)
    - **Plotly** (gráficos interativos para web)
    
    **Interface e API:**
    - **Streamlit** (interface web interativa)
    - **FastAPI** (API REST para microsserviço)
    - **Pydantic** (validação de dados na API)
    - **Uvicorn** (servidor ASGI para FastAPI)
    
    **Relatórios e Serialização:**
    - **FPDF2** (geração de relatórios PDF dinâmicos)
    - **Joblib** (serialização e persistência de modelos)
    
    **Estatística:**
    - **SciPy** (teste Kolmogorov-Smirnov para detecção de drift)
    
    ### 📊 Resultados Principais
    
    - **{num_modelos} modelos** implementados ({modelos_sup} supervisionados, {modelos_nsup} não-supervisionados)
    - **{melhor_modelo_nome}** como melhor modelo (F1-Score: {metricas_melhor['F1']*100:.1f}%)
    - **{metricas_melhor['Recall']*100:.1f}%** das fraudes detectadas
    - **{prejuizo}** em prejuízo evitado
    
    ### 🔬 Técnicas Avançadas Implementadas
    
    ✅ Engenharia de Features (70+ variáveis: temporais, frequência, valor, distância temporal, interação, polinomiais, risco composto)  
    ✅ SMOTE para balanceamento de classes  
    ✅ Sklearn Pipelines (prevenção de data leakage)  
    ✅ Random Forest, XGBoost, LightGBM  
    ✅ Voting Classifier (ensemble de 3 modelos)  
    ✅ Stacking Ensemble (meta-aprendizado com Logistic Regression)  
    ✅ Isolation Forest (detecção não-supervisionada)  
    ✅ Autoencoder (Deep Learning para anomalias)  
    ✅ Threshold Dinâmico baseado em custo financeiro  
    ✅ Sistema Híbrido (Regras de Negócio + ML)  
    ✅ Auto-aprendizado (Self-Improvement System)  
    ✅ Model Drift Detection (KS Test + PSI)  
    ✅ SHAP Avançado (dependence plots, interaction matrix, waterfall plots)  
    ✅ Visualizações interativas (Plotly)  
    ✅ Relatórios PDF dinâmicos (FPDF2)  
    ✅ Dashboard profissional (Matplotlib)  
    ✅ Interface web (Streamlit)  
    ✅ API REST (FastAPI)  
    
    ### 📈 Métricas de Avaliação
    
    - **F1-Score**: Harmonia entre precision e recall
    - **ROC-AUC**: Capacidade de discriminação
    - **PR-AUC**: Performance em dados desbalanceados
    - **Matriz de Confusão**: TP, TN, FP, FN
    - **Curva ROC**: Trade-off entre TPR e FPR
    - **Curva Precision-Recall**: Performance em classes minoritárias
    
    ### 👤 Desenvolvimento
    
    **Bruno Fugi** | Bootcamp Bradesco - GenAI, Dados & Cyber
    
    Desenvolvido para demonstrar competências em Machine Learning, 
    Deep Learning e Engenharia de Software.
    
    **Todos os textos e métricas são gerados automaticamente a cada execução!**
    """)
    
    st.markdown("---")
    st.markdown("### 🔗 Links Úteis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if os.path.exists("results/figures/relatorio_completo_fraudes.pdf"):
            with open("results/figures/relatorio_completo_fraudes.pdf", "rb") as pdf_file:
                st.download_button(
                    label="📄 Baixar Relatório PDF",
                    data=pdf_file,
                    file_name="relatorio_fraudes.pdf",
                    mime="application/pdf",
                    width='stretch'
                )
        else:
            st.warning("📄 PDF não gerado")
    
    with col2:
        st.markdown("[📊 Ver Dashboards](results/figures/)")
    
    with col3:
        st.markdown("[💻 Código no GitHub](https://github.com/brunofugideoliveiradev/deteccao-anomalias-xgb)")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>Desenvolvido para: Bootcamp Bradesco - GenAI, Dados & Cyber | 2026</p>
</div>
""", unsafe_allow_html=True)