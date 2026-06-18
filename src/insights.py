"""
Módulo de Geração de Insights Textuais.
VERSÃO 100% DINÂMICA: Tudo gerado baseado nos resultados reais.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from src.utils import logger, criar_diretorio


def gerar_relatorio_executivo(resultados: dict, df_original: pd.DataFrame):
    """
    Gera relatório executivo com insights de negócio 100% dinâmicos.
    """
    logger.info("\n" + "="*80)
    logger.info("📊 RELATÓRIO EXECUTIVO - INSIGHTS DE NEGÓCIO (100% DINÂMICO)")
    logger.info("="*80)
    
    # Encontrar melhor modelo
    melhor_modelo = max(resultados.items(), key=lambda x: x[1]['F1'])
    nome_melhor = melhor_modelo[0]
    metricas_melhor = melhor_modelo[1]
    
    # Calcular métricas derivadas
    total_transacoes = len(df_original)
    total_fraudes = int(df_original['Class'].sum())
    valor_medio_fraude = df_original[df_original['Class']==1]['Amount'].mean()
    
    fraudes_detectadas = int(total_fraudes * metricas_melhor['Recall'])
    fraudes_perdidas = int(total_fraudes - fraudes_detectadas)
    
    total_alertas = int(fraudes_detectadas / metricas_melhor['Precision']) if metricas_melhor['Precision'] > 0 else 0
    falsos_positivos = int(total_alertas - fraudes_detectadas)
    
    prejuizo_evitado = fraudes_detectadas * valor_medio_fraude
    prejuizo_nao_evitado = fraudes_perdidas * valor_medio_fraude
    
    # Classificação dinâmica de desempenho
    if metricas_melhor['F1'] >= 0.85:
        nivel_desempenho = "EXCEPCIONAL"
    elif metricas_melhor['F1'] >= 0.80:
        nivel_desempenho = "EXCELENTE"
    elif metricas_melhor['F1'] >= 0.70:
        nivel_desempenho = "BOM"
    elif metricas_melhor['F1'] >= 0.60:
        nivel_desempenho = "REGULAR"
    else:
        nivel_desempenho = "INSUFICIENTE"
    
    # =========================================================================
    # 1. RESUMO EXECUTIVO DINÂMICO
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("📋 RESUMO EXECUTIVO")
    logger.info("="*80)
    
    logger.info(f"\n🏆 MELHOR MODELO: {nome_melhor}")
    logger.info(f"   • Nível de desempenho: {nivel_desempenho}")
    logger.info(f"   • F1-Score: {metricas_melhor['F1']*100:.1f}%")
    logger.info(f"   • Detecta {metricas_melhor['Recall']*100:.1f}% das fraudes reais")
    logger.info(f"   • {metricas_melhor['Precision']*100:.1f}% dos alertas são fraudes reais")
    logger.info(f"   • ROC-AUC: {metricas_melhor['ROC-AUC']*100:.1f}%")
    
    # =========================================================================
    # 2. IMPACTO FINANCEIRO DINÂMICO
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("💰 IMPACTO FINANCEIRO ESTIMADO")
    logger.info("="*80)
    
    logger.info(f"\n📊 CENÁRIO ATUAL (Dataset):")
    logger.info(f"   • Total de transações: {total_transacoes:,}")
    logger.info(f"   • Total de fraudes: {total_fraudes:,} ({total_fraudes/total_transacoes*100:.3f}%)")
    logger.info(f"   • Valor médio por fraude: R$ {valor_medio_fraude:.2f}")
    logger.info(f"   • Prejuízo total potencial: R$ {total_fraudes * valor_medio_fraude:,.2f}")
    
    logger.info(f"\n✅ COM O MODELO {nome_melhor.upper()}:")
    logger.info(f"   • Fraudes detectadas: {fraudes_detectadas:,} de {total_fraudes:,}")
    logger.info(f"   • Prejuízo evitado: R$ {prejuizo_evitado:,.2f}")
    logger.info(f"   • Prejuízo não evitado: R$ {prejuizo_nao_evitado:,.2f}")
    logger.info(f"   • Falsos positivos (bloqueios indevidos): {falsos_positivos:,}")
    
    # Eficiência dinâmica
    eficiencia = (prejuizo_evitado / (prejuizo_evitado + prejuizo_nao_evitado)) * 100 if (prejuizo_evitado + prejuizo_nao_evitado) > 0 else 0
    logger.info(f"\n📈 EFICIÊNCIA DO MODELO: {eficiencia:.1f}%")
    
    if eficiencia >= 85:
        logger.info("   • Status: ALTAMENTE EFICIENTE - Pronto para produção")
    elif eficiencia >= 70:
        logger.info("   • Status: EFICIENTE - Pode ser implementado com monitoramento")
    elif eficiencia >= 50:
        logger.info("   • Status: MODERADO - Requer otimizações antes da produção")
    else:
        logger.info("   • Status: BAIXA EFICIÊNCIA - Necessita melhorias significativas")
    
    # =========================================================================
    # 3. ANÁLISE DE RISCO DINÂMICA
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("⚠️ ANÁLISE DE RISCO")
    logger.info("="*80)
    
    # Classificação dinâmica baseada em recall
    if metricas_melhor['Recall'] >= 0.90:
        logger.info("\n✅ EXCELENTE: Modelo detecta 90%+ das fraudes")
        logger.info("   • Risco financeiro: BAIXO")
        logger.info("   • Recomendação: PRONTO PARA PRODUÇÃO")
    elif metricas_melhor['Recall'] >= 0.80:
        logger.info("\n✅ BOM: Modelo detecta 80-90% das fraudes")
        logger.info("   • Risco financeiro: MÉDIO")
        logger.info("   • Recomendação: IMPLEMENTAR COM MONITORAMENTO")
    elif metricas_melhor['Recall'] >= 0.70:
        logger.info("\n⚠️ REGULAR: Modelo detecta 70-80% das fraudes")
        logger.info("   • Risco financeiro: ALTO")
        logger.info(f"   • {fraudes_perdidas:,} fraudes não detectadas")
        logger.info("   • Recomendação: OTIMIZAR THRESHOLD OU FEATURES")
    else:
        logger.info("\n❌ INSUFICIENTE: Modelo detecta menos de 70% das fraudes")
        logger.info("   • Risco financeiro: CRÍTICO")
        logger.info(f"   • {fraudes_perdidas:,} fraudes não detectadas")
        logger.info("   • Recomendação: REAVALIAR MODELO OU TÉCNICAS")
    
    # Análise dinâmica de precision
    if metricas_melhor['Precision'] < 0.50:
        razao_fp = 1/metricas_melhor['Precision'] - 1
        logger.info(f"\n⚠️ ATENÇÃO: Muitos falsos positivos")
        logger.info(f"   • Para cada fraude real, há {razao_fp:.1f} bloqueios indevidos")
        logger.info(f"   • Total de {falsos_positivos:,} clientes afetados indevidamente")
        logger.info("   • Impacto: Clientes insatisfeitos, custos operacionais altos")
        logger.info("   • Recomendação: AJUSTAR THRESHOLD OU MELHORAR FEATURES")
    elif metricas_melhor['Precision'] < 0.70:
        logger.info(f"\n⚠️ Precision moderada ({metricas_melhor['Precision']*100:.1f}%)")
        logger.info(f"   • {falsos_positivos:,} falsos positivos identificados")
        logger.info("   • Recomendação: Considerar ajuste fino de threshold")
    else:
        logger.info(f"\n✅ Precision excelente ({metricas_melhor['Precision']*100:.1f}%)")
        logger.info(f"   • Apenas {falsos_positivos:,} falsos positivos")
        logger.info("   • Impacto operacional mínimo")
    
    # =========================================================================
    # 4. COMPARAÇÃO DINÂMICA DE MODELOS
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("🏆 RANKING DE MODELOS")
    logger.info("="*80)
    
    modelos_ordenados = sorted(resultados.items(), key=lambda x: x[1]['F1'], reverse=True)
    
    # Identificar gap entre melhor e pior
    if len(modelos_ordenados) >= 2:
        gap = modelos_ordenados[0][1]['F1'] - modelos_ordenados[-1][1]['F1']
        logger.info(f"\n📊 Gap entre melhor e pior modelo: {gap*100:.1f} pontos percentuais")
    
    # Contar modelos por faixa
    modelos_excelentes = sum(1 for _, m in modelos_ordenados if m['F1'] >= 0.80)
    modelos_bons = sum(1 for _, m in modelos_ordenados if 0.70 <= m['F1'] < 0.80)
    modelos_regulares = sum(1 for _, m in modelos_ordenados if 0.50 <= m['F1'] < 0.70)
    modelos_ruins = sum(1 for _, m in modelos_ordenados if m['F1'] < 0.50)
    
    logger.info(f"\n📈 Distribuição de performance:")
    logger.info(f"   • Excelentes (F1 ≥ 80%): {modelos_excelentes} modelo(s)")
    logger.info(f"   • Bons (70% ≤ F1 < 80%): {modelos_bons} modelo(s)")
    logger.info(f"   • Regulares (50% ≤ F1 < 70%): {modelos_regulares} modelo(s)")
    logger.info(f"   • Insuficientes (F1 < 50%): {modelos_ruins} modelo(s)")
    
    for i, (nome, metricas) in enumerate(modelos_ordenados, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}º"
        
        # Classificação dinâmica
        if metricas['F1'] >= 0.80:
            status = "✅ EXCELENTE"
        elif metricas['F1'] >= 0.70:
            status = "🟡 BOM"
        elif metricas['F1'] >= 0.50:
            status = "🟠 REGULAR"
        else:
            status = "❌ INSUFICIENTE"
        
        logger.info(f"\n{emoji} LUGAR: {nome}")
        logger.info(f"   • Status: {status}")
        logger.info(f"   • F1-Score: {metricas['F1']*100:.1f}%")
        logger.info(f"   • Detecta {metricas['Recall']*100:.1f}% das fraudes")
        logger.info(f"   • {metricas['Precision']*100:.1f}% de precisão nos alertas")
        logger.info(f"   • ROC-AUC: {metricas['ROC-AUC']*100:.1f}%")
    
    # =========================================================================
    # 5. RECOMENDAÇÕES 100% DINÂMICAS
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("📋 RECOMENDAÇÕES ESTRATÉGICAS (GERADAS AUTOMATICAMENTE)")
    logger.info("="*80)
    
    # CURTO PRAZO - Baseado nas métricas reais
    logger.info("\n🔴 CURTO PRAZO (Próximos 30 dias):")
    
    recomendacoes_curto = []
    
    # Recomendação 1: Baseada no melhor modelo
    recomendacoes_curto.append(f"   1. Implementar modelo {nome_melhor} em ambiente de homologação")
    
    # Recomendação 2: Baseada em precision
    if metricas_melhor['Precision'] < 0.70:
        recomendacoes_curto.append(f"   2. Ajustar threshold de 0.5 para reduzir {falsos_positivos:,} falsos positivos")
    else:
        recomendacoes_curto.append(f"   2. Monitorar estabilidade do threshold atual (precision: {metricas_melhor['Precision']*100:.1f}%)")
    
    # Recomendação 3: Baseada em recall
    if metricas_melhor['Recall'] < 0.85:
        recomendacoes_curto.append(f"   3. Investigar {fraudes_perdidas:,} fraudes não detectadas para identificar padrões")
    else:
        recomendacoes_curto.append(f"   3. Validar sistema com dados de produção em shadow mode")
    
    for rec in recomendacoes_curto:
        logger.info(rec)
    
    # MÉDIO PRAZO - Baseado nas técnicas disponíveis
    logger.info("\n🟡 MÉDIO PRAZO (Próximos 90 dias):")
    
    recomendacoes_medio = []
    
    # Recomendação baseada no número de modelos
    if len(resultados) > 5:
        recomendacoes_medio.append(f"   1. Implementar ensemble dos top 3 modelos (gap de {gap*100:.1f}% pode ser reduzido)")
    else:
        recomendacoes_medio.append(f"   1. Expandir suite de modelos para aumentar robustez")
    
    # Recomendação baseada em features
    recomendacoes_medio.append(f"   2. Coletar feedback dos analistas sobre os {falsos_positivos:,} falsos positivos")
    
    # Recomendação baseada em drift
    recomendacoes_medio.append(f"   3. Implementar monitoramento de drift (dados atuais: {total_transacoes:,} transações)")
    
    for rec in recomendacoes_medio:
        logger.info(rec)
    
    # LONGO PRAZO - Baseado na eficiência
    logger.info("\n🟢 LONGO PRAZO (Próximos 6 meses):")
    
    recomendacoes_longo = []
    
    if eficiencia >= 70:
        recomendacoes_longo.append(f"   1. Escalar sistema para processamento em tempo real (eficiência atual: {eficiencia:.1f}%)")
    else:
        recomendacoes_longo.append(f"   1. Melhorar eficiência de {eficiencia:.1f}% para pelo menos 80%")
    
    recomendacoes_longo.append(f"   2. Desenvolver sistema de auto-retreinamento baseado em feedback")
    recomendacoes_longo.append(f"   3. Expandir para detecção de novos tipos de fraude")
    
    for rec in recomendacoes_longo:
        logger.info(rec)
    
    # =========================================================================
    # 6. ALERTAS AUTOMÁTICOS
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("🚨 ALERTAS AUTOMÁTICOS")
    logger.info("="*80)
    
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
        logger.info("✅ Nenhum alerta crítico - sistema em bom estado")
    else:
        for alerta in alertas:
            logger.info(alerta)
    
    logger.info("\n" + "="*80)
    logger.info("✅ RELATÓRIO GERADO COM SUCESSO (100% DINÂMICO)")
    logger.info("="*80 + "\n")


def analisar_distribuicao_fraudes(df: pd.DataFrame):
    """
    Analisa padrões de fraude e gera insights visuais 100% dinâmicos.
    """
    logger.info("\n" + "="*80)
    logger.info("🔍 ANÁLISE DE PADRÕES DE FRAUDE (100% DINÂMICO)")
    logger.info("="*80)
    
    criar_diretorio('results/figures')
    
    fraudes = df[df['Class'] == 1]
    normais = df[df['Class'] == 0]
    
    # =========================================================================
    # 1. Análise temporal dinâmica
    # =========================================================================
    if 'Hour' in df.columns:
        logger.info("\n⏰ PADRÃO TEMPORAL DAS FRAUDES:")
        
        fraudes_por_hora = fraudes['Hour'].value_counts().sort_index()
        normais_por_hora = normais['Hour'].value_counts().sort_index()
        
        taxa_fraude_hora = (fraudes_por_hora / (fraudes_por_hora + normais_por_hora)).fillna(0) * 100
        
        hora_mais_perigosa = taxa_fraude_hora.idxmax()
        taxa_maxima = taxa_fraude_hora.max()
        
        hora_mais_segura = taxa_fraude_hora.idxmin()
        taxa_minima = taxa_fraude_hora.min()
        
        # Identificar horas acima da média
        horas_perigosas = taxa_fraude_hora[taxa_fraude_hora > taxa_fraude_hora.mean() * 1.5].index.tolist()
        
        logger.info(f"   • Hora mais perigosa: {int(hora_mais_perigosa)}h ({taxa_maxima:.2f}% de fraude)")
        logger.info(f"   • Hora mais segura: {int(hora_mais_segura)}h ({taxa_minima:.2f}% de fraude)")
        logger.info(f"   • Total de fraudes: {len(fraudes):,}")
        logger.info(f"   • Horas críticas identificadas: {len(horas_perigosas)}")
        
        if horas_perigosas:
            horas_str = ', '.join([f'{int(h)}h' for h in horas_perigosas[:5]])
            logger.info(f"   • Top horas perigosas: {horas_str}")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        axes[0].plot(fraudes_por_hora.index, fraudes_por_hora.values, 'o-', label='Fraudes', color='red', linewidth=2)
        axes[0].plot(normais_por_hora.index, normais_por_hora.values, 'o-', label='Normais', color='blue', linewidth=2, alpha=0.5)
        axes[0].set_xlabel('Hora do Dia', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Quantidade de Transações', fontsize=12, fontweight='bold')
        axes[0].set_title('Distribuição de Transações por Hora', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)
        
        axes[1].bar(taxa_fraude_hora.index, taxa_fraude_hora.values, color='coral', alpha=0.7)
        axes[1].axhline(taxa_fraude_hora.mean(), color='red', linestyle='--', linewidth=2, label=f'Média: {taxa_fraude_hora.mean():.2f}%')
        axes[1].set_xlabel('Hora do Dia', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Taxa de Fraude (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Taxa de Fraude por Hora', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('results/figures/padrao_temporal_fraudes.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("   → Gráfico salvo em results/figures/padrao_temporal_fraudes.png")
    
    # =========================================================================
    # 2. Análise de valor dinâmica
    # =========================================================================
    logger.info("\n💰 PADRÃO DE VALORES:")
    
    valor_medio_fraude = fraudes['Amount'].mean()
    valor_medio_normal = normais['Amount'].mean()
    valor_mediano_fraude = fraudes['Amount'].median()
    valor_mediano_normal = normais['Amount'].median()
    
    razao_media = valor_medio_fraude / valor_medio_normal if valor_medio_normal > 0 else 0
    
    logger.info(f"   • Valor médio de fraude: R$ {valor_medio_fraude:.2f}")
    logger.info(f"   • Valor médio de normal: R$ {valor_medio_normal:.2f}")
    logger.info(f"   • Fraudes são {razao_media:.1f}x maiores em média")
    
    logger.info(f"\n   • Valor mediano de fraude: R$ {valor_mediano_fraude:.2f}")
    logger.info(f"   • Valor mediano de normal: R$ {valor_mediano_normal:.2f}")
    
    # Identificar outliers
    percentil_95_fraude = fraudes['Amount'].quantile(0.95)
    percentil_99_fraude = fraudes['Amount'].quantile(0.99)
    
    logger.info(f"\n   • Percentil 95 das fraudes: R$ {percentil_95_fraude:.2f}")
    logger.info(f"   • Percentil 99 das fraudes: R$ {percentil_99_fraude:.2f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    bp = axes[0].boxplot([normais['Amount'].clip(0, 1000), fraudes['Amount'].clip(0, 1000)], 
                          patch_artist=True)
    axes[0].set_xticklabels(['Normal', 'Fraude'])
    axes[0].set_ylabel('Valor da Transação (R$)', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribuição de Valores (limitado a R$ 1.000)', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='y')
    
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[1].hist(normais['Amount'].clip(0, 1000), bins=50, alpha=0.5, label='Normal', color='blue', density=True)
    axes[1].hist(fraudes['Amount'].clip(0, 1000), bins=50, alpha=0.7, label='Fraude', color='red', density=True)
    axes[1].set_xlabel('Valor da Transação (R$)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Densidade', fontsize=12, fontweight='bold')
    axes[1].set_title('Distribuição de Valores', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/padrao_valores_fraudes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("   → Gráfico salvo em results/figures/padrao_valores_fraudes.png")
    
    # =========================================================================
    # 3. Insights dinâmicos
    # =========================================================================
    logger.info("\n💡 INSIGHTS PRINCIPAIS (GERADOS AUTOMATICAMENTE):")
    
    insights = []
    
    if razao_media > 2:
        insights.append(f"   ✅ Fraudes têm valores {razao_media:.1f}x maiores - implementar alertas por valor")
    elif razao_media > 1.5:
        insights.append(f"   ✅ Fraudes têm valores {razao_media:.1f}x maiores - considerar threshold de valor")
    else:
        insights.append(f"   ⚠️ Valores de fraude e normal são similares ({razao_media:.1f}x) - valor sozinho não é bom indicador")
    
    if 'Hour' in df.columns and horas_perigosas:
        insights.append(f"   ✅ {len(horas_perigosas)} horas críticas identificadas - reforçar monitoramento")
    
    if valor_mediano_fraude > valor_mediano_normal * 3:
        insights.append(f"   ✅ Mediana de fraude é {valor_mediano_fraude/valor_mediano_normal:.1f}x maior - padrão forte")
    
    for insight in insights:
        logger.info(insight)
    
    logger.info("\n" + "="*80 + "\n")