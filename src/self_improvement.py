"""
Subsistema de Auto-Aprendizado e Auto-Correção.
Analisa erros do modelo, identifica padrões de falha, 
sugere melhorias e testa alternativas automaticamente.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.utils import logger, criar_diretorio
from datetime import datetime


class SelfImprovementSystem:
    """
    Sistema que aprende com os próprios erros e busca melhorias automáticas.
    """
    
    def __init__(self, modelo, X_train, y_train, X_test, y_test, 
                 nome_modelo="Modelo", threshold=0.5):
        """
        Inicializa o sistema de auto-aprendizado.
        
        Args:
            modelo: Modelo treinado (com predict_proba)
            X_train, y_train: Dados de treino
            X_test, y_test: Dados de teste
            nome_modelo: Nome do modelo para relatórios
            threshold: Threshold atual de decisão
        """
        self.modelo = modelo
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.nome_modelo = nome_modelo
        self.threshold = threshold
        self.historico_melhorias = []
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🧠 INICIALIZANDO SISTEMA DE AUTO-APRENDIZADO")
        logger.info(f"{'='*80}")
        logger.info(f"Modelo analisado: {nome_modelo}")
        logger.info(f"Threshold atual: {threshold}")
    
    def analisar_erros(self):
        """
        Analisa falsos positivos e falsos negativos para identificar padrões de falha.
        """
        logger.info("\n ANALISANDO PADRÕES DE ERRO...")
        
        y_prob = self.modelo.predict_proba(self.X_test)[:, 1]
        y_pred = (y_prob >= self.threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        
        # Identificar instâncias de erro
        falsos_positivos_idx = np.where((y_pred == 1) & (self.y_test == 0))[0]
        falsos_negativos_idx = np.where((y_pred == 0) & (self.y_test == 1))[0]
        
        resultados = {
            'total_erros': fp + fn,
            'falsos_positivos': fp,
            'falsos_negativos': fn,
            'taxa_fp': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'taxa_fn': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'fp_indices': falsos_positivos_idx,
            'fn_indices': falsos_negativos_idx,
            'fp_features': self.X_test.iloc[falsos_positivos_idx] if hasattr(self.X_test, 'iloc') else self.X_test[falsos_positivos_idx],
            'fn_features': self.X_test.iloc[falsos_negativos_idx] if hasattr(self.X_test, 'iloc') else self.X_test[falsos_negativos_idx],
            'fp_probs': y_prob[falsos_positivos_idx],
            'fn_probs': y_prob[falsos_negativos_idx]
        }
        
        logger.info(f"Total de erros: {resultados['total_erros']}")
        logger.info(f"  • Falsos Positivos: {fp} ({resultados['taxa_fp']*100:.2f}% dos normais)")
        logger.info(f"  • Falsos Negativos: {fn} ({resultados['taxa_fn']*100:.2f}% das fraudes)")
        
        return resultados
    
    def identificar_padroes_falha(self, analise_erros):
        """
        Identifica padrões comuns nos erros (ex: valores altos, horários específicos).
        """
        logger.info("\n IDENTIFICANDO PADRÕES DE FALHA...")
        
        padroes = {}
        
        # Analisar falsos positivos
        if len(analise_erros['fp_features']) > 0:
            fp_df = analise_erros['fp_features']
            
            # Padrões numéricos
            if 'Amount' in fp_df.columns:
                padroes['fp_valor_medio'] = fp_df['Amount'].mean()
                padroes['fp_valor_mediana'] = fp_df['Amount'].median()
                logger.info(f"  FP - Valor médio: R$ {padroes['fp_valor_medio']:.2f}")
            
            if 'Hour' in fp_df.columns:
                horas_fp = fp_df['Hour'].value_counts()
                padroes['fp_horas_criticas'] = horas_fp.head(3).to_dict()
                logger.info(f"  FP - Horas com mais erros: {padroes['fp_horas_criticas']}")
            
            if 'TransacoesUltimaHora' in fp_df.columns:
                padroes['fp_freq_media'] = fp_df['TransacoesUltimaHora'].mean()
                logger.info(f"  FP - Frequência média: {padroes['fp_freq_media']:.1f}")
        
        # Analisar falsos negativos
        if len(analise_erros['fn_features']) > 0:
            fn_df = analise_erros['fn_features']
            
            if 'Amount' in fn_df.columns:
                padroes['fn_valor_medio'] = fn_df['Amount'].mean()
                padroes['fn_valor_mediana'] = fn_df['Amount'].median()
                logger.info(f"  FN - Valor médio: R$ {padroes['fn_valor_medio']:.2f}")
            
            if 'Hour' in fn_df.columns:
                horas_fn = fn_df['Hour'].value_counts()
                padroes['fn_horas_criticas'] = horas_fn.head(3).to_dict()
                logger.info(f"  FN - Horas com mais erros: {padroes['fn_horas_criticas']}")
        
        return padroes
    
    def gerar_sugestoes(self, analise_erros, padroes):
        """
        Gera sugestões 100% dinâmicas baseadas nos padrões de erro.
        """
        logger.info("\n💡 GERANDO SUGESTÕES DE MELHORIA")
        
        sugestoes = []
        
        # Calcular impacto real baseado nos dados
        impacto_fp = analise_erros['falsos_positivos'] * 50  # Custo estimado por FP
        impacto_fn = analise_erros['falsos_negativos'] * 122  # Custo estimado por FN
        
        # Sugestão 1: Ajuste de threshold (com impacto calculado)
        if analise_erros['taxa_fp'] > analise_erros['taxa_fn'] * 2:
            novo_threshold = min(0.95, self.threshold + 0.1)
            impacto_estimado = int(impacto_fp * 0.3)  # Redução estimada de 30%
            sugestoes.append({
                'tipo': 'THRESHOLD',
                'acao': 'AUMENTAR_THRESHOLD',
                'justificativa': f"Muitos falsos positivos ({analise_erros['taxa_fp']*100:.1f}%) vs falsos negativos ({analise_erros['taxa_fn']*100:.1f}%). Impacto financeiro: R$ {impacto_fp:,}",
                'parametro': novo_threshold,
                'impacto_esperado': f'Reduzir {analise_erros["falsos_positivos"]} FPs em ~30% (economia estimada: R$ {impacto_estimado:,})'
            })
        elif analise_erros['taxa_fn'] > analise_erros['taxa_fp'] * 2:
            novo_threshold = max(0.1, self.threshold - 0.1)
            impacto_estimado = int(impacto_fn * 0.3)
            sugestoes.append({
                'tipo': 'THRESHOLD',
                'acao': 'DIMINUIR_THRESHOLD',
                'justificativa': f"Muitos falsos negativos ({analise_erros['taxa_fn']*100:.1f}%) vs falsos positivos ({analise_erros['taxa_fp']*100:.1f}%). Impacto financeiro: R$ {impacto_fn:,}",
                'parametro': novo_threshold,
                'impacto_esperado': f'Detectar mais {int(analise_erros["falsos_negativos"] * 0.3)} fraudes (economia estimada: R$ {impacto_estimado:,})'
            })
        
        # Sugestão 2: Features problemáticas (baseada em dados reais)
        if 'fp_valor_medio' in padroes and 'fn_valor_medio' in padroes:
            if padroes['fp_valor_medio'] > padroes['fn_valor_medio'] * 1.5:
                sugestoes.append({
                    'tipo': 'FEATURE',
                    'acao': 'ADICIONAR_FEATURE_VALOR_RELATIVO',
                    'justificativa': f'FP valor médio (R$ {padroes["fp_valor_medio"]:.2f}) é {padroes["fp_valor_medio"]/padroes["fn_valor_medio"]:.1f}x maior que FN (R$ {padroes["fn_valor_medio"]:.2f})',
                    'parametro': 'Amount/mediana_cliente',
                    'impacto_esperado': 'Melhorar discriminação por perfil de gasto individual'
                })
        
        # Sugestão 3: Padrão temporal (baseado em horas reais)
        if 'fp_horas_criticas' in padroes:
            horas_top = list(padroes['fp_horas_criticas'].keys())
            if len(horas_top) > 0:
                total_fps_horas = sum(padroes['fp_horas_criticas'].values())
                sugestoes.append({
                    'tipo': 'REGRA',
                    'acao': 'ADICIONAR_REGRA_TEMPORAL',
                    'justificativa': f'{total_fps_horas} FPs ({total_fps_horas/analise_erros["falsos_positivos"]*100:.1f}%) concentrados nas horas {[int(h) for h in horas_top[:3]]}',
                    'parametro': [int(h) for h in horas_top[:3]],
                    'impacto_esperado': f'Reduzir FPs em {total_fps_horas} transações nestes horários'
                })
        
        # Sugestão 4: Troca de modelo (baseada em performance real)
        f1_atual = self._calcular_f1_atual()
        sugestoes.append({
            'tipo': 'MODELO',
            'acao': 'TESTAR_MODELOS_ALTERNATIVOS',
            'justificativa': f'F1 atual: {f1_atual*100:.1f}%. Testar se outro modelo supera em >2%',
            'parametro': ['XGBoost', 'LightGBM', 'GradientBoosting'],
            'impacto_esperado': f'Possível ganho de 2-5% no F1-Score (de {f1_atual*100:.1f}% para ~{(f1_atual+0.03)*100:.1f}%)'
        })
        
        # Sugestão 5: Re-treinamento (baseado em número real de erros)
        sugestoes.append({
            'tipo': 'TREINAMENTO',
            'acao': 'RE_TREINAR_COM_FOCUS_ERROS',
            'justificativa': f"Re-treinar dando mais peso às {analise_erros['total_erros']} instâncias de erro (impacto: R$ {impacto_fp + impacto_fn:,})",
            'parametro': {'sample_weight': 'inverso_da_confianca'},
            'impacto_esperado': f'Melhorar performance nos {analise_erros["total_erros"]} casos difíceis'
        })
        
        for i, sug in enumerate(sugestoes, 1):
            logger.info(f"  Sugestão {i}: [{sug['tipo']}] {sug['acao']}")
            logger.info(f"    Justificativa: {sug['justificativa']}")
            logger.info(f"    Impacto esperado: {sug['impacto_esperado']}")
        
        return sugestoes
    
    def testar_threshold_alternativos(self, analise_erros):
        """
        Testa automaticamente diferentes thresholds para encontrar o ótimo.
        """
        logger.info("\n🧪 TESTANDO THRESHOLDS ALTERNATIVOS...")
        
        y_prob = self.modelo.predict_proba(self.X_test)[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        resultados = []
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            report = classification_report(self.y_test, y_pred, output_dict=True, zero_division=0)
            
            resultados.append({
                'threshold': thresh,
                'f1': report['1']['f1-score'],
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'accuracy': report['accuracy']
            })
        
        # Encontrar melhor threshold
        melhor = max(resultados, key=lambda x: x['f1'])
        
        logger.info(f"  Threshold atual: {self.threshold:.2f} | F1: {self._calcular_f1_atual():.4f}")
        logger.info(f"  Melhor threshold encontrado: {melhor['threshold']:.2f} | F1: {melhor['f1']:.4f}")
        logger.info(f"  Ganho potencial: +{(melhor['f1'] - self._calcular_f1_atual())*100:.2f}%")
        
        return resultados, melhor
    
    def testar_modelos_alternativos(self):
        """
        Testa automaticamente modelos alternativos para comparar performance.
        """
        logger.info("\n🧪 TESTANDO MODELOS ALTERNATIVOS...")
        
        modelos_teste = {
            'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=1, eval_metric='aucpr'),
            'LightGBM': LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=1, verbose=-1),
            
        }
        
        resultados = []
        
        for nome, modelo in modelos_teste.items():
            logger.info(f"  Testando {nome}...")
            try:
                # Cross-validation rápido
                scores = cross_val_score(modelo, self.X_train, self.y_train, 
                                        cv=3, scoring='f1', n_jobs=1)
                
                # Treinar e avaliar no teste
                modelo.fit(self.X_train, self.y_train)
                y_pred = modelo.predict(self.X_test)
                report = classification_report(self.y_test, y_pred, output_dict=True, zero_division=0)
                
                resultados.append({
                    'modelo': nome,
                    'f1_cv': scores.mean(),
                    'f1_teste': report['1']['f1-score'],
                    'precision': report['1']['precision'],
                    'recall': report['1']['recall'],
                    'std_cv': scores.std()
                })
                
                logger.info(f"    F1 (CV): {scores.mean():.4f} ± {scores.std():.4f}")
                logger.info(f"    F1 (Teste): {report['1']['f1-score']:.4f}")
                
            except Exception as e:
                logger.error(f"    Erro ao testar {nome}: {e}")
        
        # Melhor modelo alternativo
        if resultados:
            melhor = max(resultados, key=lambda x: x['f1_teste'])
            logger.info(f"\n  Melhor modelo alternativo: {melhor['modelo']}")
            logger.info(f"  F1-Score: {melhor['f1_teste']:.4f}")
            logger.info(f"  Ganho vs atual: +{(melhor['f1_teste'] - self._calcular_f1_atual())*100:.2f}%")
            
            return resultados, melhor
        
        return resultados, None
    
    def aplicar_melhoria(self, tipo, parametro):
        """
        Aplica uma melhoria sugerida e retorna o novo estado.
        """
        logger.info(f"\n✅ APLICANDO MELHORIA: {tipo}")
        
        if tipo == 'THRESHOLD':
            self.threshold = parametro
            logger.info(f"  Threshold atualizado para: {parametro}")
            return {'tipo': 'threshold', 'valor': parametro}
        
        elif tipo == 'MODELO':
            # Treinar novo modelo
            if parametro == 'XGBoost':
                novo_modelo = XGBClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=1, eval_metric='aucpr')
            elif parametro == 'LightGBM':
                novo_modelo = LGBMClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=1, verbose=-1)
            elif parametro == 'GradientBoosting':
                novo_modelo = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)
            else:
                return None
            
            novo_modelo.fit(self.X_train, self.y_train)
            self.modelo = novo_modelo
            self.nome_modelo = parametro
            logger.info(f"  Modelo substituído por: {parametro}")
            return {'tipo': 'modelo', 'valor': parametro}
        
        return None
    
    def executar_ciclo_auto_melhoria(self):
        """
        Executa um ciclo completo de auto-análise e melhoria.
        """
        logger.info("\n" + "="*80)
        logger.info("🔄 INICIANDO CICLO DE AUTO-MELHORIA")
        logger.info("="*80)
        
        # Passo 1: Analisar erros atuais
        analise = self.analisar_erros()
        
        # Passo 2: Identificar padrões
        padroes = self.identificar_padroes_falha(analise)
        
        # Passo 3: Gerar sugestões
        sugestoes = self.gerar_sugestoes(analise, padroes)
        
        # Passo 4: Testar thresholds alternativos
        thresholds_resultados, melhor_threshold = self.testar_threshold_alternativos(analise)
        
        # Passo 5: Testar modelos alternativos
        modelos_resultados, melhor_modelo = self.testar_modelos_alternativos()
        
        # Passo 6: Decidir e aplicar melhorias
        melhorias_aplicadas = []
        
        # Aplicar melhor threshold se for significativamente melhor
        f1_atual = self._calcular_f1_atual()
        if melhor_threshold and melhor_threshold['f1'] > f1_atual + 0.01:
            logger.info(f"\n🎯 Threshold atual ({self.threshold:.2f}) está subótimo!")
            logger.info(f"   Aplicando threshold ótimo: {melhor_threshold['threshold']:.2f}")
            resultado = self.aplicar_melhoria('THRESHOLD', melhor_threshold['threshold'])
            melhorias_aplicadas.append(resultado)
        
        # Aplicar melhor modelo se for significativamente melhor
        if melhor_modelo and melhor_modelo['f1_teste'] > f1_atual + 0.02:
            logger.info(f"\n🎯 Modelo atual está sendo superado por {melhor_modelo['modelo']}!")
            logger.info(f"   Aplicando novo modelo...")
            resultado = self.aplicar_melhoria('MODELO', melhor_modelo['modelo'])
            melhorias_aplicadas.append(resultado)
        
        # Passo 7: Registrar no histórico
        self.historico_melhorias.append({
            'timestamp': datetime.now().isoformat(),
            'melhorias_aplicadas': melhorias_aplicadas,
            'f1_antes': f1_atual,
            'f1_depois': self._calcular_f1_atual(),
            'sugestoes_geradas': len(sugestoes)
        })
        
        # Passo 8: Relatório final
        self.gerar_relatorio_auto_melhoria(analise, padroes, sugestoes, 
                                          thresholds_resultados, modelos_resultados,
                                          melhorias_aplicadas)
        
        return {
            'analise_erros': analise,
            'padroes': padroes,
            'sugestoes': sugestoes,
            'melhorias_aplicadas': melhorias_aplicadas,
            'f1_final': self._calcular_f1_atual()
        }
    
    def gerar_relatorio_auto_melhoria(self, analise, padroes, sugestoes, 
                                     thresholds_resultados, modelos_resultados,
                                     melhorias_aplicadas):
        """
        Gera relatório completo do ciclo de auto-melhoria.
        """
        logger.info("\n" + "="*80)
        logger.info("📊 RELATÓRIO DE AUTO-MELHORIA")
        logger.info("="*80)
        
        logger.info(f"\n PERFORMANCE ATUAL:")
        logger.info(f"   F1-Score: {self._calcular_f1_atual():.4f}")
        logger.info(f"   Threshold: {self.threshold:.2f}")
        logger.info(f"   Modelo: {self.nome_modelo}")
        
        logger.info(f"\n🔍 DIAGNÓSTICO DE ERROS:")
        logger.info(f"   Total de erros: {analise['total_erros']}")
        logger.info(f"   Falsos Positivos: {analise['falsos_positivos']}")
        logger.info(f"   Falsos Negativos: {analise['falsos_negativos']}")
        
        if padroes:
            logger.info(f"\n🎯 PADRÕES IDENTIFICADOS:")
            for chave, valor in padroes.items():
                logger.info(f"   • {chave}: {valor}")
        
        logger.info(f"\n💡 SUGESTÕES GERADAS: {len(sugestoes)}")
        for i, sug in enumerate(sugestoes, 1):
            logger.info(f"   {i}. [{sug['tipo']}] {sug['acao']}")
        
        logger.info(f"\n✅ MELHORIAS APLICADAS: {len(melhorias_aplicadas)}")
        for mel in melhorias_aplicadas:
            logger.info(f"   • {mel['tipo']}: {mel['valor']}")
        
        logger.info(f"\n HISTÓRICO DE MELHORIAS: {len(self.historico_melhorias)} ciclos")
        
        logger.info("="*80 + "\n")
    
    def _calcular_f1_atual(self):
        """Calcula F1-Score atual do modelo."""
        from sklearn.metrics import f1_score
        y_prob = self.modelo.predict_proba(self.X_test)[:, 1]
        y_pred = (y_prob >= self.threshold).astype(int)
        return f1_score(self.y_test, y_pred, zero_division=0)
    
    def get_status(self):
        """Retorna status atual do sistema."""
        return {
            'modelo': self.nome_modelo,
            'threshold': self.threshold,
            'f1_score': self._calcular_f1_atual(),
            'ciclos_executados': len(self.historico_melhorias),
            'melhorias_totais': sum(len(h['melhorias_aplicadas']) for h in self.historico_melhorias)
        }