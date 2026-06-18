import sys
import os
import numpy as np
import pandas as pd
import joblib
sys.path.append(os.path.dirname(__file__))

from src.utils import logger, criar_diretorio
from src.data_loader import carregar_dataset
from src.preprocessing import preparar_dados, dividir_e_balancear
from src.models import treinar_isolation_forest
from src.pipelines import (
    criar_pipeline_random_forest, criar_pipeline_xgboost,
    criar_pipeline_lightgbm, criar_pipeline_stacking, 
    criar_pipeline_voting, treinar_pipeline
)
from src.autoencoder import treinar_autoencoder, detectar_anomalias_autoencoder, plot_distribuicao_erros
from src.evaluation import avaliar_modelo
from src.insights import gerar_relatorio_executivo, analisar_distribuicao_fraudes
from src.decision_engine import calcular_threshold_dinamico, sistema_hibrido
from src.relatorio_pdf import gerar_relatorio_pdf
from src.dashboard_profissional import criar_dashboard_onepager
from src.insights_visuais import criar_cards_insights
from src.visualizacoes_interativas import criar_dashboard_interativo_completo
from src.self_improvement import SelfImprovementSystem
from src.model_drift import ModelDriftDetector
from src.shap_avancado import gerar_todos_shap_avancados

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts


def main():
    logger.info("="*80)
    logger.info("🚀 DETECÇÃO DE ANOMALIAS EM TRANSAÇÕES DE CARTÕES DE CRÉDITO")
    logger.info("="*80)
    
    criar_diretorio('results/figures')
    criar_diretorio('models')
    
    # 1. Carregar dados
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = carregar_dataset(url)
    
    # 2. Pré-processamento
    X, y = preparar_dados(df)
    X_train_bal, X_test, y_train_bal, y_test = dividir_e_balancear(X, y)
    
    logger.info(f"\nTotal de features: {X.shape[1]}")
    
    # 3. Escalonar dados para Autoencoder
    logger.info("\n🔧 ESCALONANDO DADOS PARA AUTOENCODER")
    scaler_ae = StandardScaler()
    X_train_bal_scaled = scaler_ae.fit_transform(X_train_bal)
    X_test_scaled = scaler_ae.transform(X_test)
    
    mask_normal = y_train_bal == 0
    X_train_normal_scaled = X_train_bal_scaled[mask_normal]
    X_train_ae, X_val_ae = tts(X_train_normal_scaled, test_size=0.1, random_state=42)
    
    # 4. Treinar Modelos Supervisionados (com Pipeline)
    resultados = {}
    modelos_treinados = {}
    pipelines_salvos = {}
    
    pipelines_config = {
        "Random Forest": criar_pipeline_random_forest(),
        "XGBoost": criar_pipeline_xgboost(),
        "LightGBM": criar_pipeline_lightgbm(),
        "Voting Classifier": criar_pipeline_voting(),
        "Stacking Ensemble": criar_pipeline_stacking()
    }
    
    logger.info("\n" + "="*80)
    logger.info("🤖 TREINANDO MODELOS COM PIPELINE")
    logger.info("="*80)
    
    for nome, pipeline in pipelines_config.items():
        model, y_pred, y_prob = treinar_pipeline(pipeline, X_train_bal, y_train_bal, X_test, nome)
        metricas, pr_auc, roc_auc = avaliar_modelo(y_test, y_pred, y_prob, nome)
        resultados[nome] = {"F1": metricas['f1-score'], "Recall": metricas['recall'], "Precision": metricas['precision'], "PR-AUC": pr_auc, "ROC-AUC": roc_auc}
        
        # Salvar pipeline
        joblib.dump(pipeline, f"models/pipeline_{nome.replace(' ', '_')}.pkl")
        pipelines_salvos[nome] = pipeline
        
        # Guardar o classificador interno
        if 'stacking' in model.named_steps:
            modelos_treinados[nome] = model.named_steps['stacking']
        elif 'voting' in model.named_steps:
            modelos_treinados[nome] = model.named_steps['voting']
        else:
            modelos_treinados[nome] = model.named_steps['classifier']

    # 5. Isolation Forest e Autoencoder
    try:
        model, y_pred, y_prob = treinar_isolation_forest(X_train_bal, X_test)
        metricas, pr_auc, roc_auc = avaliar_modelo(y_test, y_pred, y_prob, "Isolation Forest")
        resultados["Isolation Forest"] = {"F1": metricas['f1-score'], "Recall": metricas['recall'], "Precision": metricas['precision'], "PR-AUC": pr_auc, "ROC-AUC": roc_auc}
        joblib.dump(model, "models/isolation_forest.pkl")
    except Exception as e: 
        logger.error(f"Erro Isolation Forest: {e}")

    try:
        ae_model, _ = treinar_autoencoder(X_train_ae, X_val_ae, epochs=50, batch_size=512)
        y_pred, y_prob, threshold = detectar_anomalias_autoencoder(ae_model, X_train_normal_scaled, X_test_scaled, 99.5)
        plot_distribuicao_erros(ae_model, X_train_normal_scaled, X_test_scaled, y_test, threshold)
        metricas, pr_auc, roc_auc = avaliar_modelo(y_test, y_pred, y_prob, "Autoencoder")
        resultados["Autoencoder"] = {"F1": metricas['f1-score'], "Recall": metricas['recall'], "Precision": metricas['precision'], "PR-AUC": pr_auc, "ROC-AUC": roc_auc}
        modelos_treinados["Autoencoder"] = ae_model
    except Exception as e: 
        logger.error(f"Erro Autoencoder: {e}")

    # =========================================================================
    # TÉCNICAS PROFISSIONAIS: THRESHOLD DINÂMICO E SISTEMA HÍBRIDO
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("⚙️ APLICANDO TÉCNICAS PROFISSIONAIS DE DECISÃO")
    logger.info("="*80)
    
    # Encontrar o melhor modelo supervisionado
    modelos_sup = {k: v for k, v in resultados.items() if k not in ["Isolation Forest", "Autoencoder"]}
    nome_melhor = max(modelos_sup, key=lambda x: modelos_sup[x]['F1'])
    metricas_melhor = modelos_sup[nome_melhor]
    modelo_melhor = modelos_treinados[nome_melhor]
    
    logger.info(f"Melhor modelo supervisionado: {nome_melhor} (F1: {metricas_melhor['F1']*100:.1f}%)")
    
    # Threshold Dinâmico
    y_prob_melhor = modelo_melhor.predict_proba(X_test)[:, 1]
    
    threshold_otimo, custo_minimo = calcular_threshold_dinamico(
        y_test, y_prob_melhor, 
        custo_falso_positivo=50, 
        custo_falso_negativo=122
    )
    
    # Avaliar com threshold dinâmico
    y_pred_dinamico = (y_prob_melhor >= threshold_otimo).astype(int)
    metricas, pr_auc, roc_auc = avaliar_modelo(y_test, y_pred_dinamico, y_prob_melhor, f"{nome_melhor} (Threshold Dinâmico)")
    resultados[f"{nome_melhor} (Threshold Dinâmico)"] = {"F1": metricas['f1-score'], "Recall": metricas['recall'], "Precision": metricas['precision'], "PR-AUC": pr_auc, "ROC-AUC": roc_auc}

    # Sistema Híbrido
    y_pred_hibrido, y_prob_hibrido = sistema_hibrido(X_test, modelo_melhor, threshold_otimo)
    metricas, pr_auc, roc_auc = avaliar_modelo(y_test, y_pred_hibrido, y_prob_hibrido, f"{nome_melhor} (Sistema Híbrido)")
    resultados[f"{nome_melhor} (Sistema Híbrido)"] = {"F1": metricas['f1-score'], "Recall": metricas['recall'], "Precision": metricas['precision'], "PR-AUC": pr_auc, "ROC-AUC": roc_auc}

    # =========================================================================
    # SUBSISTEMA DE AUTO-APRENDIZADO
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("🧠 INICIANDO SUBSISTEMA DE AUTO-APRENDIZADO")
    logger.info("="*80)
    
    sistema_auto = SelfImprovementSystem(
        modelo=modelo_melhor,
        X_train=X_train_bal,
        y_train=y_train_bal,
        X_test=X_test,
        y_test=y_test,
        nome_modelo=nome_melhor,
        threshold=0.5
    )
    
    resultado_auto = sistema_auto.executar_ciclo_auto_melhoria()
    
    if resultado_auto['melhorias_aplicadas']:
        logger.info("\n✨ AUTO-MELHORIA APLICADA COM SUCESSO!")
        logger.info(f"   Novo F1-Score: {resultado_auto['f1_final']:.4f}")
        
        y_prob_novo = sistema_auto.modelo.predict_proba(X_test)[:, 1]
        y_pred_novo = (y_prob_novo >= sistema_auto.threshold).astype(int)
        metricas_novas, pr_auc_novo, roc_auc_novo = avaliar_modelo(
            y_test, y_pred_novo, y_prob_novo, 
            f"{nome_melhor} (Auto-Melhorado)"
        )
        resultados[f"{nome_melhor} (Auto-Melhorado)"] = {
            "F1": metricas_novas['f1-score'],
            "Recall": metricas_novas['recall'],
            "Precision": metricas_novas['precision'],
            "PR-AUC": pr_auc_novo,
            "ROC-AUC": roc_auc_novo
        }
    else:
        logger.info("\n⚠️ Nenhuma melhoria significativa encontrada.")

    # =========================================================================
    # DETECÇÃO DE DRIFT
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("📉 DETECÇÃO DE MODEL DRIFT")
    logger.info("="*80)
    
    try:
        drift_detector = ModelDriftDetector(X_train_bal, threshold=0.05)
        drift_detected, detalhes = drift_detector.detectar_drift_ks_test(X_test)
        
        if drift_detected:
            logger.warning("⚠️ DRIFT DETECTADO: Distribuição dos dados mudou!")
            logger.warning("Recomendação: Considere retreinar o modelo")
        else:
            logger.info("✅ Nenhum drift significativo detectado")
        
        psi_total, psi_detalhes = drift_detector.detectar_drift_psi(X_test)
        logger.info(f"PSI Total: {psi_total:.4f}")
        
        if psi_total > 0.25:
            logger.warning("⚠️ PSI alto indica drift significativo")
        
        drift_detector.gerar_relatorio_drift()
        joblib.dump(drift_detector, "models/drift_detector.pkl")
        
    except Exception as e:
        logger.error(f"Erro ao executar detecção de drift: {e}")

    # =========================================================================
    # SHAP AVANÇADO
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("🔬 SHAP AVANÇADO - EXPLICABILIDADE DETALHADA")
    logger.info("="*80)
    
    try:
        if hasattr(modelo_melhor, 'feature_importances_'):
            feature_names = X_test.columns.tolist() if hasattr(X_test, 'columns') else [f'Feature_{i}' for i in range(X_test.shape[1])]
            gerar_todos_shap_avancados(modelo_melhor, X_test, feature_names)
        else:
            logger.info("⚠️ Modelo não suporta SHAP avançado (sem feature_importances_)")
        
    except Exception as e:
        logger.error(f"Erro ao gerar SHAP avançado: {e}")

    # =========================================================================
    # RESUMO FINAL
    # =========================================================================
    logger.info("\n" + "="*100)
    logger.info("📊 RESUMO FINAL - TODOS OS MODELOS E TÉCNICAS")
    logger.info("="*100)
    logger.info(f"{'Modelo':<35} | {'Recall':<8} | {'Precision':<10} | {'F1':<8} | {'ROC-AUC':<10}")
    logger.info("-"*100)
    for nome, res in sorted(resultados.items(), key=lambda x: x[1]['F1'], reverse=True):
        logger.info(f"{nome:<35} | {res['Recall']:<8.4f} | {res['Precision']:<10.4f} | {res['F1']:<8.4f} | {res['ROC-AUC']:<10.4f}")
    logger.info("="*100)

    # =========================================================================
    # GERAR RELATÓRIOS E VISUALIZAÇÕES
    # =========================================================================
    try:
        gerar_relatorio_executivo(resultados, df)
        analisar_distribuicao_fraudes(df)
        criar_dashboard_onepager(resultados, df, None)
        criar_cards_insights(resultados, df)
        criar_dashboard_interativo_completo(resultados, df, modelo_melhor, X_test, y_test)
        
        # Gerar Relatório PDF (SEM técnicas removidas)
        gerar_relatorio_pdf(
            resultados, df, None,
            resultados_ts=None,  # Removido Time Series CV
            resultados_calib=None,  # Removido Calibration
            resultados_fs=None  # Removido Feature Selection
        )
                
        # Salvar dados para o Streamlit
        dados_streamlit = {
            'resultados': resultados,
            'melhor_modelo': nome_melhor,
            'metricas_melhor': metricas_melhor,
            'threshold_otimo': threshold_otimo
        }
        joblib.dump(dados_streamlit, 'results/dados_streamlit.pkl')
        
        # Salvar scaler e X_test para API
        joblib.dump(scaler_ae, 'models/scaler.pkl')
        joblib.dump(X_test, 'models/X_test.pkl')
        joblib.dump(y_test, 'models/y_test.pkl')
        
    except Exception as e:
        logger.error(f"Erro ao gerar relatórios: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\n" + "="*80)
    logger.info("✅ PROJETO CONCLUÍDO COM SUCESSO!")
    logger.info("📁 Modelos salvos em: models/")
    logger.info("📊 Dados salvos em: results/")
    logger.info("📈 Verifique results/figures/ para todos os gráficos e PDF")
    logger.info("="*80)

if __name__ == "__main__":
    main()