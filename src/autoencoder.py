"""
Módulo de Autoencoder para Detecção de Anomalias.
Versão MELHORADA com arquitetura otimizada.
"""
# pylint: disable=import-error
# pyright: reportMissingModuleSource=false
import numpy as np

# Imports diretos do TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    Model = keras.Model
    Input = keras.Input
    Dense = keras.layers.Dense
    Dropout = keras.layers.Dropout
    BatchNormalization = keras.layers.BatchNormalization
    EarlyStopping = keras.callbacks.EarlyStopping
    ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau
    Adam = keras.optimizers.Adam
except ImportError:
    raise ImportError("TensorFlow não está instalado. Execute: pip install tensorflow")

from src.utils import logger


def construir_autoencoder(input_dim: int):
    """
    Constrói Autoencoder com arquitetura otimizada.
    """
    logger.info(f"Construindo Autoencoder com {input_dim} features...")
    
    # Encoder
    input_layer = Input(shape=(input_dim,))
    
    x = Dense(128, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Bottleneck
    bottleneck = Dense(16, activation='relu')(x)
    
    # Decoder
    x = Dense(32, activation='relu')(bottleneck)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    
    output_layer = Dense(input_dim, activation='linear')(x)
    
    autoencoder = Model(input_layer, output_layer)
    
    autoencoder.compile(
        optimizer=Adam(learning_rate=0.001, weight_decay=1e-5),
        loss='mse'
    )
    
    logger.info("Autoencoder construído com sucesso")
    return autoencoder


def treinar_autoencoder(
    X_train_normal: np.ndarray,
    X_val_normal: np.ndarray,
    epochs: int = 100,
    batch_size: int = 512
) -> tuple:
    """
    Treina Autoencoder com early stopping avançado.
    """
    logger.info("="*60)
    logger.info("TREINANDO AUTOENCODER")
    logger.info("="*60)
    logger.info(f"Dados de treino: {X_train_normal.shape}")
    logger.info(f"Dados de validação: {X_val_normal.shape}")
    
    input_dim = X_train_normal.shape[1]
    model = construir_autoencoder(input_dim)
    
    # Callbacks avançados
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    history = model.fit(
        X_train_normal, X_train_normal,
        validation_data=(X_val_normal, X_val_normal),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    logger.info(f"Treinamento concluído em {len(history.history['loss'])} épocas")
    logger.info(f"Loss final treino: {history.history['loss'][-1]:.6f}")
    logger.info(f"Loss final validação: {history.history['val_loss'][-1]:.6f}")
    
    return model, history


def calcular_erro_reconstrucao(model, X: np.ndarray) -> np.ndarray:
    """Calcula erro de reconstrução (MSE)."""
    X_reconstructed = model.predict(X, verbose=0)
    mse = np.mean(np.power(X - X_reconstructed, 2), axis=1)
    return mse


def detectar_anomalias_autoencoder(
    model,
    X_train_normal: np.ndarray,
    X_test: np.ndarray,
    threshold_percentil: float = 99.5
) -> tuple:
    """
    Detecta anomalias com threshold ajustável.
    """
    logger.info(f"Detectando anomalias (percentil: {threshold_percentil})...")
    
    train_errors = calcular_erro_reconstrucao(model, X_train_normal)
    threshold = np.percentile(train_errors, threshold_percentil)
    logger.info(f"Threshold: {threshold:.6f}")
    
    test_errors = calcular_erro_reconstrucao(model, X_test)
    
    y_pred = (test_errors > threshold).astype(int)
    
    # Probabilidades normalizadas
    test_errors_clipped = np.clip(test_errors - threshold, -500, 500)
    y_prob = 1 / (1 + np.exp(-test_errors_clipped))
    
    n_anomalias = y_pred.sum()
    logger.info(f"Anomalias detectadas: {n_anomalias} ({n_anomalias/len(y_pred)*100:.2f}%)")
    logger.info(f"Erro médio: {test_errors.mean():.6f}, Max: {test_errors.max():.6f}")
    
    return y_pred, y_prob, threshold


def plot_distribuicao_erros(
    model,
    X_train_normal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    nome_modelo: str = "Autoencoder"
):
    """Plota distribuição de erros."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.utils import criar_diretorio
    
    criar_diretorio('results/figures')
    
    train_errors = calcular_erro_reconstrucao(model, X_train_normal)
    test_errors = calcular_erro_reconstrucao(model, X_test)
    
    test_errors_normal = test_errors[y_test == 0]
    test_errors_fraude = test_errors[y_test == 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histograma
    ax = axes[0]
    ax.hist(train_errors, bins=100, alpha=0.5, label='Treino (Normal)', color='blue', density=True)
    ax.hist(test_errors_normal, bins=100, alpha=0.5, label='Teste (Normal)', color='green', density=True)
    ax.hist(test_errors_fraude, bins=50, alpha=0.7, label='Teste (Fraude)', color='red', density=True)
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
    ax.set_xlabel('Erro de Reconstrução (MSE)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Densidade', fontsize=12, fontweight='bold')
    ax.set_title(f'Distribuição de Erros - {nome_modelo}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, np.percentile(test_errors, 99))
    ax.grid(alpha=0.3)
    
    # Boxplot
    ax = axes[1]
    data_to_plot = [train_errors, test_errors_normal, test_errors_fraude]
    bp = ax.boxplot(data_to_plot, patch_artist=True, showfliers=False)
    
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xticklabels(['Treino\n(Normal)', 'Teste\n(Normal)', 'Teste\n(Fraude)'])
    ax.axhline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax.set_ylabel('Erro de Reconstrução (MSE)', fontsize=12, fontweight='bold')
    ax.set_title('Comparação de Erros', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'results/figures/autoencoder_error_distribution.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Gráfico de erros salvo em results/figures/")