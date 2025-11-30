"""
Module: train.py
Description: Script complet d'entraînement du modèle Xception avec gestion des hyperparamètres,
             callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau), et sauvegarde de l'historique.
"""

import os
import sys
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
    TensorBoard,
    Callback
)
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Ajouter le chemin du projet pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_xception_model
from data.preprocess import (
    train_generator,
    val_generator,
    IMAGE_SIZE,
    BATCH_SIZE,
    DATASET_DIR,
    SEED
)


# =======================
# CONFIGURATION DES HYPERPARAMÈTRES
# =======================
class TrainingConfig:
    """Configuration centralisée pour l'entraînement"""
    
    # Hyperparamètres d'entraînement
    LEARNING_RATE = 1e-4  # Taux d'apprentissage initial
    BATCH_SIZE = BATCH_SIZE  # Taille du batch (32 par défaut)
    EPOCHS = 50  # Nombre maximum d'epochs
    PATIENCE_EARLY_STOPPING = 12  # Patience pour EarlyStopping
    PATIENCE_LR_REDUCTION = 6  # Patience pour réduire le learning rate
    
    # Fine-tuning (optionnel)
    FINE_TUNE_AFTER_EPOCHS = 6  # Commencer le fine-tuning tôt pour améliorer la généralisation
    FINE_TUNE_LEARNING_RATE = 1e-5  # Learning rate plus faible pour fine-tuning
    
    # Chemins de sauvegarde
    RESULTS_DIR = 'results'
    MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, 'model_xception_best.h5')
    MODEL_FINAL_PATH = os.path.join(RESULTS_DIR, 'model_xception_final.h5')
    HISTORY_SAVE_PATH = os.path.join(RESULTS_DIR, 'training_history.json')
    CSV_LOG_PATH = os.path.join(RESULTS_DIR, 'training_log.csv')
    TENSORBOARD_LOG_DIR = os.path.join(RESULTS_DIR, 'tensorboard_logs')
    
    GLOBAL_SEED = SEED
    VERBOSE = 1  # Affichage détaillé (0=silencieux, 1=progress bar, 2=une ligne par epoch)


class FineTuneCallback(Callback):
    """Callback pour déclencher le fine-tuning au bon moment."""

    def __init__(self, unfreeze_epoch: int):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self._has_unfroze = False

    def on_epoch_begin(self, epoch, logs=None):
        if not self._has_unfroze and (epoch + 1) == self.unfreeze_epoch:
            unfreeze_model_for_fine_tuning(self.model, epoch + 1)
            self._has_unfroze = True


# =======================
# FONCTIONS UTILITAIRES
# =======================
def create_results_directory():
    """Crée le dossier results s'il n'existe pas"""
    os.makedirs(TrainingConfig.RESULTS_DIR, exist_ok=True)
    os.makedirs(TrainingConfig.TENSORBOARD_LOG_DIR, exist_ok=True)
    print(f"Dossier '{TrainingConfig.RESULTS_DIR}' prêt")


def set_global_seed(seed: int):
    """Initialise de manière déterministe toutes les librairies utilisées."""

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compute_balanced_class_weights(generator):
    """Calcule les poids de classe équilibrés à partir d'un générateur Keras."""

    class_ids = np.arange(len(generator.class_indices))
    weights = compute_class_weight(class_weight='balanced', classes=class_ids, y=generator.classes)
    return {idx: float(weight) for idx, weight in zip(class_ids, weights)}


def get_class_names():
    """Récupère les noms des classes depuis le générateur"""
    class_names = sorted(list(train_generator.class_indices.keys()))
    return class_names


def print_training_info():
    """Affiche les informations sur le dataset et la configuration"""
    print("\n" + "="*70)
    print("CONFIGURATION D'ENTRAÎNEMENT - XCEPTION CNN")
    print("="*70)
    
    # Informations sur le dataset
    print(f"\n DATASET:")
    print(f"   - Taille des images: {IMAGE_SIZE}")
    print(f"   - Batch size: {TrainingConfig.BATCH_SIZE}")
    print(f"   - Images d'entraînement: {train_generator.samples}")
    print(f"   - Images de validation: {val_generator.samples}")
    print(f"   - Classes: {get_class_names()}")
    print(f"   - Mapping des classes: {train_generator.class_indices}")
    
    # Hyperparamètres
    print(f"\n HYPERPARAMÈTRES:")
    print(f"   - Learning rate initial: {TrainingConfig.LEARNING_RATE}")
    print(f"   - Epochs maximum: {TrainingConfig.EPOCHS}")
    print(f"   - Early stopping patience: {TrainingConfig.PATIENCE_EARLY_STOPPING}")
    print(f"   - LR reduction patience: {TrainingConfig.PATIENCE_LR_REDUCTION}")
    print(f"   - Fine-tuning après: {TrainingConfig.FINE_TUNE_AFTER_EPOCHS} epochs")
    
    # Chemins de sauvegarde
    print(f"\n SAUVEGARDE:")
    print(f"   - Meilleur modèle: {TrainingConfig.MODEL_SAVE_PATH}")
    print(f"   - Modèle final: {TrainingConfig.MODEL_FINAL_PATH}")
    print(f"   - Historique: {TrainingConfig.HISTORY_SAVE_PATH}")
    print(f"   - Log CSV: {TrainingConfig.CSV_LOG_PATH}")
    print(f"   - TensorBoard: {TrainingConfig.TENSORBOARD_LOG_DIR}")
    print("="*70 + "\n")


# =======================
# CALLBACKS
# =======================
def create_callbacks():
    """Crée tous les callbacks nécessaires pour l'entraînement"""
    
    callbacks = [
        # 1. Sauvegarde du meilleur modèle (basé sur val_accuracy)
        ModelCheckpoint(
            filepath=TrainingConfig.MODEL_SAVE_PATH,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            save_fmt='h5'
        ),
        
        # 2. Early Stopping pour éviter l'overfitting
        EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=TrainingConfig.PATIENCE_EARLY_STOPPING,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.0001
        ),
        
        # 3. Réduction automatique du learning rate
        ReduceLROnPlateau(
            monitor='val_loss',
            mode='min',
            factor=0.3,
            patience=TrainingConfig.PATIENCE_LR_REDUCTION,
            min_lr=1e-6,
            verbose=1
        ),
        
        # 4. Logger CSV pour suivre l'évolution
        CSVLogger(
            filename=TrainingConfig.CSV_LOG_PATH,
            separator=',',
            append=False
        ),
        
        # 5. TensorBoard pour visualisation (optionnel)
        TensorBoard(
            log_dir=TrainingConfig.TENSORBOARD_LOG_DIR,
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch'
        ),

        FineTuneCallback(TrainingConfig.FINE_TUNE_AFTER_EPOCHS)
    ]
    
    return callbacks


# =======================
# FINE-TUNING
# =======================
def unfreeze_model_for_fine_tuning(model, epoch):
    """
    Débloque certaines couches du modèle de base pour le fine-tuning.
    
    Args:
        model: Modèle Keras
        epoch: Numéro de l'epoch actuel
    """
    if epoch == TrainingConfig.FINE_TUNE_AFTER_EPOCHS:
        print(f"\n DÉBUT DU FINE-TUNING (Epoch {epoch})")
        print("    Déblocage des dernières couches de Xception...")
        
        # Débloquer les dernières couches de Xception 
        base_model = model.layers[0]
        base_model.trainable = True
        
        # Geler les premières couches 
        # Débloquer seulement les 50 dernières couches
        for layer in base_model.layers[:-50]:
            layer.trainable = False
        
        # Recompiler avec un learning rate plus faible
        model.compile(
            optimizer=Adam(learning_rate=TrainingConfig.FINE_TUNE_LEARNING_RATE),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=['accuracy']
        )
        
        print(f" Learning rate réduit à: {TrainingConfig.FINE_TUNE_LEARNING_RATE}")
        print(f" Couches débloquées: {sum([1 for l in base_model.layers if l.trainable])}/{len(base_model.layers)}")
        print()


# =======================
# VISUALISATION
# =======================
def plot_training_history(history):
    """
    Visualise l'historique d'entraînement (loss et accuracy).
    
    Args:
        history: Objet History retourné par model.fit()
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Loss
    axes[0].plot(history.history['loss'], label='Train Loss', marker='o')
    axes[0].plot(history.history['val_loss'], label='Validation Loss', marker='s')
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(TrainingConfig.RESULTS_DIR, 'training_history_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Graphique sauvegardé: {save_path}")
    plt.close()


def save_training_history(history):
    """Sauvegarde l'historique d'entraînement en JSON"""
    # Convertir les valeurs numpy en listes Python pour JSON
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(v) for v in values]
    
    with open(TrainingConfig.HISTORY_SAVE_PATH, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f" Historique sauvegardé: {TrainingConfig.HISTORY_SAVE_PATH}")


# =======================
# FONCTION PRINCIPALE D'ENTRAÎNEMENT
# =======================
def train_model():
    """
    Fonction principale pour entraîner le modèle Xception.
    
    Returns:
        model: Modèle entraîné
        history: Historique d'entraînement
    """
    # 1. Préparation
    set_global_seed(TrainingConfig.GLOBAL_SEED)
    create_results_directory()
    print_training_info()
    
    # 2. Charger le modèle
    print("Chargement du modèle Xception...")
    model = load_xception_model(
        input_shape=(*IMAGE_SIZE, 3),
        num_classes=len(get_class_names()),
        trainable=False  # Commencer avec les couches gelées
    )
    print("Modèle chargé avec succès!")
    print(f"   - Paramètres totaux: {model.count_params():,}")
    print(f"   - Paramètres entraînables: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    print()
    
    # 3. Créer les callbacks
    callbacks = create_callbacks()
    print(f" {len(callbacks)} callbacks configurés\n")
    
    # 4. Entraînement
    print("DÉBUT DE L'ENTRAÎNEMENT\n")

    # Calcul des poids de classes pour compenser les déséquilibres
    class_weights = compute_balanced_class_weights(train_generator)
    print(f"Poids de classe utilisés: {class_weights}\n")

    train_generator.reset()
    val_generator.reset()

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        epochs=TrainingConfig.EPOCHS,
        callbacks=callbacks,
        verbose=TrainingConfig.VERBOSE,
        class_weight=class_weights
    )
    
    print("\n ENTRAÎNEMENT TERMINÉ!\n")
    
    # 5. Sauvegarder le modèle final
    print("Sauvegarde du modèle final...")
    model.save(TrainingConfig.MODEL_FINAL_PATH)
    print(f"Modèle final sauvegardé: {TrainingConfig.MODEL_FINAL_PATH}\n")
    
    # 6. Sauvegarder et visualiser l'historique
    save_training_history(history)
    plot_training_history(history)
    
    # 7. Résumé final
    print("\n" + "="*70)
    print("RÉSUMÉ DE L'ENTRAÎNEMENT")
    print("="*70)
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    
    print(f"   - Epochs effectués: {len(history.history['loss'])}")
    print(f"   - Meilleure validation accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"   - Accuracy finale (train): {final_train_acc:.4f}")
    print(f"   - Accuracy finale (validation): {final_val_acc:.4f}")
    print(f"   - Loss finale (train): {history.history['loss'][-1]:.4f}")
    print(f"   - Loss finale (validation): {history.history['val_loss'][-1]:.4f}")
    print("="*70 + "\n")
    
    return model, history


# =======================
# EXÉCUTION
# =======================
if __name__ == "__main__":
    # Désactiver les warnings TensorFlow si nécessaire
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Vérifier que le dataset existe
    if not os.path.exists(DATASET_DIR):
        print(f"Erreur: Le dossier '{DATASET_DIR}' n'existe pas!")
        print(f"Assurez-vous que le dataset est dans: {os.path.abspath(DATASET_DIR)}")
        sys.exit(1)
    
    # Lancer l'entraînement
    try:
        model, history = train_model()
        print("Entraînement réussi!")
    except Exception as e:
        print(f"\n Erreur lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
