"""
Module: train.py
Description: Script d'entraînement du modèle Xception.
"""

from src.model import build_xception_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

def train_model(train_data, val_data, save_path="results/model_xception_best.h5"):
    """Entraîne le modèle et sauvegarde le meilleur."""
    model = build_xception_model()

    checkpoint = ModelCheckpoint(
        save_path, monitor='val_accuracy', save_best_only=True, verbose=1
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=25,
        callbacks=[checkpoint, early_stop]
    )
    return model, history
