"""
Module: preprocess.py
Description: Fonctions de prétraitement pour les images du dataset BUSI.
"""

# -----------------------------
# Imports
# -----------------------------
import tensorflow as tf

# -----------------------------
# Configuration de base
# -----------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = 'Dataset_BUSI'  

# -----------------------------
# Générateur pour l'entraînement
# -----------------------------
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    f'{DATASET_DIR}/train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# -----------------------------
# Générateur pour la validation
# -----------------------------
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    f'{DATASET_DIR}/validation',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# -----------------------------
# Générateur pour le test
# -----------------------------
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    f'{DATASET_DIR}/test',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # Important pour l'évaluation
)

print("Préprocessing prêt : train, validation et test chargés !")
