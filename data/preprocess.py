"""
Module: preprocess.py
Description: Fonctions de prétraitement pour les images du dataset BUSI.
"""

# -----------------------------
# Imports
# -----------------------------
import os
import tensorflow as tf

# -----------------------------
# Configuration de base
# -----------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Chemin du dataset - basé sur l'emplacement de ce fichier
_current_file_dir = os.path.dirname(os.path.abspath(__file__))  
_project_root = os.path.dirname(_current_file_dir) 
DATASET_DIR = os.path.join(_project_root, 'data', 'Dataset_BUSI')

# Debug: Afficher le chemin calculé
print(f"[DEBUG preprocess.py] __file__ = {__file__}")
print(f"[DEBUG preprocess.py] _current_file_dir = {_current_file_dir}")
print(f"[DEBUG preprocess.py] _project_root = {_project_root}")
print(f"[DEBUG preprocess.py] DATASET_DIR = {DATASET_DIR}")
print(f"[DEBUG preprocess.py] DATASET_DIR existe? {os.path.exists(DATASET_DIR)}") 

# -----------------------------
# Générateur pour l'entraînement
# -----------------------------
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    f'{DATASET_DIR}/train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical', 
    shuffle=True
)

# -----------------------------
# Générateur pour la validation
# -----------------------------
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    f'{DATASET_DIR}/validation',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# -----------------------------
# Générateur pour le test
# -----------------------------
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    f'{DATASET_DIR}/test',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False 
)

print("Préprocessing prêt : train, validation et test chargés !")
