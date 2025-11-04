"""
Module: preprocess.py
Description: Fonctions de prétraitement pour les images du dataset BUSI.
"""

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_image(path):
    """Charge une image et la convertit en RGB."""
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_image(img, size=(224, 224)):
    """Redimensionne une image."""
    return cv2.resize(img, size)

def augment_data():
    """Retourne un générateur d’augmentation d’images."""
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        rescale=1./255
    )
# TODO: ajouter fonction pour diviser en train/test et sauvegarder en npz
# test avec colab 
def say_hello(name):
    return f"Hello {name}, preprocessing is ready!"



