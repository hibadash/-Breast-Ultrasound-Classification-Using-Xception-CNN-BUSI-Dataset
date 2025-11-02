"""
Module: model.py
Description: Définition du modèle CNN basé sur Xception.
"""

from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

def build_xception_model(input_shape=(224, 224, 3), num_classes=3):
    """Construit et compile un modèle basé sur Xception."""
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    for layer in base_model.layers:
        layer.trainable = False  # fine-tuning plus tard

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
