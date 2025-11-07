# src/model.py

from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

def load_xception_model(input_shape=(224, 224, 3), num_classes=2, trainable=False):
    """
    Charge le modèle Xception pré-entraîné avec des poids ImageNet.
    Les couches de base sont gelées par défaut pour le Transfer Learning.
    
    Args:
        input_shape (tuple): dimensions des images d'entrée.
        num_classes (int): nombre de classes pour la classification.
        trainable (bool): si True, permet le fine-tuning des couches de base.
        
    Returns:
        model (tf.keras.Model): modèle Keras prêt pour ajout de couches custom.
    """
    # Charger le modèle Xception pré-entraîné sans la tête
    base_model = Xception(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    # Geler les couches de base si nécessaire
    base_model.trainable = trainable
    
    # Ajouter des couches globales pour préparation à la classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Hiba: Ici les couches Dense sont ajoutées pour extraire des features spécifiques à notre dataset .
    # j'ai choisis 512 puis 256 neurones pour capturer progressivement des patterns complexes :)
    x = Dense(512, activation='relu')(x)
    # J'ajoute le dropout :)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    # J'applique le dropout aux couches Dense pour éviter l'overfitting et améliorer la généralisation du modèle sur d'autres images hors notre dataset de train.
    x = Dropout(0.3)(x)


    # Couche finale pour classification binaire (num_classes)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Créer le modèle complet
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

# Exemple d'utilisation (pour debug ou notebook)
if __name__ == "__main__":
    model = load_xception_model()
    model.summary()