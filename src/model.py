# src/model.py

from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def load_xception_model(input_shape=(224, 224, 3), num_classes=3, trainable=False):
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

    # Hiba: Ici les couches Dense sont ajoutées pour extraire des features spécifiques à notre dataset.
    # J'ai choisi 512 puis 256 neurones pour capturer progressivement des patterns complexes :)
    x = Dense(512, activation='relu')(x) # Activation est RELU
    # Normalisation des activations de la couche Dense pour stabiliser et accélérer l'entraînement
    x = BatchNormalization()(x)
    # J'ajoute le dropout :)
    x = Dropout(0.2)(x)

    x = Dense(256, activation='relu')(x) 
    # Normalisation après la couche Dense 2
    x = BatchNormalization()(x)
    # J'applique le dropout aux couches Dense pour éviter l'overfitting et améliorer la généralisation
    x = Dropout(0.25)(x)

    # On ajoute une couche finale pour classification de 3 classes (bénign, normal, malin)
    # On a utilisé Softmax pour obtenir des probabilités pour chaque classe (le cas multi-classes)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Remarque : GlobalAveragePooling2D résume chaque feature map en une seule valeur,
    # ce qui réduit les paramètres et stabilise l’entraînement en conservant l’information essentielle.
    # Donc on n’a pas besoin de FLATTEN ici.

    # On crée le modèle final 
    model = Model(inputs=base_model.input, outputs=predictions) 

    # Compiler le modèle pour multi-classes
    model.compile(
        optimizer=Adam(learning_rate=1e-4),  # Optimizer Adam avec learning rate 0.0001
        loss='categorical_crossentropy',     # Cross-entropy pour classification multi-classes
        metrics=['accuracy']                 # Suivi de la précision
    )

    return model

# Exemple d'utilisation (pour debug ou notebook)
if __name__ == "__main__":
    model = load_xception_model()
    # On vérifie la compilation du modèle (information utile pour debug)
    print(model.optimizer)  # Optimiseur utilisé
    print(model.loss)       # Fonction de perte utilisée
    print(model.metrics)    # Quelle métrique est suivie
    model.summary()
