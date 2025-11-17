"""
Module: data_augmentation.py
Description: Augmentation de données pour équilibrer le dataset BUSI
             en AJOUTANT des images ET leurs masques associés.
"""

# -----------------------------
# Imports
# -----------------------------
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import shutil

# -----------------------------
# Configuration
# -----------------------------
IMAGE_SIZE = (224, 224)
DATASET_DIR = 'Dataset_BUSI'
TARGET_COUNT = 600  # Nombre cible d'images par classe

# Générateur d'augmentation (MÊME seed pour image et masque)
augmentation_params = dict(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Générateur pour les images
image_datagen = ImageDataGenerator(**augmentation_params)

# Générateur pour les masques (sans brightness_range)
mask_params = augmentation_params.copy()
mask_params.pop('brightness_range')  # Pas de changement de luminosité pour les masques
mask_datagen = ImageDataGenerator(**mask_params)


# -----------------------------
# Fonction: Trouver le masque associé
# -----------------------------
def find_mask_file(image_file, class_path):
    """
    Trouve le fichier masque correspondant à une image.
    
    Args:
        image_file (str): Nom du fichier image (ex: 'normal (1).png')
        class_path (str): Chemin du dossier de classe
    
    Returns:
        str or None: Nom du fichier masque (ex: 'normal (1)_mask.png')
    """
    # Retirer l'extension
    base_name = os.path.splitext(image_file)[0]
    
    # Chercher le masque avec le pattern _mask
    mask_patterns = [
        f"{base_name}_mask.png",
        f"{base_name}_mask.jpg",
        f"{base_name}_mask.jpeg"
    ]
    
    for mask_name in mask_patterns:
        if os.path.exists(os.path.join(class_path, mask_name)):
            return mask_name
    
    return None


# -----------------------------
# Fonction: Séparer images et masques
# -----------------------------
def separate_images_and_masks(class_path):
    """
    Sépare les images originales des masques.
    
    Returns:
        tuple: (liste_images, dict_masques)
    """
    all_files = [f for f in os.listdir(class_path) 
                 if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    images = []
    masks_dict = {}
    
    for file in all_files:
        if '_mask' in file:
            continue  # Skip les masques pour l'instant
        
        # C'est une image originale
        mask_file = find_mask_file(file, class_path)
        if mask_file:
            images.append(file)
            masks_dict[file] = mask_file
    
    return images, masks_dict


# -----------------------------
# Fonction: Augmenter image + masque
# -----------------------------
def augment_image_and_mask(image_path, mask_path, class_path, aug_index):
    """
    Génère une image augmentée ET son masque (avec les MÊMES transformations).
    
    Args:
        image_path (str): Chemin de l'image source
        mask_path (str): Chemin du masque source
        class_path (str): Dossier de destination
        aug_index (int): Index pour nommer le fichier
    
    Returns:
        tuple: (nom_image_générée, nom_masque_généré)
    """
    # Charger l'image
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Charger le masque
    mask = load_img(mask_path, target_size=IMAGE_SIZE)
    mask_array = img_to_array(mask)
    mask_array = np.expand_dims(mask_array, axis=0)
    
    # IMPORTANT: Utiliser le même seed pour les mêmes transformations
    seed = np.random.randint(10000)
    
    # Générer l'image augmentée
    augmented_img = image_datagen.flow(img_array, batch_size=1, seed=seed)
    augmented_image = next(augmented_img)[0].astype('uint8')
    
    # Générer le masque augmenté (avec le même seed)
    augmented_msk = mask_datagen.flow(mask_array, batch_size=1, seed=seed)
    augmented_mask = next(augmented_msk)[0].astype('uint8')
    
    # Noms des fichiers générés
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    new_image_name = f"{base_name}_aug_{aug_index}.png"
    new_mask_name = f"{base_name}_aug_{aug_index}_mask.png"
    
    # Sauvegarder
    array_to_img(augmented_image).save(os.path.join(class_path, new_image_name))
    array_to_img(augmented_mask).save(os.path.join(class_path, new_mask_name))
    
    return new_image_name, new_mask_name


# -----------------------------
# Fonction: Augmenter une classe
# -----------------------------
def augment_class_to_target(class_path, target_count):
    """
    Augmente une classe jusqu'à atteindre target_count paires (image+masque).
    
    Args:
        class_path (str): Chemin vers le dossier de la classe
        target_count (int): Nombre cible de paires image/masque
    """
    class_name = os.path.basename(class_path)
    
    # Séparer images et masques
    images, masks_dict = separate_images_and_masks(class_path)
    current_count = len(images)
    
    print(f"\n📁 Classe: {class_name}")
    print(f"   Images actuelles: {current_count}")
    print(f"   Objectif: {target_count}")
    
    if current_count >= target_count:
        print(f"   ✓ Aucune augmentation nécessaire")
        return
    
    images_to_generate = target_count - current_count
    print(f"   → Génération de {images_to_generate} paires image/masque...")
    
    generated = 0
    idx = 0
    
    while generated < images_to_generate:
        # Sélectionner une image source (cyclique)
        source_image = images[idx % len(images)]
        source_mask = masks_dict[source_image]
        
        image_path = os.path.join(class_path, source_image)
        mask_path = os.path.join(class_path, source_mask)
        
        # Générer la paire augmentée
        augment_image_and_mask(image_path, mask_path, class_path, generated)
        
        generated += 1
        idx += 1
        
        # Afficher la progression
        if generated % 50 == 0 or generated == images_to_generate:
            print(f"   Progression: {generated}/{images_to_generate}")
    
    print(f"   ✓ Terminé! Total: {target_count} paires")


# -----------------------------
# Fonction: Augmenter tout le dataset
# -----------------------------
def augment_dataset(split='train', target_count=TARGET_COUNT):
    """
    Augmente toutes les classes d'un split jusqu'à target_count.
    
    Args:
        split (str): 'train', 'validation' ou 'test'
        target_count (int): Nombre cible d'images par classe
    """
    dataset_path = os.path.join(DATASET_DIR, split)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Erreur: {dataset_path} n'existe pas!")
        return
    
    print(f"\n{'='*60}")
    print(f"🔬 AUGMENTATION DU DATASET: {split.upper()}")
    print(f"{'='*60}")
    print(f"🎯 Objectif: {target_count} images par classe")
    
    # Traiter chaque classe
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            augment_class_to_target(class_path, target_count)
    
    print(f"\n{'='*60}")
    print(f"✅ AUGMENTATION TERMINÉE!")
    print(f"{'='*60}\n")


# -----------------------------
# Fonction: Créer backup
# -----------------------------
def create_backup(split='train'):
    """
    Crée une sauvegarde avant l'augmentation.
    """
    source = os.path.join(DATASET_DIR, split)
    backup = os.path.join(DATASET_DIR, f'{split}_backup')
    
    if os.path.exists(backup):
        print(f"⚠️  Backup existe déjà: {backup}")
        response = input("Voulez-vous le remplacer? (o/n): ")
        if response.lower() != 'o':
            return
        shutil.rmtree(backup)
    
    print(f"💾 Création du backup: {backup}")
    shutil.copytree(source, backup)
    print("✓ Backup créé avec succès!")


# -----------------------------
# Fonction principale
# -----------------------------
def main():
    """
    Lance l'augmentation sur le dataset d'entraînement.
    """
    print("\n" + "="*60)
    print("🔬 AUGMENTATION DE DONNÉES - DATASET BUSI (Images + Masques)")
    print("="*60)
    
    # Créer un backup de sécurité
    print("\n[ÉTAPE 1] Backup de sécurité")
    create_backup('train')
    
    # Augmenter le dataset d'entraînement
    print("\n[ÉTAPE 2] Augmentation du dataset")
    augment_dataset('train', target_count=TARGET_COUNT)
    
    print("\n💡 Conseils:")
    print("  - Un backup a été créé dans 'Dataset_BUSI/train_backup'")
    print("  - Chaque image a son masque augmenté de la même manière")
    print("  - Les transformations sont synchronisées (même seed)")
    print()


# -----------------------------
# Exécution
# -----------------------------
if __name__ == "__main__":
    main()