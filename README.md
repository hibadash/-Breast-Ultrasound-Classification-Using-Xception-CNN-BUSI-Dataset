# Classification d'Images de Cancer du Sein - Xception
Ce projet propose une approche d'apprentissage profond pour la classification histopathologique automatisée des images de tissus mammaires. En exploitant l'architecture Xception basée sur les convolutions séparables en profondeur, le système réalise une classification ternaire distinguant les tissus **bénins**, **malins** et **normaux**. L'approche intègre un pipeline de prétraitement rigoureux, des stratégies d'augmentation de données adaptées au domaine médical, ainsi qu'une évaluation quantitative exhaustive des performances du modèle.



## Installation

Clonez le repository et installez les dépendances :

```bash
git clone https://github.com/hibadash -Breast-Ultrasound-Classification-Using-Xception-CNN-BUSI-Dataset.git
pip install -r requirements.txt
```



# Classifier une image
![Apercu des images à classifier](overview/Overview.png)



## Caractéristiques

- **Classification multi-classe (3 classes dans notre cas d'étude)** : Distinction entre tissus bénins, malins et normaux
- **Architecture Xception** : Réseau de neurones convolutifs pré-entraîné sur ImageNet
- **Prétraitement automatique** : Redimensionnement et normalisation des images
- **Augmentation de données** : Rotation, zoom, flip pour améliorer la généralisation
- **Métriques détaillées pour l'évaluation** : Accuracy , matrice de confusion, courbes ROC et AUC

## Ensemble de données

Le modèle est entraîné sur des images de tissus mammaires :

| Classe | Description |
|--------|-------------|
| **Normal** | Tissu mammaire sain |
| **Bénin** | Tumeurs non cancéreuses (fibroadénomes, kystes) |
| **Malin** | Tumeurs cancéreuses (carcinomes) |

Répartition : 70% entraînement, 15% validation, 15% test



## Contributeurs

- [DADDA Hiba](https://github.com/hibadash)
- [LAMSSANE Fatima](https://github.com/zohrae)
- [BERROUCH Kawtar](https://github.com/kawtar-Berr)

Université Cadi Ayyad - Faculté des Sciences Semlalia Marrakech, Maroc



