# Guide de Fine-Tuning YOLO pour Détecter des Tournevis et Outils

Ce guide explique comment entraîner un modèle YOLO personnalisé pour détecter des tournevis et autres outils.

## Prérequis

```bash
pip install ultralytics roboflow
```

## Méthode 1 : Fine-tuning avec Ultralytics (Recommandé)

### Étape 1 : Préparer vos données

1. **Collecter des images** :
   - Prenez 100-200 photos de tournevis/outils sous différents angles
   - Varier l'éclairage, les arrière-plans, les positions
   - Format : JPG ou PNG

2. **Annoter les images** :
   - Utilisez [LabelImg](https://github.com/tzutalin/labelImg) ou [Roboflow](https://roboflow.com)
   - Créez des bounding boxes autour de chaque outil
   - Exportez au format YOLO (fichiers .txt avec coordonnées normalisées)

Structure des dossiers :
```
dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

### Étape 2 : Créer le fichier data.yaml

Créez `dataset/data.yaml` :
```yaml
path: /home/axel/Mark3_ws/dataset
train: train/images
val: val/images

nc: 3  # Nombre de classes
names:
  0: screwdriver  # Tournevis
  1: wrench       # Clé
  2: pliers       # Pince
```

### Étape 3 : Entraîner le modèle

Créez un script `train_yolo_tools.py` :

```python
from ultralytics import YOLO

# Charger un modèle pré-entraîné
model = YOLO('yolov8n.pt')  # ou yolov8s.pt, yolov8m.pt pour plus de précision

# Fine-tuning
results = model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='tools_detector',
    project='yolo_training'
)

# Le modèle entraîné sera sauvegardé dans:
# yolo_training/tools_detector/weights/best.pt
```

Lancez l'entraînement :
```bash
python3 train_yolo_tools.py
```

### Étape 4 : Utiliser le modèle entraîné

Modifiez le paramètre `model_path` dans votre launch file :
```python
'model_path': '/home/axel/Mark3_ws/yolo_training/tools_detector/weights/best.pt'
```

## Méthode 2 : Utiliser Roboflow (Plus simple, nécessite compte)

1. **Créer un compte** sur [roboflow.com](https://roboflow.com)
2. **Uploader vos images** et annoter directement sur la plateforme
3. **Exporter** au format YOLO v8
4. **Télécharger** le dataset et entraîner avec Ultralytics

## Méthode 3 : Augmentation de données

Si vous avez peu d'images, utilisez l'augmentation de données :

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='tools_detector',
    project='yolo_training',
    # Augmentation de données
    hsv_h=0.015,      # Variation teinte
    hsv_s=0.7,        # Variation saturation
    hsv_v=0.4,        # Variation luminosité
    degrees=10,        # Rotation
    translate=0.1,    # Translation
    scale=0.5,        # Zoom
    flipud=0.0,       # Flip vertical
    fliplr=0.5,       # Flip horizontal
    mosaic=1.0,       # Mosaic augmentation
    mixup=0.1         # Mixup augmentation
)
```

## Conseils pour améliorer la détection

1. **Plus d'images = meilleure détection** : Minimum 100 images par classe
2. **Variété** : Différents angles, éclairages, arrière-plans
3. **Qualité des annotations** : Bounding boxes précises
4. **Classes spécifiques** : Séparer "tournevis plat" et "tournevis cruciforme" si nécessaire
5. **Test/Validation** : Gardez 20% des images pour la validation

## Structure recommandée pour les outils

```yaml
nc: 5
names:
  0: screwdriver_flat      # Tournevis plat
  1: screwdriver_phillips   # Tournevis cruciforme
  2: wrench                 # Clé
  3: pliers                 # Pince
  4: hammer                 # Marteau
```

## Script d'entraînement complet

Créez `config/train_tools_detector.py` :

```python
#!/usr/bin/env python3
"""
Script pour entraîner YOLO sur des outils
"""
from ultralytics import YOLO
import os

# Chemin vers le dataset
dataset_path = os.path.expanduser('~/Mark3_ws/dataset')

# Vérifier que le dataset existe
if not os.path.exists(dataset_path):
    print(f"Erreur: Dataset non trouvé à {dataset_path}")
    print("Créez d'abord votre dataset avec la structure train/val")
    exit(1)

# Charger le modèle pré-entraîné
print("Chargement du modèle YOLOv8n...")
model = YOLO('yolov8n.pt')

# Entraîner
print("Démarrage de l'entraînement...")
results = model.train(
    data=os.path.join(dataset_path, 'data.yaml'),
    epochs=100,
    imgsz=640,
    batch=16,
    name='tools_detector',
    project='yolo_training',
    patience=20,  # Arrêt anticipé si pas d'amélioration
    save=True,
    plots=True
)

print(f"\nEntraînement terminé!")
print(f"Meilleur modèle: yolo_training/tools_detector/weights/best.pt")
print(f"Modèle final: yolo_training/tools_detector/weights/last.pt")
```

## Utilisation après entraînement

1. **Tester le modèle** :
```python
from ultralytics import YOLO
model = YOLO('yolo_training/tools_detector/weights/best.pt')
results = model('test_image.jpg')
results[0].show()
```

2. **Intégrer dans ObjectDetectionNode** :
   - Modifiez `model_path` dans le launch file
   - Ou passez-le en paramètre : `-p model_path:=/path/to/best.pt`

## Ressources utiles

- **LabelImg** : https://github.com/tzutalin/labelImg
- **Roboflow** : https://roboflow.com
- **Documentation Ultralytics** : https://docs.ultralytics.com
- **Tutoriel YOLO** : https://docs.ultralytics.com/modes/train/

