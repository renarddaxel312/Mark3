# Guide Rapide - Entraînement YOLO

## Vérification du Dataset

Votre dataset est prêt :
- **Images d'entraînement**: 10,998
- **Images de validation**: 1,382
- **Classe**: screwdriver (tournevis)

Pour vérifier le dataset :
```bash
cd ~/Mark3_ws
python3 config/verify_dataset.py
```

## Lancer l'Entraînement

### Option 1 : Script interactif (recommandé)

```bash
cd ~/Mark3_ws
python3 config/train_tools_detector.py
```

Le script vous demandera :
1. Le modèle à utiliser (nano, small, medium, large)
2. Le nombre d'epochs (défaut: 100)
3. La taille du batch (défaut: 16)
4. La taille des images (défaut: 640)

### Option 2 : Entraînement direct avec Ultralytics

```bash
cd ~/Mark3_ws
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(
    data='Screwdriver.v17i.yolov8/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='screwdriver_detector',
    project='yolo_training'
)
"
```

## Modèles Disponibles

- **yolov8n.pt** : Nano - Rapide, moins précis (recommandé pour commencer)
- **yolov8s.pt** : Small - Équilibré
- **yolov8m.pt** : Medium - Plus précis, plus lent
- **yolov8l.pt** : Large - Très précis, lent

## Résultats de l'Entraînement

Après l'entraînement, les modèles seront dans :
- `yolo_training/screwdriver_detector/weights/best.pt` - Meilleur modèle
- `yolo_training/screwdriver_detector/weights/last.pt` - Dernier modèle

## Utiliser le Modèle Entraîné

### Option 1 : Modifier le launch file

Dans `src/system/launch/robot_system.launch.py`, modifiez :
```python
'model_path': '/home/axel/Mark3_ws/yolo_training/screwdriver_detector/weights/best.pt'
```

### Option 2 : Lancer avec paramètre ROS2

```bash
ros2 run ObjectDetectionNode object_detection_node \
  --ros-args \
  -p model_path:=/home/axel/Mark3_ws/yolo_training/screwdriver_detector/weights/best.pt \
  -p fixed_depth:=0.0
```

## Conseils

- **Premier entraînement** : Utilisez `yolov8n.pt` avec 50 epochs pour tester rapidement
- **Entraînement complet** : Utilisez `yolov8s.pt` ou `yolov8m.pt` avec 100-200 epochs
- **Surveillance** : Les graphiques de performance sont dans `yolo_training/screwdriver_detector/`
- **GPU** : Si vous avez une GPU NVIDIA, l'entraînement sera beaucoup plus rapide

## Problèmes Courants

### "CUDA out of memory"
- Réduisez la taille du batch (ex: batch=8 ou batch=4)
- Utilisez un modèle plus petit (yolov8n.pt)

### "ModuleNotFoundError: No module named 'ultralytics'"
- Installez dans votre venv : `pip install ultralytics`
- Ou utilisez le venv : `source ~/venv/bin/activate`

### Dataset non trouvé
- Vérifiez que le dataset est dans `~/Mark3_ws/Screwdriver.v17i.yolov8/`
- Lancez `python3 config/verify_dataset.py` pour diagnostiquer

