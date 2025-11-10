# Guide de Calibration de la Caméra Nuroum 1080p

## Spécifications de la caméra
- **Marque**: Nuroum
- **Résolution**: 1080p (1920x1080)
- **Focale**: 4.4 mm
- **Ouverture**: f/2.4
- **Connectivité**: USB

## Prérequis

1. **Damier (Chessboard) à imprimer**
   - Téléchargez un damier depuis: https://docs.opencv.org/4.x/pattern.png
   - Ou créez-en un avec 9x6 coins internes (10x7 carrés)
   - Taille recommandée: A4 (21x29.7 cm)
   - Mesurez la taille réelle d'un carré en mètres (important!)

2. **Dépendances Python**
   ```bash
   pip3 install opencv-python numpy pyyaml
   ```

## Étapes de calibration

### 1. Préparer le damier
- Imprimez le damier sur une feuille rigide (évitez le papier souple)
- Mesurez la taille réelle d'un carré (en mètres)
- Modifiez `SQUARE_SIZE` dans `calibrate_camera.py` si nécessaire

### 2. Lancer le script de calibration
```bash
cd /home/axel/Mark3_ws/config
python3 calibrate_camera.py
```

### 3. Capturer les images
- **Prenez 15-20 photos** du damier sous différents angles:
  - Vue de face
  - Vue de côté
  - Vue en diagonale
  - Vue de dessus
  - Vue de dessous
  - Inclinez le damier à différents angles
- **Appuyez sur ESPACE** pour capturer une image valide
- **Appuyez sur 'q'** pour terminer et sauvegarder

### 4. Vérifier les résultats
Le script affichera:
- L'erreur de reprojection (doit être < 0.5 pixels idéalement)
- La matrice de caméra
- Les coefficients de distortion

Le fichier `camera_calibration.yaml` sera créé dans le dossier `config/`.

## Utilisation de la calibration

### Option 1: Via paramètre ROS2
```bash
ros2 run ObjectDetectionNode object_detection_node \
  --ros-args \
  -p calibration_file:=/home/axel/Mark3_ws/config/camera_calibration.yaml
```

### Option 2: Via launch file
Ajoutez dans votre launch file:
```python
Node(
    package='ObjectDetectionNode',
    executable='object_detection_node',
    parameters=[{
        'calibration_file': '/home/axel/Mark3_ws/config/camera_calibration.yaml',
        'model_path': 'yolov8n.pt',
        'camera_frame': 'camera_frame',
        'base_frame': 'base_link'
    }]
)
```

## Format du fichier de calibration

Le fichier YAML généré contient:
```yaml
image_width: 1920
image_height: 1080
camera_name: "nuroum_1080p"
camera_matrix:
  rows: 3
  cols: 3
  data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
distortion_coefficients:
  rows: 1
  cols: 5
  data: [k1, k2, p1, p2, k3]
reprojection_error: 0.xxx
num_images: XX
```

## Conseils pour une bonne calibration

1. **Éclairage uniforme**: Évitez les reflets et ombres fortes
2. **Damier rigide**: Utilisez un support rigide (carton épais, planche)
3. **Variété d'angles**: Plus vous variez les angles, meilleure sera la calibration
4. **Distance**: Variez la distance entre la caméra et le damier
5. **Damier bien visible**: Le damier doit remplir au moins 1/3 de l'image
6. **Minimum 10 images**: Mais 15-20 images donnent de meilleurs résultats

## Dépannage

### "Damier non détecté"
- Vérifiez que le damier est bien visible
- Assurez-vous que le damier n'est pas trop petit dans l'image
- Vérifiez l'éclairage (pas trop sombre, pas de reflets)

### Erreur de reprojection élevée (> 1.0 pixel)
- Capturez plus d'images (20-30)
- Variez davantage les angles
- Vérifiez que le damier est bien imprimé (lignes droites)

### Caméra non détectée
- Vérifiez que la caméra est branchée en USB
- Testez avec: `v4l2-ctl --list-devices`
- Changez l'index de la caméra dans le script (0, 1, 2...)

## Notes importantes

⚠️ **Les valeurs par défaut ne sont pas précises!**
- Vous DEVEZ calibrer votre caméra pour obtenir des coordonnées 3D précises
- Les valeurs par défaut sont des estimations grossières
- Sans calibration précise, l'estimation 3D sera imprécise

