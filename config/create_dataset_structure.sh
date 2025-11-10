#!/bin/bash
# Script pour créer la structure de dataset pour l'entraînement YOLO

DATASET_DIR="$HOME/Mark3_ws/dataset"

echo "Création de la structure de dataset pour YOLO..."
echo ""

mkdir -p "$DATASET_DIR/train/images"
mkdir -p "$DATASET_DIR/train/labels"
mkdir -p "$DATASET_DIR/val/images"
mkdir -p "$DATASET_DIR/val/labels"

echo "✅ Structure créée:"
echo "$DATASET_DIR/"
echo "├── train/"
echo "│   ├── images/  (placez vos images d'entraînement ici)"
echo "│   └── labels/  (les annotations .txt seront ici)"
echo "├── val/"
echo "│   ├── images/  (placez vos images de validation ici)"
echo "│   └── labels/  (les annotations .txt seront ici)"
echo "└── data.yaml"
echo ""
echo "📝 Créez maintenant le fichier data.yaml avec vos classes"
echo ""
echo "Exemple de data.yaml:"
cat << 'EOF'
path: /home/axel/Mark3_ws/dataset
train: train/images
val: val/images

nc: 3
names:
  0: screwdriver
  1: wrench
  2: pliers
EOF

