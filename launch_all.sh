#!/bin/bash
# Script pour lancer tout le système Mark3 avec détection d'objets

cd "$(dirname "$0")"

# Sourcer l'environnement ROS2
source /opt/ros/jazzy/setup.bash
source install/setup.bash

echo "=========================================="
echo "Lancement du système Mark3"
echo "=========================================="
echo ""

# Vérifier que les packages sont buildés
if [ ! -d "install" ]; then
    echo "Erreur: Le dossier install/ n'existe pas."
    echo "Veuillez d'abord builder les packages:"
    echo "  colcon build --symlink-install"
    exit 1
fi

# Lancer tous les nodes
echo "Lancement des nodes..."
echo ""

# Node 1: Caméra
echo "[1/5] Lancement de CameraNode..."
ros2 run CameraNode CameraNode &
CAMERA_PID=$!

sleep 1

# Node 2: Détection d'objets
echo "[2/5] Lancement de ObjectDetectionNode..."
ros2 run ObjectDetectionNode object_detection_node \
  --ros-args \
  -p calibration_file:=$(pwd)/config/camera_calibration.yaml \
  -p fixed_depth:=0.0 &
DETECTION_PID=$!

sleep 1

# Node 3: Serveur IK
echo "[3/5] Lancement de IKsolverNode..."
ros2 run IKsolverNode IKsolverNode &
IK_PID=$!

sleep 1

# Node 4: State Publisher
echo "[4/5] Lancement de state_publisher..."
ros2 run state_publisher state_publisher &
STATE_PID=$!

sleep 1

# Node 5: URDF Reloader
echo "[5/5] Lancement de urdf_reloader..."
ros2 run state_publisher urdf_reloader &
URDF_PID=$!

sleep 2

# Interface (bloquant)
echo ""
echo "=========================================="
echo "Lancement de l'interface graphique..."
echo "=========================================="
echo ""
echo "Appuyez sur Ctrl+C pour arrêter tous les nodes"
echo ""

# Fonction de nettoyage
cleanup() {
    echo ""
    echo "Arrêt des nodes..."
    kill $CAMERA_PID $DETECTION_PID $IK_PID $STATE_PID $URDF_PID 2>/dev/null
    echo "Nodes arrêtés."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Lancer l'interface (bloquant)
ros2 run Interface Interface

# Nettoyage à la fin
cleanup

