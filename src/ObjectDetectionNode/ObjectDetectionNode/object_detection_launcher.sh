#!/bin/bash
# Launcher pour object_detection_node qui utilise le venv Python
VENV_PYTHON="$HOME/venv/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$VENV_PYTHON" ]; then
    exec "$VENV_PYTHON" "$SCRIPT_DIR/ObjectDetectionNode.py" "$@"
else
    echo "Erreur: venv Python non trouve a $VENV_PYTHON"
    exit 1
fi

