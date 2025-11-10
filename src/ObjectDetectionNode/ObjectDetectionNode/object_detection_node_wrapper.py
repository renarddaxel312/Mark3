#!/usr/bin/env python3
"""
Wrapper pour object_detection_node qui s'assure d'utiliser le venv Python
"""
import sys
import os

# Ajouter le venv au path si disponible
venv_python = os.path.expanduser('~/venv/bin/python3')
if os.path.exists(venv_python):
    # Utiliser le venv Python pour exécuter le script
    import subprocess
    script_path = os.path.join(os.path.dirname(__file__), 'ObjectDetectionNode.py')
    sys.exit(subprocess.call([venv_python, script_path] + sys.argv[1:]))
else:
    # Fallback: importer normalement
    from ObjectDetectionNode.ObjectDetectionNode import main
    if __name__ == '__main__':
        main()

