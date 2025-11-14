# Session de débogage - Mark3

Ce document récapitule tous les problèmes rencontrés et les solutions testées lors de la configuration et du débogage du projet Mark3.

## 1. Problème d'installation de PyAudio

### Erreur initiale
```
AttributeError: module 'pkgutil' has no attribute 'ImpImporter'
```

### Cause
- PyAudio 0.2.13 est une version très ancienne (2017)
- Python 3.12 a supprimé `pkgutil.ImpImporter` (déprécié depuis Python 3.4)
- Les anciennes versions de `setuptools/pkg_resources` ne sont pas compatibles avec Python 3.12

### Solutions testées

#### Solution 1 : Installation des dépendances système (RECOMMANDÉ)
```bash
sudo apt-get install -y portaudio19-dev python3-pyaudio
pip install --upgrade pip setuptools wheel
pip install pyaudio
```

#### Solution 2 : Utiliser Python 3.11
```bash
python3.11 -m venv ~/venv311
source ~/venv311/bin/activate
pip install -r requirements.txt
```

#### Solution 3 : Alternative avec sounddevice
```bash
pip install sounddevice
# Nécessite de modifier le code pour utiliser sounddevice au lieu de pyaudio
```

### Fichiers modifiés
- `requirements.txt` : Version de PyAudio rendue flexible
- `INSTALL_PYAUDIO.md` : Guide d'installation créé
- `install_pyaudio.sh` : Script d'installation automatique créé

---

## 2. Problème avec le module `em` vs `empy`

### Erreur initiale
```
AttributeError: module 'em' has no attribute 'Interpreter'
```

### Cause
- Le package `em` (version 0.4.0) était installé au lieu de `empy`
- `em` est un outil de coloration de terminal, pas le moteur de templates ROS2
- Les deux packages créent un module Python nommé `em`, mais seul `empy` contient `Interpreter`

### Solution
```bash
pip uninstall -y em
pip install empy
```

### Fichiers modifiés
- `requirements.txt` : Ajout de `empy` avec avertissement sur le conflit

---

## 3. Problème avec `catkin_pkg`

### Erreur initiale
```
ModuleNotFoundError: No module named 'catkin_pkg'
```

### Cause
- `catkin_pkg` est requis par ROS2 pour parser les fichiers `package.xml`
- Non installé dans l'environnement virtuel

### Solution
```bash
pip install catkin_pkg
```

### Fichiers modifiés
- `requirements.txt` : Ajout de `catkin_pkg`

---

## 4. Problème avec `lark`

### Erreur initiale
```
ModuleNotFoundError: No module named 'lark'
```

### Cause
- `lark` est un parser Python requis par `rosidl_parser` pour parser les fichiers IDL ROS2
- Non installé dans l'environnement virtuel

### Solution
```bash
pip install lark
```

### Fichiers modifiés
- `requirements.txt` : Ajout de `lark`

---

## 5. Problème X11/Wayland lors de la fermeture de l'Interface

### Erreur initiale
```
X Error of failed request: BadWindow (invalid Window parameter)
Major opcode of failed request: 12 (X_ConfigureWindow)
QThreadStorage: entry 1 destroyed before end of thread
QThreadStorage: entry 0 destroyed before end of thread
```

### Cause
- Environnement Wayland avec compatibilité X11 (XWayland)
- Race conditions lors de la fermeture entre Qt, VTK et les threads ROS2
- Le widget VTK n'était pas nettoyé avant la destruction de la fenêtre
- Les threads Qt/VTK accédaient à des ressources X11 après leur destruction

### Solutions testées

#### Solution 1 : Amélioration du nettoyage des threads
**Fichier** : `src/Interface/Interface/Interface.py`

- Ajout d'un bloc `try/except/finally` dans `RosThread.run()`
- Nettoyage ROS2 (`destroy_node()` et `rclpy.shutdown()`) dans le `finally`
- Amélioration de `RosThread.stop()` avec timeout et attente

#### Solution 2 : Amélioration de `closeEvent()`
**Fichier** : `src/Interface/Interface/Interface.py`

- Ajout d'un flag `_closing` pour empêcher les mises à jour pendant la fermeture
- Ordre de nettoyage : timer → VTK → signaux Qt → thread ROS2 → stdout/stderr
- Ajout de `app.processEvents()` à plusieurs endroits pour traiter les événements en attente
- Délai synchrone de 50ms après le nettoyage VTK

#### Solution 3 : Nettoyage VTK amélioré
**Fichier** : `src/Interface/Interface/robot_3d_viewer.py`

- Création de la méthode `cleanup()` dans `Robot3DViewer`
- Désactivation de l'interactor VTK avant nettoyage
- Délai de 10ms pour laisser les événements se terminer
- Nettoyage des actors, renderer et fenêtre de rendu
- Retrait du widget du layout

#### Solution 4 : Forcer Qt à utiliser X11
**Fichier** : `src/Interface/Interface/Interface.py`

```python
if 'QT_QPA_PLATFORM' not in os.environ:
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
```

#### Solution 5 : Protections dans les callbacks
**Fichier** : `src/Interface/Interface/Interface.py`

- Vérification du flag `_closing` dans `update_image()` et `update_3d_view()`
- Vérification de l'existence des widgets avant mise à jour
- Gestion d'exceptions pour éviter les erreurs lors de la fermeture

### Solutions alternatives testées

#### Option A : Forcer X11 manuellement
```bash
export QT_QPA_PLATFORM=xcb
ros2 run Interface Interface
```

#### Option B : Vérifier l'environnement
```bash
echo $XDG_SESSION_TYPE  # Affiche "wayland" ou "x11"
echo $DISPLAY           # Affiche ":0" ou similaire
xdpyinfo | head -5      # Informations sur le serveur X
```

### Fichiers modifiés
- `src/Interface/Interface/Interface.py` : Améliorations du nettoyage
- `src/Interface/Interface/robot_3d_viewer.py` : Méthode `cleanup()` ajoutée
- `TROUBLESHOOTING_X11.md` : Guide de dépannage créé

### Notes importantes
- Le problème est spécifique à l'environnement Wayland/X11
- Le code fonctionne sur un autre PC avec les mêmes versions Python, confirmant que c'est un problème d'environnement système
- Les délais ajoutés peuvent ralentir légèrement la fermeture, mais garantissent un nettoyage propre

---

## Résumé des dépendances ROS2 ajoutées

Toutes ces dépendances ont été ajoutées à `requirements.txt` :

```python
# ROS2 build tools
empy  # Template engine required by rosidl_adapter (DO NOT install 'em' package - it conflicts!)
catkin_pkg  # Required for parsing package.xml files during ROS2 build
lark  # Parser library required by rosidl_parser for parsing IDL files
```

---

## Commandes utiles pour le débogage

### Vérifier les versions Python
```bash
python3 --version
```

### Vérifier les modules installés
```bash
pip list | grep -E "empy|catkin|lark|pyaudio"
```

### Vérifier l'environnement X11/Wayland
```bash
echo $XDG_SESSION_TYPE
echo $DISPLAY
echo $QT_QPA_PLATFORM
```

### Recompiler après modifications
```bash
colcon build --symlink-install
source install/setup.bash
```

### Tester l'interface
```bash
ros2 run Interface Interface
```

---

## Fichiers de documentation créés

1. **INSTALL_PYAUDIO.md** : Guide d'installation de PyAudio
2. **TROUBLESHOOTING_X11.md** : Guide de dépannage X11/Wayland
3. **DEBUG_SESSION.md** : Ce document (récapitulatif complet)

---

## État final

### Dépendances Python installées
- ✅ `empy` (moteur de templates ROS2)
- ✅ `catkin_pkg` (parsing package.xml)
- ✅ `lark` (parser IDL)
- ✅ `pyaudio` (après installation des dépendances système)

### Corrections appliquées
- ✅ Nettoyage propre des threads ROS2
- ✅ Nettoyage propre du viewer VTK
- ✅ Gestion des événements Qt améliorée
- ✅ Protection contre les race conditions
- ✅ Support Wayland/X11 amélioré

### Problèmes résolus
- ✅ Installation de PyAudio avec Python 3.12
- ✅ Conflit entre `em` et `empy`
- ✅ Modules ROS2 manquants (`catkin_pkg`, `lark`)
- ⚠️ Problème X11/Wayland (améliorations apportées, peut nécessiter ajustements selon l'environnement)

---

## Notes finales

- La plupart des problèmes étaient liés à des dépendances manquantes ou des conflits de noms de modules
- Le problème X11/Wayland est spécifique à l'environnement système et peut varier selon la configuration
- Toutes les solutions ont été documentées pour faciliter le débogage futur
- Les scripts et guides créés permettent une installation et un dépannage plus faciles

