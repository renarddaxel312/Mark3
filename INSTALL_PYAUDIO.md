# Résolution du problème d'installation de PyAudio

## Explication de l'erreur

L'erreur `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'` se produit car :

1. **PyAudio 0.2.13** est une version très ancienne (dernière mise à jour en 2017)
2. Cette version utilise des outils de build obsolètes qui dépendent de `pkgutil.ImpImporter`
3. **Python 3.12** a supprimé `ImpImporter` du module `pkgutil` (déprécié depuis Python 3.4)
4. Les anciennes versions de `setuptools/pkg_resources` utilisées lors du build ne sont pas compatibles avec Python 3.12

## Solutions

### Solution 1 : Installer les dépendances système + mettre à jour les outils (RECOMMANDÉ)

```bash
# 1. Installer les dépendances système pour PyAudio
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio

# 2. Mettre à jour pip et setuptools dans votre venv
source ~/venv/bin/activate  # ou votre chemin de venv
pip install --upgrade pip setuptools wheel

# 3. Essayer d'installer PyAudio (sans version spécifique pour obtenir la dernière compatible)
pip install pyaudio
```

### Solution 2 : Utiliser Python 3.11

Si la solution 1 ne fonctionne pas, créez un nouvel environnement avec Python 3.11 :

```bash
# Installer Python 3.11 si nécessaire
sudo apt-get install python3.11 python3.11-venv

# Créer un nouvel environnement
python3.11 -m venv ~/venv311
source ~/venv311/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Solution 3 : Utiliser une alternative (sounddevice)

Si PyAudio continue de poser problème, vous pouvez utiliser `sounddevice` qui est plus moderne et mieux maintenu :

```bash
pip install sounddevice
```

Puis modifier le code pour utiliser `sounddevice` au lieu de `pyaudio`. Cette bibliothèque a une API similaire.

### Solution 4 : Utiliser un wheel précompilé

Si disponible pour votre architecture, vous pouvez installer un wheel précompilé :

```bash
pip install --only-binary :all: pyaudio
```

## Vérification

Après installation, testez PyAudio :

```python
python3 -c "import pyaudio; print('PyAudio installé avec succès')"
```

