#!/bin/bash
# Script d'installation de PyAudio pour Python 3.12
# Ce script résout le problème de compatibilité avec Python 3.12

set -e  # Arrêter en cas d'erreur

echo "=== Installation de PyAudio ==="
echo ""

# Vérifier si on est dans un venv
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Avertissement: Vous n'êtes pas dans un environnement virtuel"
    echo "   Activez votre venv avec: source ~/venv/bin/activate"
    read -p "Continuer quand même? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Étape 1: Installer les dépendances système
echo "📦 Étape 1: Installation des dépendances système (portaudio19-dev)..."
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio || {
    echo "❌ Échec de l'installation des dépendances système"
    exit 1
}

# Étape 2: Mettre à jour pip et setuptools
echo ""
echo "🔄 Étape 2: Mise à jour de pip, setuptools et wheel..."
pip install --upgrade pip setuptools wheel

# Étape 3: Installer PyAudio
echo ""
echo "📥 Étape 3: Installation de PyAudio..."
pip install pyaudio || {
    echo ""
    echo "❌ L'installation de PyAudio a échoué"
    echo ""
    echo "Solutions alternatives:"
    echo "1. Utiliser Python 3.11 au lieu de 3.12"
    echo "2. Voir INSTALL_PYAUDIO.md pour plus d'options"
    exit 1
}

# Vérification
echo ""
echo "✅ Vérification de l'installation..."
python3 -c "import pyaudio; print('✓ PyAudio installé avec succès!')" || {
    echo "❌ PyAudio n'a pas pu être importé"
    exit 1
}

echo ""
echo "🎉 PyAudio a été installé avec succès!"

