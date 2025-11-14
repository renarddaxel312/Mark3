# Résolution des problèmes X11/Wayland avec l'Interface

## Problème

L'erreur `BadWindow (invalid Window parameter)` avec `X_ConfigureWindow` et les erreurs `QThreadStorage` peuvent se produire lors de la fermeture de l'interface, particulièrement dans les environnements Wayland avec compatibilité X11 (XWayland).

## Causes possibles

1. **Environnement Wayland/X11 hybride** : Le système utilise Wayland mais avec compatibilité X11, ce qui peut causer des problèmes de timing lors de la fermeture des fenêtres.

2. **Race conditions** : Les threads Qt et VTK peuvent essayer d'accéder à des ressources X11 après leur destruction.

3. **Nettoyage incomplet** : Les ressources VTK/X11 ne sont pas libérées dans le bon ordre.

## Solutions implémentées

### 1. Forcer Qt à utiliser X11

Le code force maintenant Qt à utiliser le backend X11 (`xcb`) au lieu de Wayland :

```python
if 'QT_QPA_PLATFORM' not in os.environ:
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
```

### 2. Traitement des événements

Des appels à `app.processEvents()` ont été ajoutés à plusieurs endroits dans `closeEvent()` pour traiter les événements en attente et éviter les race conditions.

### 3. Délais pour VTK

Des délais synchrones ont été ajoutés pour permettre à VTK de finaliser proprement ses ressources avant la destruction de la fenêtre.

### 4. Nettoyage amélioré

- Le viewer VTK est nettoyé avant l'arrêt des threads
- Les signaux Qt sont déconnectés avant la fermeture
- Un flag `_closing` empêche les mises à jour pendant la fermeture

## Solutions alternatives

Si le problème persiste, vous pouvez essayer :

### Option 1 : Forcer X11 au niveau système

Avant de lancer l'application :

```bash
export QT_QPA_PLATFORM=xcb
ros2 run Interface Interface
```

### Option 2 : Utiliser un serveur X virtuel (pour tests)

Si vous testez sans interface graphique :

```bash
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
ros2 run Interface Interface
```

### Option 3 : Vérifier les permissions X11

Assurez-vous que vous avez les permissions nécessaires :

```bash
xhost +local:  # Attention : réduit la sécurité
```

### Option 4 : Désactiver Wayland (si possible)

Si vous utilisez GNOME, vous pouvez forcer X11 au démarrage de la session en modifiant `/etc/gdm3/custom.conf` ou en utilisant l'option de démarrage.

## Vérification

Pour vérifier votre environnement :

```bash
echo $XDG_SESSION_TYPE  # Affiche "wayland" ou "x11"
echo $DISPLAY           # Affiche ":0" ou similaire
xdpyinfo | head -5      # Informations sur le serveur X
```

## Notes

- Ces problèmes sont souvent spécifiques à l'environnement système
- Si le code fonctionne sur un autre PC avec les mêmes versions Python, c'est probablement un problème d'environnement système (Wayland/X11, drivers graphiques, etc.)
- Les délais ajoutés peuvent ralentir légèrement la fermeture, mais garantissent un nettoyage propre

