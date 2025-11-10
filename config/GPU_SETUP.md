# Configuration GPU pour l'Entraînement YOLO

## Vérification de la GPU

Pour vérifier si vous avez une GPU NVIDIA :
```bash
nvidia-smi
```

Si cette commande fonctionne, vous avez une GPU NVIDIA.

## Installation de PyTorch avec Support CUDA

### Option 1 : Installation via pip (recommandé)

1. **Vérifier la version de CUDA installée** :
```bash
nvcc --version
# ou
nvidia-smi
```

2. **Installer PyTorch avec CUDA** :

Pour CUDA 11.8 :
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Pour CUDA 12.1 :
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Pour la dernière version stable :
```bash
pip install torch torchvision torchaudio
```

3. **Vérifier l'installation** :
```bash
python3 -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### Option 2 : Installation via conda

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Vérification

Après installation, lancez :
```bash
cd ~/Mark3_ws
python3 config/train_tools_detector.py
```

Le script affichera maintenant si le GPU est détecté et utilisé.

## Performance

- **Avec GPU** : L'entraînement peut être 10-50x plus rapide
- **Sans GPU** : L'entraînement sur CPU est très lent (peut prendre des heures/jours)

## Problèmes Courants

### "CUDA out of memory"
- Réduisez la taille du batch (batch=8 ou batch=4)
- Utilisez un modèle plus petit (yolov8n.pt au lieu de yolov8m.pt)

### "No CUDA GPUs are available"
- Vérifiez que les drivers NVIDIA sont installés : `nvidia-smi`
- Vérifiez que PyTorch avec CUDA est installé : `python3 -c "import torch; print(torch.cuda.is_available())"`
- Si False, réinstallez PyTorch avec support CUDA

### "torch.cuda.is_available() returns False"
- Vérifiez la compatibilité entre la version de CUDA et PyTorch
- Réinstallez PyTorch avec la bonne version de CUDA

## Installation des Drivers NVIDIA

Si `nvidia-smi` ne fonctionne pas :

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install nvidia-driver-535  # ou version plus récente
sudo reboot
```

### Vérification après reboot
```bash
nvidia-smi
```

## Notes

- Ultralytics YOLO utilise automatiquement le GPU s'il est disponible
- Si aucun GPU n'est disponible, l'entraînement se fera sur CPU (beaucoup plus lent)
- Pour un entraînement efficace, une GPU NVIDIA avec au moins 4GB de VRAM est recommandée

