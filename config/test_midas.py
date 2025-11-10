#!/usr/bin/env python3
"""
Script de test pour MiDaS - Estimation de profondeur monoculaire
"""
import sys
import os

# Forcer l'utilisation du venv Python si disponible
venv_python = os.path.expanduser('~/venv/bin/python3')
if os.path.exists(venv_python) and sys.executable != venv_python:
    try:
        import torch
    except ImportError:
        os.execv(venv_python, [venv_python] + sys.argv)

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def test_midas():
    print("=" * 60)
    print("Test MiDaS - Estimation de profondeur monoculaire")
    print("=" * 60)
    
    # Vérifier PyTorch
    print(f"\n1. Vérification PyTorch...")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Charger MiDaS
    print(f"\n2. Chargement du modèle MiDaS...")
    try:
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
            print("   Modèle chargé sur GPU")
        else:
            print("   Modèle chargé sur CPU")
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.small_transform
        print("   Transformations chargées")
    except Exception as e:
        print(f"   ERREUR lors du chargement: {e}")
        return False
    
    # Charger une image
    print(f"\n3. Chargement de l'image...")
    
    # Essayer d'utiliser la caméra
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("   Caméra non disponible, création d'une image de test...")
        # Créer une image de test
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (200, 150), (440, 330), (255, 255, 255), -1)
        cv2.circle(test_image, (320, 240), 50, (128, 128, 128), -1)
        img = test_image
    else:
        print("   Capture d'une image depuis la caméra...")
        ret, frame = cap.read()
        if ret:
            img = frame
            print(f"   Image capturée: {img.shape}")
        else:
            print("   Échec de la capture, création d'une image de test...")
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        cap.release()
    
    # Réduire la résolution pour accélérer
    max_dim = 384
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        print(f"   Image redimensionnée: {new_w}x{new_h}")
    else:
        img_resized = img
    
    # Préparer l'image pour MiDaS
    print(f"\n4. Traitement avec MiDaS...")
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_batch = input_batch.to(device)
    
    # Prédire la profondeur
    print("   Calcul de la carte de profondeur...")
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    # Convertir en numpy
    depth_map = prediction.cpu().numpy()
    
    print(f"   Carte de profondeur calculée: {depth_map.shape}")
    print(f"   Valeurs min: {np.min(depth_map):.2f}, max: {np.max(depth_map):.2f}, mean: {np.mean(depth_map):.2f}")
    
    # Normaliser pour affichage
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)
    
    # Afficher les résultats
    print(f"\n5. Affichage des résultats...")
    
    # Créer une figure avec matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title("Image originale")
    axes[0].axis('off')
    
    axes[1].imshow(depth_map, cmap='magma')
    axes[1].set_title("Carte de profondeur (raw)")
    axes[1].axis('off')
    
    axes[2].imshow(depth_colored)
    axes[2].set_title("Carte de profondeur (colormap)")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Sauvegarder
    output_dir = Path.home() / "Mark3_ws" / "config" / "midas_test_output"
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "midas_test_result.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Résultats sauvegardés dans: {output_path}")
    
    # Test de conversion en mètres
    print(f"\n6. Test de conversion en mètres...")
    max_depth = np.max(depth_map)
    min_depth = np.min(depth_map)
    
    # Prendre un point au centre
    center_y, center_x = depth_map.shape[0] // 2, depth_map.shape[1] // 2
    center_depth_value = depth_map[center_y, center_x]
    
    # Normaliser
    if max_depth > min_depth:
        normalized = 1.0 - (center_depth_value - min_depth) / (max_depth - min_depth)
    else:
        normalized = 0.5
    
    # Convertir avec différents facteurs d'échelle
    scale_factors = [0.3, 0.5, 0.7, 1.0]
    print(f"   Profondeur au centre (valeur brute): {center_depth_value:.2f}")
    print(f"   Profondeur normalisée: {normalized:.3f}")
    print(f"   Conversions en mètres avec différents facteurs:")
    for scale in scale_factors:
        depth_meters = normalized * scale
        print(f"     - Facteur {scale}: {depth_meters:.3f}m")
    
    # Afficher la figure
    print(f"\n7. Affichage de la fenêtre (fermez-la pour terminer)...")
    plt.show()
    
    print(f"\n" + "=" * 60)
    print("Test terminé avec succès!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = test_midas()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

