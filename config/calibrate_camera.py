#!/usr/bin/env python3
"""
Script de calibration de caméra OpenCV pour la caméra Nuroum
Utilise un damier (chessboard) pour calibrer les paramètres intrinsèques

Approche en 2 étapes:
1. Capture d'images: Prenez 20-30 photos du damier (sauvegardées automatiquement)
2. Analyse: Le script analyse toutes les images et trouve le damier

Usage:
1. Imprimez un damier (chessboard) depuis: https://docs.opencv.org/4.x/pattern.png
2. Lancez ce script: python3 calibrate_camera.py
3. Mode capture: Appuyez sur ESPACE pour capturer une image (20-30 images recommandées)
4. Appuyez sur 'q' pour passer à l'analyse
5. Le script analyse toutes les images et calcule la calibration
"""

import cv2
import numpy as np
import yaml
import os
from pathlib import Path

# Paramètres du damier (chessboard)
CHESSBOARD_SIZE = (7, 9)  # Nombre de coins internes (width, height)
SQUARE_SIZE = 0.021  # Taille d'un carré en mètres (21mm = 0.021m)

# Dossier pour sauvegarder les images
CALIBRATION_IMAGES_DIR = Path('calibration_images')
CALIBRATION_IMAGES_DIR.mkdir(exist_ok=True)

# Préparer les points objet (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def capture_images():
    """Étape 1: Capturer des images du damier"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra")
        exit(1)
    
    # Définir la résolution à 1080p (si supportée)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Nettoyer les anciennes images
    for img_file in CALIBRATION_IMAGES_DIR.glob('calib_*.jpg'):
        img_file.unlink()
    
    print("=" * 60)
    print("ETAPE 1: CAPTURE D'IMAGES")
    print("=" * 60)
    print("Instructions:")
    print("- Prenez 20-30 photos du damier sous différents angles")
    print("- Appuyez sur ESPACE pour capturer une image")
    print("- Vous pouvez bouger la caméra/damier librement entre les captures")
    print("- Appuyez sur 'q' pour terminer la capture et passer à l'analyse")
    print(f"- Images sauvegardées dans: {CALIBRATION_IMAGES_DIR.absolute()}")
    print("=" * 60)
    
    image_count = 0
    actual_width = None
    actual_height = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur: Impossible de lire la caméra")
            break
        
        # Capturer la résolution réelle
        if actual_width is None:
            actual_height, actual_width = frame.shape[:2]
            print(f"\nResolution reelle detectee: {actual_width}x{actual_height}")
            if actual_width != 1920 or actual_height != 1080:
                print("ATTENTION: La resolution reelle est differente de 1080p!")
                print("La calibration sera faite avec la resolution reelle.\n")
        
        # Afficher le compteur
        status_text = f"Images capturees: {image_count} (Appuyez sur ESPACE pour capturer)"
        cv2.putText(frame, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Appuyez sur 'q' pour analyser les images", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Capture - Calibration Camera', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # ESPACE pour capturer
            filename = CALIBRATION_IMAGES_DIR / f'calib_{image_count:03d}.jpg'
            cv2.imwrite(str(filename), frame)
            image_count += 1
            print(f"Image {image_count} sauvegardee: {filename.name}")
        
        elif key == ord('q'):  # 'q' pour quitter
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nCapture terminee: {image_count} images sauvegardees")
    return actual_width, actual_height, image_count

def analyze_images(actual_width, actual_height):
    """Étape 2: Analyser les images sauvegardées pour trouver le damier"""
    print("\n" + "=" * 60)
    print("ETAPE 2: ANALYSE DES IMAGES")
    print("=" * 60)
    
    image_files = sorted(CALIBRATION_IMAGES_DIR.glob('calib_*.jpg'))
    
    if len(image_files) == 0:
        print("Erreur: Aucune image trouvee dans", CALIBRATION_IMAGES_DIR)
        return None, None, None, None
    
    print(f"Analyse de {len(image_files)} images...")
    
    objpoints = []  # Points 3D dans l'espace réel
    imgpoints = []  # Points 2D dans le plan image
    valid_images = []
    
    for i, img_file in enumerate(image_files):
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"  Erreur: Impossible de lire {img_file.name}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Chercher les coins du damier
        ret_corners, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        
        if ret_corners:
            # Affiner la position des coins
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            valid_images.append(img_file.name)
            print(f"  ✓ {img_file.name}: Damier detecte")
        else:
            print(f"  ✗ {img_file.name}: Damier non detecte")
    
    print(f"\nResultat: {len(valid_images)}/{len(image_files)} images valides")
    
    if len(objpoints) < 10:
        print(f"\nErreur: Pas assez d'images valides ({len(objpoints)}). Minimum 10 images requises.")
        print("Conseils:")
        print("- Assurez-vous que le damier est bien visible dans les images")
        print("- Vérifiez que le damier a {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} coins internes")
        print("- Relancez la capture avec plus d'images")
        return None, None, None, None
    
    # Calibration
    print(f"\nCalibration en cours avec {len(objpoints)} images valides...")
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    # Calculer l'erreur de reprojection
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                          camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    mean_error = mean_error / len(objpoints)
    
    print(f"Erreur moyenne de reprojection: {mean_error:.3f} pixels")
    print("(Plus bas est mieux, typiquement < 0.5 pixels)")
    
    # Afficher les résultats
    print("\nMatrice de caméra:")
    print(camera_matrix)
    print("\nCoefficients de distortion:")
    print(dist_coeffs.flatten())
    
    return camera_matrix, dist_coeffs, mean_error, len(objpoints)

def save_calibration(camera_matrix, dist_coeffs, mean_error, num_images, actual_width, actual_height):
    """Sauvegarder la calibration en YAML"""
    calibration_data = {
        'image_width': int(actual_width),
        'image_height': int(actual_height),
        'camera_name': 'nuroum_1080p',
        'camera_matrix': {
            'rows': 3,
            'cols': 3,
            'data': camera_matrix.flatten().tolist()
        },
        'distortion_coefficients': {
            'rows': 1,
            'cols': len(dist_coeffs),
            'data': dist_coeffs.flatten().tolist()
        },
        'reprojection_error': float(mean_error),
        'num_images': num_images
    }
    
    output_file = 'camera_calibration.yaml'
    with open(output_file, 'w') as f:
        yaml.dump(calibration_data, f, default_flow_style=False)
    
    print(f"\n✓ Calibration sauvegardee dans: {os.path.abspath(output_file)}")
    print("\nVous pouvez maintenant utiliser ce fichier avec ObjectDetectionNode:")
    print(f"  ros2 run ObjectDetectionNode object_detection_node \\")
    print(f"    --ros-args -p calibration_file:={os.path.abspath(output_file)}")

def main():
    # Étape 1: Capturer les images
    actual_width, actual_height, num_captured = capture_images()
    
    if num_captured == 0:
        print("Aucune image capturee. Arret.")
        return
    
    # Étape 2: Analyser les images
    camera_matrix, dist_coeffs, mean_error, num_valid = analyze_images(actual_width, actual_height)
    
    if camera_matrix is None:
        print("\nCalibration echouee. Relancez le script pour capturer plus d'images.")
        return
    
    # Étape 3: Sauvegarder
    save_calibration(camera_matrix, dist_coeffs, mean_error, num_valid, actual_width, actual_height)
    
    print("\n" + "=" * 60)
    print("CALIBRATION TERMINEE AVEC SUCCES!")
    print("=" * 60)

if __name__ == '__main__':
    main()
