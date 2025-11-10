#!/usr/bin/env python3
"""
Script pour vérifier que le dataset est correctement configuré
"""
import os
import yaml
from pathlib import Path

def verify_dataset(dataset_path):
    """Vérifie la structure et la configuration du dataset"""
    print(f"Vérification du dataset: {dataset_path}\n")
    
    errors = []
    warnings = []
    
    # Vérifier que le répertoire existe
    if not os.path.exists(dataset_path):
        errors.append(f"Le répertoire {dataset_path} n'existe pas")
        return errors, warnings
    
    # Vérifier data.yaml
    data_yaml = os.path.join(dataset_path, 'data.yaml')
    if not os.path.exists(data_yaml):
        errors.append(f"Fichier data.yaml non trouvé dans {dataset_path}")
        return errors, warnings
    
    # Lire la configuration
    try:
        with open(data_yaml, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        errors.append(f"Erreur lors de la lecture de data.yaml: {e}")
        return errors, warnings
    
    print("OK: Fichier data.yaml trouvé et valide")
    
    # Vérifier les chemins
    train_path = os.path.join(dataset_path, config.get('train', 'train/images'))
    val_path = os.path.join(dataset_path, config.get('val', 'valid/images'))
    
    # Vérifier train
    if not os.path.exists(train_path):
        errors.append(f"Répertoire d'entraînement non trouvé: {train_path}")
    else:
        train_images = list(Path(train_path).glob('*.jpg')) + list(Path(train_path).glob('*.png'))
        train_labels_path = Path(train_path).parent / 'labels'
        train_labels = list(train_labels_path.glob('*.txt')) if train_labels_path.exists() else []
        if not train_images:
            errors.append(f"Aucune image trouvée dans {train_path}")
        else:
            print(f"OK: Images d'entraînement: {len(train_images)}")
            if train_labels and len(train_labels) != len(train_images):
                warnings.append(f"Nombre de labels ({len(train_labels)}) ne correspond pas au nombre d'images ({len(train_images)})")
    
    # Vérifier validation
    if not os.path.exists(val_path):
        errors.append(f"Répertoire de validation non trouvé: {val_path}")
    else:
        val_images = list(Path(val_path).glob('*.jpg')) + list(Path(val_path).glob('*.png'))
        val_labels_path = Path(val_path).parent / 'labels'
        val_labels = list(val_labels_path.glob('*.txt')) if val_labels_path.exists() else []
        if not val_images:
            errors.append(f"Aucune image trouvée dans {val_path}")
        else:
            print(f"OK: Images de validation: {len(val_images)}")
            if val_labels and len(val_labels) != len(val_images):
                warnings.append(f"Nombre de labels ({len(val_labels)}) ne correspond pas au nombre d'images ({len(val_images)})")
    
    # Vérifier les classes
    nc = config.get('nc', 0)
    names = config.get('names', [])
    
    if nc == 0:
        errors.append("Nombre de classes (nc) non défini ou égal à 0")
    else:
        print(f"OK: Nombre de classes: {nc}")
    
    if not names:
        errors.append("Noms des classes non définis")
    else:
        if isinstance(names, list):
            print(f"OK: Classes: {', '.join(names)}")
        elif isinstance(names, dict):
            print(f"OK: Classes: {', '.join(names.values())}")
    
    return errors, warnings

def main():
    workspace = os.path.expanduser('~/Mark3_ws')
    
    # Chercher les datasets
    datasets = [
        os.path.join(workspace, 'Screwdriver.v17i.yolov8'),
        os.path.join(workspace, 'dataset'),
    ]
    
    found = False
    for dataset_path in datasets:
        if os.path.exists(dataset_path):
            found = True
            errors, warnings = verify_dataset(dataset_path)
            
            print("\n" + "="*60)
            if errors:
                print("ERREURS TROUVEES:")
                for error in errors:
                    print(f"   - {error}")
            else:
                print("OK: Dataset valide et prêt pour l'entraînement!")
            
            if warnings:
                print("\nAVERTISSEMENTS:")
                for warning in warnings:
                    print(f"   - {warning}")
            print("="*60)
            break
    
    if not found:
        print("ERREUR: Aucun dataset trouvé dans:")
        for d in datasets:
            print(f"   - {d}")
        print("\nPlacez votre dataset dans l'un de ces répertoires")

if __name__ == '__main__':
    main()

