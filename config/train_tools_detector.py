#!/usr/bin/env python3
"""
Script pour entraîner YOLO sur des outils (tournevis, clés, pinces, etc.)
"""
from ultralytics import YOLO
import os
import sys
import yaml
import multiprocessing
from pathlib import Path

def check_gpu():
    """Vérifie la disponibilité du GPU"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            # Obtenir la mémoire totale de la GPU
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return True, gpu_name, gpu_count, gpu_memory_gb
        else:
            return False, None, 0, 0
    except ImportError:
        return None, None, 0, 0

def find_dataset():
    """Trouve automatiquement le dataset dans le workspace"""
    workspace = os.path.expanduser('~/Mark3_ws')
    
    # Chercher les datasets Roboflow (format .yolov8)
    possible_datasets = [
        os.path.join(workspace, 'Screwdriver.v17i.yolov8'),
        os.path.join(workspace, 'dataset'),
    ]
    
    for dataset_path in possible_datasets:
        data_yaml = os.path.join(dataset_path, 'data.yaml')
        if os.path.exists(data_yaml):
            return dataset_path, data_yaml
    
    return None, None

def main():
    # Chercher automatiquement le dataset
    dataset_path, data_yaml = find_dataset()
    
    if not dataset_path or not data_yaml:
        print("ERREUR: Aucun dataset trouvé automatiquement.")
        print("\nDatasets recherchés:")
        print("   - ~/Mark3_ws/Screwdriver.v17i.yolov8/")
        print("   - ~/Mark3_ws/dataset/")
        
        manual_path = input("\nEntrez le chemin manuel du dataset (ou appuyez sur Entrée pour quitter): ").strip()
        if not manual_path:
            sys.exit(1)
        
        dataset_path = os.path.expanduser(manual_path)
        data_yaml = os.path.join(dataset_path, 'data.yaml')
    
    # Vérifier que le dataset existe
    if not os.path.exists(dataset_path):
        print(f"ERREUR: Dataset non trouvé à {dataset_path}")
        sys.exit(1)
    
    if not os.path.exists(data_yaml):
        print(f"ERREUR: Fichier data.yaml non trouvé à {data_yaml}")
        print("\nLe dataset doit contenir un fichier data.yaml")
        sys.exit(1)
    
    print(f"OK: Dataset trouvé: {dataset_path}")
    print(f"OK: Fichier de configuration: {data_yaml}")
    
    # Vérifier le GPU
    gpu_status, gpu_name, gpu_count, gpu_memory_gb = check_gpu()
    print(f"\nConfiguration matérielle:")
    if gpu_status is True:
        print(f"   GPU: {gpu_name} (CUDA disponible)")
        print(f"   Nombre de GPUs: {gpu_count}")
        print(f"   Mémoire GPU: {gpu_memory_gb:.2f} GB")
        print(f"   L'entraînement utilisera le GPU (beaucoup plus rapide)")
    elif gpu_status is False:
        print(f"   GPU: Non disponible")
        print(f"   L'entraînement utilisera le CPU (plus lent)")
        print(f"   AVERTISSEMENT: L'entraînement sera beaucoup plus lent sur CPU")
    else:
        print(f"   GPU: Impossible de vérifier (torch non installé)")
    
    # Afficher les statistiques du dataset
    try:
        with open(data_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        train_path = os.path.join(dataset_path, config.get('train', 'train/images'))
        val_path = os.path.join(dataset_path, config.get('val', 'valid/images'))
        
        train_images = len(list(Path(train_path).glob('*.jpg'))) + len(list(Path(train_path).glob('*.png')))
        val_images = len(list(Path(val_path).glob('*.jpg'))) + len(list(Path(val_path).glob('*.png')))
        
        print(f"\nStatistiques du dataset:")
        print(f"   Images d'entraînement: {train_images}")
        print(f"   Images de validation: {val_images}")
        print(f"   Classes: {config.get('nc', 'N/A')}")
        if 'names' in config:
            names = config['names']
            if isinstance(names, list):
                print(f"   Noms: {', '.join(names)}")
            elif isinstance(names, dict):
                print(f"   Noms: {', '.join(names.values())}")
    except Exception as e:
        print(f"AVERTISSEMENT: Impossible de lire les statistiques: {e}")
    
    # Choix du modèle
    print("Modèles disponibles:")
    print("1. yolov8n.pt - Nano (rapide, moins précis)")
    print("2. yolov8s.pt - Small (équilibré)")
    print("3. yolov8m.pt - Medium (plus précis, plus lent)")
    print("4. yolov8l.pt - Large (très précis, lent, nécessite beaucoup de VRAM)")
    
    choice = input("\nChoisissez le modèle (1-4, défaut: 1): ").strip()
    model_map = {'1': 'yolov8n.pt', '2': 'yolov8s.pt', '3': 'yolov8m.pt', '4': 'yolov8l.pt'}
    model_name = model_map.get(choice, 'yolov8n.pt')
    
    # Recommandations de batch size selon le modèle et la GPU
    recommended_batch = 16
    if gpu_status is True and gpu_memory_gb > 0:
        if model_name == 'yolov8l.pt':
            if gpu_memory_gb < 8:
                recommended_batch = 4
                print(f"\nAVERTISSEMENT: yolov8l.pt nécessite beaucoup de VRAM.")
                print(f"   Avec {gpu_memory_gb:.2f} GB, batch size recommandé: {recommended_batch}")
            elif gpu_memory_gb < 12:
                recommended_batch = 8
        elif model_name == 'yolov8m.pt':
            if gpu_memory_gb < 6:
                recommended_batch = 8
            elif gpu_memory_gb < 8:
                recommended_batch = 12
        elif model_name == 'yolov8s.pt':
            if gpu_memory_gb < 6:
                recommended_batch = 12
    
    # Paramètres d'entraînement
    epochs = int(input("Nombre d'epochs (défaut: 100): ").strip() or "100")
    batch_input = input(f"Taille du batch (défaut: {recommended_batch}, recommandé pour {model_name}): ").strip()
    batch = int(batch_input) if batch_input else recommended_batch
    imgsz = int(input("Taille des images (défaut: 640): ").strip() or "640")
    
    # Nombre de workers pour le chargement des données
    cpu_count = multiprocessing.cpu_count()
    recommended_workers = min(8, cpu_count)
    workers_input = input(f"Nombre de workers pour chargement données (défaut: {recommended_workers}, max: {cpu_count}): ").strip()
    workers = int(workers_input) if workers_input else recommended_workers
    
    # Avertissement si batch size trop élevé
    if gpu_status is True and gpu_memory_gb > 0:
        if model_name == 'yolov8l.pt' and batch > 8 and gpu_memory_gb < 8:
            print(f"\nAVERTISSEMENT: Batch size {batch} peut causer 'CUDA out of memory'")
            print(f"   Recommandation: batch <= 8 pour yolov8l.pt sur {gpu_memory_gb:.2f} GB GPU")
            confirm = input("   Continuer quand même? (o/n): ").strip().lower()
            if confirm != 'o':
                print("Entraînement annulé")
                sys.exit(0)
    
    print(f"\nChargement du modèle {model_name}...")
    model = YOLO(model_name)
    
    # Afficher le device utilisé
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device utilisé: {device.upper()}")
    except:
        pass
    
    print(f"Démarrage de l'entraînement...")
    print(f"   Dataset: {dataset_path}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch: {batch}")
    print(f"   Image size: {imgsz}")
    print(f"   Workers: {workers} (chargement parallèle des données)")
    if gpu_status is False:
        print(f"   AVERTISSEMENT: Entraînement sur CPU - très lent!")
    elif gpu_status is True:
        print(f"   GPU: {gpu_name} ({gpu_memory_gb:.2f} GB)")
    print()
    
    # Nettoyer le cache CUDA avant de commencer
    if gpu_status is True:
        try:
            import torch
            torch.cuda.empty_cache()
            print("Cache CUDA nettoyé")
        except:
            pass
    print("="*80)
    print("PROGRESSION DE L'ENTRAÎNEMENT")
    print("="*80)
    print()
    
    # Callback personnalisé pour afficher les métriques à chaque epoch
    def on_train_epoch_end(trainer):
        """Affiche les métriques à la fin de chaque epoch"""
        try:
            epoch = trainer.epoch + 1
            total_epochs = trainer.epochs
            
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{total_epochs}")
            print(f"{'='*80}")
            
            # Afficher les métriques disponibles
            if hasattr(trainer, 'metrics') and trainer.metrics:
                metrics = trainer.metrics
                print("\nMétriques d'entraînement:")
                for key, value in sorted(metrics.items()):
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
            
            # Afficher les résultats de validation si disponibles
            if hasattr(trainer, 'validator') and trainer.validator:
                print("\nMétriques de validation:")
                if hasattr(trainer.validator, 'metrics'):
                    val_metrics = trainer.validator.metrics
                    if isinstance(val_metrics, dict):
                        for key, value in sorted(val_metrics.items()):
                            if isinstance(value, (int, float)):
                                print(f"  {key}: {value:.4f}")
                            else:
                                print(f"  {key}: {value}")
            
            print()
        except Exception as e:
            print(f"Erreur dans le callback: {e}")
    
    # Ajouter le callback
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    
    # Entraîner avec verbose activé
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,  # Nombre de processus pour charger les données
        name='tools_detector',
        project='yolo_training',
        patience=20,  # Arrêt anticipé si pas d'amélioration pendant 20 epochs
        save=True,
        plots=True,
        verbose=True,  # Affichage détaillé
        # Augmentation de données
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1
    )
    
    # Obtenir le chemin réel du modèle depuis les résultats
    if hasattr(results, 'save_dir'):
        save_dir = results.save_dir
        best_model = os.path.join(save_dir, 'weights', 'best.pt')
        last_model = os.path.join(save_dir, 'weights', 'last.pt')
    else:
        # Fallback si save_dir n'est pas disponible
        project_dir = os.path.join('yolo_training', 'tools_detector')
        best_model = os.path.join(project_dir, 'weights', 'best.pt')
        last_model = os.path.join(project_dir, 'weights', 'last.pt')
    
    # Vérifier si le fichier existe, sinon chercher dans les sous-dossiers
    if not os.path.exists(best_model):
        import glob
        possible_paths = glob.glob(os.path.join('yolo_training', '*', 'weights', 'best.pt'))
        if possible_paths:
            best_model = possible_paths[-1]  # Prendre le plus récent
            last_model = best_model.replace('best.pt', 'last.pt')
    
    print(f"\nEntraînement terminé!")
    print(f"Meilleur modèle: {best_model}")
    print(f"Modèle final: {last_model}")
    
    if os.path.exists(best_model):
        print(f"\nMétriques finales (d'après la validation):")
        try:
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                if 'metrics/mAP50(B)' in metrics:
                    print(f"   mAP50: {metrics['metrics/mAP50(B)']:.4f}")
                if 'metrics/mAP50-95(B)' in metrics:
                    print(f"   mAP50-95: {metrics['metrics/mAP50-95(B)']:.4f}")
                if 'metrics/precision(B)' in metrics:
                    print(f"   Precision: {metrics['metrics/precision(B)']:.4f}")
                if 'metrics/recall(B)' in metrics:
                    print(f"   Recall: {metrics['metrics/recall(B)']:.4f}")
        except:
            pass
    
    print(f"\nPour utiliser le modèle, modifiez le launch file:")
    print(f"   'model_path': '{os.path.abspath(best_model)}'")
    print(f"\n   Ou lancez avec:")
    print(f"   ros2 run ObjectDetectionNode object_detection_node \\")
    print(f"     --ros-args -p model_path:={os.path.abspath(best_model)}")

if __name__ == '__main__':
    main()

