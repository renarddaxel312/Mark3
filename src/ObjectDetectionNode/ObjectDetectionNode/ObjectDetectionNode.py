#!/usr/bin/env python3
import sys
import os

# Forcer l'utilisation du venv Python si disponible
venv_python = os.path.expanduser('~/venv/bin/python3')
if os.path.exists(venv_python) and sys.executable != venv_python:
    try:
        import ultralytics
    except ImportError:
        # Si ultralytics n'est pas disponible, utiliser le venv
        os.execv(venv_python, [venv_python] + sys.argv)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from object_detection_interface.srv import DetectObject3D
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

# MiDaS pour estimation de profondeur
try:
    import torch
    MIDAS_AVAILABLE = True
except ImportError:
    MIDAS_AVAILABLE = False


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        
        self.bridge = CvBridge()
        self.latest_image = None
        self.captured_image = None  # Image capturée pour MiDaS
        self.depth_map_cache = None
        self.depth_map_timestamp = None
        self.depth_cache_duration = 0.5  # Cache la carte de profondeur pendant 0.5 secondes
        
        # Paramètres pour la capture stable
        self.capture_confidence_threshold = self.declare_parameter('capture_confidence_threshold', 0.7).value
        self.stable_frames_required = self.declare_parameter('stable_frames_required', 5).value
        self.last_detection_bbox = None
        self.stable_detection_count = 0
        self.last_capture_time = 0.0
        self.min_capture_interval = 1.0  # Minimum 1 seconde entre captures
        
        self.image_sub = self.create_subscription(
            Image, '/camera/raw', self.image_callback, 10)
        
        self.service = self.create_service(
            DetectObject3D, 'detect_object_3d', self.detect_callback)
        
        model_path = self.declare_parameter('model_path', 'yolov8n.pt').value
        self.model = YOLO(model_path)
        self.get_logger().info(f"YOLO model loaded: {model_path}")
        
        self.fixed_depth = self.declare_parameter('fixed_depth', 0.5).value
        self.use_midas = self.declare_parameter('use_midas', False).value
        self.midas_depth_scale = self.declare_parameter('midas_depth_scale', 0.5).value
        
        # Estimation basée sur la taille du bbox
        self.use_bbox_size = self.declare_parameter('use_bbox_size', True).value
        self.known_object_size = self.declare_parameter('known_object_size', 0.20).value  # Taille réelle en mètres (20 cm pour un tournevis)
        
        # Initialiser MiDaS si disponible et activé
        self.midas_model = None
        self.midas_transform = None
        if self.use_midas and MIDAS_AVAILABLE:
            try:
                self.get_logger().info("Chargement du modèle MiDaS pour estimation de profondeur...")
                # Charger MiDaS avec trust_repo=True pour éviter les warnings
                self.midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
                self.midas_model.eval()
                if torch.cuda.is_available():
                    self.midas_model = self.midas_model.cuda()
                    self.get_logger().info("MiDaS chargé sur GPU")
                else:
                    self.get_logger().info("MiDaS chargé sur CPU")
                
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                self.midas_transform = midas_transforms.small_transform
                self.get_logger().info("Modèle MiDaS chargé avec succès")
            except Exception as e:
                self.get_logger().error(f"Impossible de charger MiDaS: {e}", exc_info=True)
                self.get_logger().warn("Utilisation de profondeur fixe en fallback")
                self.use_midas = False
                self.midas_model = None
        else:
            if not MIDAS_AVAILABLE:
                self.get_logger().warn("PyTorch non disponible. Utilisation de profondeur fixe.")
            self.get_logger().info(f"Profondeur fixe (Z) configurée à: {self.fixed_depth} mètres")
        
        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_width = None
        self.image_height = None
        self.load_camera_calibration()
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera_frame = self.declare_parameter('camera_frame', 'camera_frame').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        
        self.get_logger().info("Object Detection Node ready")
        self.get_logger().info(f"Camera frame: {self.camera_frame}, Base frame: {self.base_frame}")
    
    def load_camera_calibration(self):
        calib_path = self.declare_parameter('calibration_file', '').value
        
        if not calib_path:
            self.get_logger().warn("No calibration file specified. Using default values.")
            self.setup_default_calibration()
            return
        
        if not os.path.exists(calib_path):
            self.get_logger().error(f"Calibration file not found: {calib_path}")
            self.setup_default_calibration()
            return
        
        try:
            if calib_path.endswith('.yaml') or calib_path.endswith('.yml'):
                self.load_yaml_calibration(calib_path)
            elif calib_path.endswith('.npz'):
                self.load_npz_calibration(calib_path)
            else:
                self.get_logger().error(f"Unsupported calibration file format: {calib_path}")
                self.setup_default_calibration()
        except Exception as e:
            self.get_logger().error(f"Error loading calibration: {e}")
            self.setup_default_calibration()
    
    def load_yaml_calibration(self, calib_path):
        with open(calib_path, 'r') as f:
            calib = yaml.safe_load(f)
        
        if 'camera_matrix' in calib:
            data = calib['camera_matrix']
            if isinstance(data, dict) and 'data' in data:
                self.camera_matrix = np.array(data['data']).reshape(data['rows'], data['cols'])
            else:
                self.camera_matrix = np.array(data)
        else:
            raise ValueError("camera_matrix not found in calibration file")
        
        if 'distortion_coefficients' in calib:
            data = calib['distortion_coefficients']
            if isinstance(data, dict) and 'data' in data:
                dist_array = np.array(data['data'])
                # Gérer le cas où cols=1 mais il y a plusieurs valeurs (5 coefficients)
                if 'rows' in data and 'cols' in data:
                    if data['rows'] == 1 and len(dist_array) > 1:
                        # Cas spécial: rows=1, cols=1 mais 5 valeurs -> reshape en (5, 1)
                        self.dist_coeffs = dist_array.reshape(-1, 1)
                    else:
                        self.dist_coeffs = dist_array.reshape(data['rows'], data['cols'])
                else:
                    self.dist_coeffs = dist_array.reshape(-1, 1)
            else:
                dist_array = np.array(data)
                self.dist_coeffs = dist_array.reshape(-1, 1) if dist_array.ndim == 1 else dist_array
        else:
            self.dist_coeffs = np.zeros((5, 1))
        
        if 'image_width' in calib:
            self.image_width = calib['image_width']
        if 'image_height' in calib:
            self.image_height = calib['image_height']
        
        self.get_logger().info(f"Camera calibration loaded from YAML: {calib_path}")
        self.get_logger().info(f"Camera matrix:\n{self.camera_matrix}")
        self.get_logger().info(f"Distortion coefficients: {self.dist_coeffs.flatten()}")
    
    def load_npz_calibration(self, calib_path):
        calib = np.load(calib_path)
        self.camera_matrix = calib['camera_matrix']
        self.dist_coeffs = calib['distortion_coefficients'] if 'distortion_coefficients' in calib else np.zeros((4, 1))
        self.get_logger().info(f"Camera calibration loaded from NPZ: {calib_path}")
    
    def setup_default_calibration(self):
        self.get_logger().warn("Using default calibration for 1080p camera (Nuroum)")
        fx = fy = 1500.0
        cx = 960.0
        cy = 540.0
        self.camera_matrix = np.array([[fx, 0, cx],
                                      [0, fy, cy],
                                      [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1))
        self.image_width = 1920
        self.image_height = 1080
    
    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.image_width is None:
                self.image_height, self.image_width = self.latest_image.shape[:2]
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
    
    def estimate_depth_midas(self, image, center_x, center_y):
        """Estime la profondeur à un point spécifique de l'image using MiDaS"""
        if self.midas_model is None or self.midas_transform is None:
            return None
        
        try:
            import time
            current_time = time.time()
            
            # Variables pour le scale (utilisées si on recalcule)
            scale_factor_x = 1.0
            scale_factor_y = 1.0
            
            # Utiliser le cache si disponible et récent
            if (self.depth_map_cache is not None and 
                self.depth_map_timestamp is not None and
                current_time - self.depth_map_timestamp < self.depth_cache_duration):
                depth_map = self.depth_map_cache
                self.get_logger().debug("Utilisation de la carte de profondeur en cache")
            else:
                # Réduire la résolution pour accélérer le traitement
                # MiDaS fonctionne bien avec des images plus petites
                max_dimension = 384  # Réduire à 384px pour la dimension max
                h, w = image.shape[:2]
                if max(h, w) > max_dimension:
                    scale = max_dimension / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    scale_factor_x = w / new_w
                    scale_factor_y = h / new_h
                else:
                    img_resized = image
                    scale_factor_x = 1.0
                    scale_factor_y = 1.0
                
                # Préparer l'image pour MiDaS
                img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                input_batch = self.midas_transform(img)
                
                # Déplacer sur GPU si disponible
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                input_batch = input_batch.to(device)
                
                # Prédire la profondeur
                with torch.no_grad():
                    prediction = self.midas_model(input_batch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                
                # Convertir en numpy
                depth_map = prediction.cpu().numpy()
                
                # Mettre en cache
                self.depth_map_cache = depth_map
                self.depth_map_timestamp = current_time
                
                self.get_logger().debug(f"MiDaS: nouvelle carte de profondeur calculée ({new_w}x{new_h})")
            
            # Ajuster les coordonnées du centre si l'image a été redimensionnée
            if scale_factor_x != 1.0 or scale_factor_y != 1.0:
                center_x_scaled = center_x / scale_factor_x
                center_y_scaled = center_y / scale_factor_y
            else:
                center_x_scaled = center_x
                center_y_scaled = center_y
            
            # Extraire la profondeur au centre du bounding box
            center_x_int = int(np.clip(center_x_scaled, 0, depth_map.shape[1] - 1))
            center_y_int = int(np.clip(center_y_scaled, 0, depth_map.shape[0] - 1))
            
            # Prendre une moyenne sur une petite région autour du centre
            region_size = 5
            y_start = max(0, center_y_int - region_size)
            y_end = min(depth_map.shape[0], center_y_int + region_size)
            x_start = max(0, center_x_int - region_size)
            x_end = min(depth_map.shape[1], center_x_int + region_size)
            
            depth_value = np.mean(depth_map[y_start:y_end, x_start:x_end])
            
            # MiDaS retourne des valeurs de profondeur inverses (plus proche = valeur plus grande)
            # Basé sur les tests, les valeurs sont généralement entre 400-900
            # Pour convertir en mètres, on utilise une normalisation et un facteur d'échelle
            
            # Normaliser par rapport à la valeur max de la carte de profondeur
            max_depth = np.max(depth_map)
            min_depth = np.min(depth_map)
            
            if max_depth > min_depth:
                # Normaliser entre 0 et 1 (inverser car plus grande valeur = plus proche)
                normalized = 1.0 - (depth_value - min_depth) / (max_depth - min_depth)
            else:
                normalized = 0.5
            
            # Convertir en mètres avec le facteur d'échelle
            # Le facteur d'échelle représente la distance maximale en mètres
            # Basé sur les tests, un facteur de 0.5 donne des résultats raisonnables (0.3-0.7m)
            depth_meters = normalized * self.midas_depth_scale
            
            # Limiter la profondeur à une plage raisonnable (0.1m à 2m pour un robot de table)
            depth_meters = np.clip(depth_meters, 0.1, 2.0)
            
            self.get_logger().info(
                f"MiDaS: depth_value={depth_value:.2f}, normalized={normalized:.3f}, "
                f"depth_meters={depth_meters:.3f}m (scale={self.midas_depth_scale})"
            )
            
            return depth_meters
            
        except Exception as e:
            self.get_logger().error(f"Erreur estimation MiDaS: {e}")
            return None
    
    def estimate_depth_from_bbox_size(self, bbox, image):
        """Estime la profondeur basée sur la taille du bounding box"""
        if self.camera_matrix is None:
            return None
        
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Utiliser la plus grande dimension (largeur ou hauteur) pour plus de robustesse
        bbox_size_pixels = max(bbox_width, bbox_height)
        
        if bbox_size_pixels <= 0:
            return None
        
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        focal_length = (fx + fy) / 2.0  # Moyenne des focales
        
        # Formule: distance = (taille_réelle * focale) / taille_image
        # On utilise la taille réelle de l'objet (en mètres) et la taille en pixels
        estimated_depth = (self.known_object_size * focal_length) / bbox_size_pixels
        
        # Limiter à une plage raisonnable (0.1m à 1.5m pour un robot de table)
        estimated_depth = np.clip(estimated_depth, 0.1, 1.5)
        
        self.get_logger().info(
            f"Estimation bbox: taille_pixels={bbox_size_pixels:.1f}, "
            f"taille_réelle={self.known_object_size:.3f}m, "
            f"focale={focal_length:.1f}, "
            f"profondeur={estimated_depth:.3f}m"
        )
        
        return estimated_depth
    
    def estimate_3d_position(self, bbox, object_class, known_size=None, image=None):
        # Utiliser l'image fournie ou l'image actuelle
        if image is None:
            image = self.latest_image
        
        if image is None or self.camera_matrix is None:
            return None, "No image or calibration available"
        
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # Estimer la profondeur : priorité à bbox_size, puis MiDaS, puis fixe
        z_cam = None
        depth_source = None
        
        if self.use_bbox_size:
            estimated_depth = self.estimate_depth_from_bbox_size(bbox, image)
            if estimated_depth is not None:
                z_cam = estimated_depth
                depth_source = "bbox_size"
        
        if z_cam is None and self.use_midas and self.midas_model is not None:
            image_for_midas = self.captured_image if self.captured_image is not None else image
            estimated_depth = self.estimate_depth_midas(image_for_midas, center_x, center_y)
            if estimated_depth is not None:
                z_cam = estimated_depth
                depth_source = "MiDaS"
        
        if z_cam is None:
            z_cam = self.fixed_depth
            depth_source = "fixe"
        
        # Calcul de X et Y avec Z estimé (projection inverse)
        # Formule: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
        x_cam = (center_x - cx) * z_cam / fx
        y_cam = (center_y - cy) * z_cam / fy
        
        # Log pour debug
        self.get_logger().info(
            f"Calcul 3D: bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) "
            f"center=({center_x:.1f},{center_y:.1f}) "
            f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} "
            f"depth={z_cam:.3f}m ({depth_source}) "
            f"-> X={x_cam:.3f}m Y={y_cam:.3f}m Z={z_cam:.3f}m"
        )
        
        point_3d = np.array([x_cam, y_cam, z_cam])
        
        return point_3d, None
    
    def transform_to_base_frame(self, point_camera):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            point_stamped = tf2_geometry_msgs.PointStamped()
            point_stamped.header.frame_id = self.camera_frame
            point_stamped.point.x = float(point_camera[0])
            point_stamped.point.y = float(point_camera[1])
            point_stamped.point.z = float(point_camera[2])
            
            point_transformed = tf2_geometry_msgs.do_transform_point(
                point_stamped, transform
            )
            
            return np.array([
                point_transformed.point.x,
                point_transformed.point.y,
                point_transformed.point.z
            ]), None
        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}. Using camera frame coordinates.")
            return point_camera, "TF transform unavailable"
    
    def should_capture_frame(self, bbox, confidence):
        """Détermine si on doit capturer une nouvelle frame pour MiDaS"""
        import time
        current_time = time.time()
        
        # Vérifier l'intervalle minimum entre captures
        if current_time - self.last_capture_time < self.min_capture_interval:
            return False
        
        # Vérifier le seuil de confiance
        if confidence < self.capture_confidence_threshold:
            self.stable_detection_count = 0
            return False
        
        # Vérifier la stabilité de la détection (même position approximative)
        if self.last_detection_bbox is not None:
            x1, y1, x2, y2 = bbox
            last_x1, last_y1, last_x2, last_y2 = self.last_detection_bbox
            
            # Calculer le centre des deux bounding boxes
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            last_center_x = (last_x1 + last_x2) / 2.0
            last_center_y = (last_y1 + last_y2) / 2.0
            
            # Distance entre les centres
            distance = np.sqrt((center_x - last_center_x)**2 + (center_y - last_center_y)**2)
            
            # Si la position a changé de plus de 20 pixels, réinitialiser le compteur
            if distance > 20.0:
                self.stable_detection_count = 0
                self.last_detection_bbox = bbox
                return False
        
        # Mettre à jour le compteur de stabilité
        self.stable_detection_count += 1
        self.last_detection_bbox = bbox
        
        # Capturer si stable pendant assez de frames
        if self.stable_detection_count >= self.stable_frames_required:
            self.stable_detection_count = 0
            self.last_capture_time = current_time
            return True
        
        return False
    
    def detect_callback(self, request, response):
        if self.latest_image is None:
            response.success = False
            response.message = "No camera image available"
            self.get_logger().warn("Service appelé mais aucune image disponible")
            return response
        
        try:
            # Faire une copie de l'image pour éviter qu'elle change pendant le traitement
            image_copy = self.latest_image.copy()
            
            # Seuil de confiance minimum pour la détection
            conf_threshold = 0.25
            results = self.model(image_copy, conf=conf_threshold, verbose=False)
            
            if len(results) == 0 or len(results[0].boxes) == 0:
                response.success = False
                response.message = "No objects detected"
                self.get_logger().debug("No detections with confidence >= " + str(conf_threshold))
                # Réinitialiser le compteur si pas de détection
                self.stable_detection_count = 0
                return response
            
            best_detection = None
            best_confidence = 0.0
            
            # Log toutes les détections trouvées
            all_detections = []
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls]
                all_detections.append(f"{class_name}({conf:.2f})")
            
            self.get_logger().info(f"Toutes les détections YOLO: {', '.join(all_detections) if all_detections else 'aucune'}")
            
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls]
                
                # Log toutes les détections pour debug
                self.get_logger().debug(f"Detection: {class_name} (conf: {conf:.2f})")
                
                # Filtrer uniquement les tournevis (screwdriver)
                if class_name.lower() != 'screwdriver':
                    self.get_logger().debug(f"Ignoré (pas un tournevis): {class_name}")
                    continue
                
                # Si une classe spécifique est demandée, vérifier
                if request.object_class and request.object_class.lower() != class_name.lower():
                    self.get_logger().debug(f"Classe demandée '{request.object_class}' ne correspond pas à '{class_name}'")
                    continue
                
                if conf > best_confidence:
                    best_confidence = conf
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    best_detection = {
                        'bbox': [x1, y1, x2, y2],
                        'class': class_name,
                        'confidence': conf
                    }
            
            if best_detection is None:
                self.get_logger().warn(
                    f"Tournevis non trouvé. Détections disponibles: {', '.join(all_detections) if all_detections else 'aucune'}. "
                    f"Classe demandée: '{request.object_class}'"
                )
                response.success = False
                response.message = f"Object class '{request.object_class}' not found" if request.object_class else "No objects detected"
                return response
            
            # Vérifier si on doit capturer une nouvelle frame pour MiDaS
            if self.use_midas and self.midas_model is not None:
                if self.should_capture_frame(best_detection['bbox'], best_detection['confidence']):
                    self.captured_image = image_copy.copy()
                    self.get_logger().info(
                        f"Frame capturée pour MiDaS: conf={best_detection['confidence']:.2f}, "
                        f"bbox={best_detection['bbox']}"
                    )
                    # Réinitialiser le cache de profondeur pour forcer un nouveau calcul
                    self.depth_map_cache = None
            
            # Utiliser l'image capturée si disponible, sinon l'image copiée
            image_for_depth = self.captured_image if self.captured_image is not None else image_copy
            
            self.get_logger().info(
                f"Calcul position 3D pour {best_detection['class']} "
                f"(conf: {best_detection['confidence']:.2f}, "
                f"image_capturée: {self.captured_image is not None})"
            )
            
            position_3d_cam, error = self.estimate_3d_position(
                best_detection['bbox'],
                best_detection['class'],
                request.known_object_size if request.known_object_size > 0 else None,
                image_for_depth
            )
            
            if error:
                response.success = False
                response.message = error
                self.get_logger().error(f"Erreur estimation 3D: {error}")
                return response
            
            if position_3d_cam is None:
                response.success = False
                response.message = "Failed to estimate 3D position"
                self.get_logger().error("Position 3D est None")
                return response
            
            response.position_camera_frame = Point()
            response.position_camera_frame.x = float(position_3d_cam[0])
            response.position_camera_frame.y = float(position_3d_cam[1])
            response.position_camera_frame.z = float(position_3d_cam[2])
            
            self.get_logger().info(
                f"Position caméra calculée: X={response.position_camera_frame.x:.3f}m "
                f"Y={response.position_camera_frame.y:.3f}m Z={response.position_camera_frame.z:.3f}m"
            )
            
            position_3d_base, tf_error = self.transform_to_base_frame(position_3d_cam)
            
            response.position_3d = Point()
            response.position_3d.x = float(position_3d_base[0])
            response.position_3d.y = float(position_3d_base[1])
            response.position_3d.z = float(position_3d_base[2])
            
            response.success = True
            response.message = "Object detected and 3D position estimated" + (f" ({tf_error})" if tf_error else "")
            response.detected_class = best_detection['class']
            response.confidence = best_detection['confidence']
            
            self.get_logger().info(
                f"✓ Service réussi: {best_detection['class']} (conf: {best_detection['confidence']:.2f}) "
                f"at camera frame: ({position_3d_cam[0]:.3f}, {position_3d_cam[1]:.3f}, {position_3d_cam[2]:.3f}) "
                f"base frame: ({position_3d_base[0]:.3f}, {position_3d_base[1]:.3f}, {position_3d_base[2]:.3f})"
            )
            self.get_logger().info(
                f"  Réponse envoyée: success=True, position_3d=({response.position_3d.x:.3f}, {response.position_3d.y:.3f}, {response.position_3d.z:.3f})"
            )
            
        except Exception as e:
            self.get_logger().error(f"Error in detection: {e}")
            response.success = False
            response.message = f"Detection error: {str(e)}"
        
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

