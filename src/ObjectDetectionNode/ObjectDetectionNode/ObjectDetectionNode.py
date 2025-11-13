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

import json
import yaml

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from object_detection_interface.srv import DetectObject3D
from ultralytics import YOLO
from tf2_geometry_msgs import do_transform_point
from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_width = None
        self.image_height = None
        self.camera_matrix = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.warn_missing_calibration_logged = False
        
        self.calibration_file = self.declare_parameter('calibration_file', '').value
        self.fixed_depth = float(self.declare_parameter('fixed_depth', 0.0).value)
        self.camera_frame = self.declare_parameter('camera_frame', 'camera_frame').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.segmentation_conf_threshold = self.declare_parameter('segmentation_conf_threshold', 0.7).value
        
        self.load_calibration()
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        
        self.image_sub = self.create_subscription(
            Image, '/camera/raw', self.image_callback, 10)
        
        self.image_pub = self.create_publisher(Image, '/camera/object', 10)
        
        self.service = self.create_service(
            DetectObject3D, 'detect_object_3d', self.detect_callback)
        
        model_path = self.declare_parameter('model_path', 'yolov8n.pt').value
        self.model = YOLO(model_path)
        self.get_logger().info(f"YOLO model loaded: {model_path}")
        
        self.get_logger().info(f"Segmentation threshold: {self.segmentation_conf_threshold}")
        
        if self.camera_matrix is not None:
            self.get_logger().info(
                f"Camera calibration loaded (fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f})"
            )
        else:
            self.get_logger().warn("No camera calibration loaded; 3D conversions will be limited.")
        
        if self.fixed_depth <= 0.0:
            self.get_logger().warn(
                "Parameter 'fixed_depth' is <= 0.0. 3D coordinates will stay at z=0 unless a valid depth is provided."
            )
        
        self.get_logger().info("Object Detection Node ready (image coordinates mode with classical segmentation)")
    
    def load_calibration(self):
        if not self.calibration_file:
            return
        
        if not os.path.exists(self.calibration_file):
            self.get_logger().warn(f"Calibration file not found: {self.calibration_file}")
            return
        
        try:
            with open(self.calibration_file, 'r', encoding='utf-8') as f:
                calibration_data = yaml.safe_load(f)
            
            camera_matrix_data = calibration_data.get('camera_matrix')
            if camera_matrix_data is None:
                self.get_logger().warn(f"No 'camera_matrix' entry in calibration file: {self.calibration_file}")
                return
            
            if isinstance(camera_matrix_data, dict) and 'data' in camera_matrix_data:
                rows = camera_matrix_data.get('rows', 3)
                cols = camera_matrix_data.get('cols', 3)
                self.camera_matrix = np.array(camera_matrix_data['data']).reshape(rows, cols)
            else:
                self.camera_matrix = np.array(camera_matrix_data)
            
            if self.camera_matrix.shape != (3, 3):
                self.get_logger().warn(f"Camera matrix has unexpected shape: {self.camera_matrix.shape}")
                self.camera_matrix = None
                return
            
            self.fx = float(self.camera_matrix[0, 0])
            self.fy = float(self.camera_matrix[1, 1])
            self.cx = float(self.camera_matrix[0, 2])
            self.cy = float(self.camera_matrix[1, 2])
            
            if 'image_width' in calibration_data:
                self.image_width = int(calibration_data['image_width'])
            if 'image_height' in calibration_data:
                self.image_height = int(calibration_data['image_height'])
        
        except Exception as exc:
            self.get_logger().error(f"Failed to load calibration file '{self.calibration_file}': {exc}")
            self.camera_matrix = None
    
    def convert_pixel_to_camera_point(self, u, v, depth=None):
        if depth is None or depth <= 0.0:
            depth = self.fixed_depth
        
        if depth is None or depth <= 0.0:
            return 0.0, 0.0, 0.0
        
        if self.camera_matrix is None:
            if not self.warn_missing_calibration_logged:
                self.get_logger().warn(
                    "Camera calibration not available; returning zero-centered coordinates."
                )
                self.warn_missing_calibration_logged = True
            return 0.0, 0.0, float(depth)
        
        if self.fx == 0.0 or self.fy == 0.0:
            self.get_logger().warn("Invalid focal lengths (fx or fy is zero).")
            return 0.0, 0.0, float(depth)
        
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        
        return float(x), float(y), float(depth)
    
    def transform_camera_to_base(self, x, y, z):
        if self.base_frame == self.camera_frame:
            return float(x), float(y), float(z)
        
        point_stamped = PointStamped()
        point_stamped.header.frame_id = self.camera_frame
        point_stamped.header.stamp = self.get_clock().now().to_msg()
        point_stamped.point.x = float(x)
        point_stamped.point.y = float(y)
        point_stamped.point.z = float(z)
        
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                Time()
            )
            transformed = do_transform_point(point_stamped, transform)
            return (
                float(transformed.point.x),
                float(transformed.point.y),
                float(transformed.point.z),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            self.get_logger().warn(
                f"Unable to transform point from {self.camera_frame} to {self.base_frame}: {exc}"
            )
            return None
    
    def calculate_mask_centroid(self, mask):
        """
        Calculate the centroid (center of mass) of a binary mask.
        Returns (center_x, center_y) in pixel coordinates.
        """
        if mask is None:
            return None, None
        
        try:
            # Convert mask to numpy array if needed
            if hasattr(mask, 'cpu'):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)
            
            # Resize mask to image dimensions if needed
            if len(mask_np.shape) == 2:
                # Already 2D mask
                pass
            elif len(mask_np.shape) == 3:
                # Take first channel if 3D
                mask_np = mask_np[:, :, 0]
            else:
                return None, None
            
            # Ensure binary mask
            if mask_np.dtype != bool:
                mask_np = mask_np > 0.5
            
            # Check if mask has any pixels
            if mask_np.sum() == 0:
                return None, None
            
            # Calculate moments to find centroid
            moments = cv2.moments(mask_np.astype(np.uint8))
            
            if moments['m00'] == 0:
                return None, None
            
            center_x = moments['m10'] / moments['m00']
            center_y = moments['m01'] / moments['m00']
            
            return float(center_x), float(center_y)
        except Exception as e:
            self.get_logger().warn(f"Error calculating mask centroid: {e}")
            return None, None
    
    def segment_object_in_bbox(self, image, bbox):
        """
        Segment l'objet dans la bounding box en utilisant des techniques classiques.
        Retourne le masque binaire de l'objet segmenté.
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # S'assurer que les coordonnées sont dans les limites de l'image
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Extraire la région d'intérêt (ROI)
        roi = image[y1:y2, x1:x2].copy()
        
        if roi.size == 0:
            return None
        
        # Méthode 1: GrabCut (bon pour objets avec contraste)
        try:
            mask = np.zeros(roi.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            
            # Rectangle initial (presque toute la ROI, avec marge)
            rect = (5, 5, max(5, roi.shape[1]-10), max(5, roi.shape[0]-10))
            cv2.grabCut(roi, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Si GrabCut a trouvé quelque chose, l'utiliser
            if mask2.sum() > 0:
                # Créer le masque complet de l'image
                full_mask = np.zeros((h, w), dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = mask2
                return full_mask
        except Exception as e:
            self.get_logger().debug(f"GrabCut failed: {e}")
        
        # Méthode 2: Segmentation par couleur (si l'objet a une couleur distinctive)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculer l'histogramme de couleur dans le centre de la ROI (probablement l'objet)
        center_h, center_w = roi.shape[0]//2, roi.shape[1]//2
        center_region = hsv_roi[max(0, center_h-10):min(roi.shape[0], center_h+10), 
                                max(0, center_w-10):min(roi.shape[1], center_w+10)]
        
        if center_region.size > 0:
            # Calculer la couleur moyenne dans le centre
            mean_hue = np.mean(center_region[:, :, 0])
            mean_sat = np.mean(center_region[:, :, 1])
            mean_val = np.mean(center_region[:, :, 2])
            
            # Créer un masque basé sur la similarité de couleur
            lower_bound = np.array([max(0, mean_hue - 20), max(0, mean_sat - 40), max(0, mean_val - 40)])
            upper_bound = np.array([min(179, mean_hue + 20), min(255, mean_sat + 40), min(255, mean_val + 40)])
            
            mask_hsv = cv2.inRange(hsv_roi, lower_bound, upper_bound)
            
            # Nettoyer le masque avec morphologie
            kernel = np.ones((5, 5), np.uint8)
            mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)
            mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel)
            
            if mask_hsv.sum() > 0:
                full_mask = np.zeros((h, w), dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = mask_hsv
                return full_mask
        
        # Méthode 3: Threshold adaptatif sur la valeur (luminosité)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(
            gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Filtrer les petits blobs
        kernel = np.ones((3, 3), np.uint8)
        adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        # Prendre le plus grand contour
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:  # Seuil minimum
                mask_contour = np.zeros(roi.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask_contour, [largest_contour], -1, 255, -1)
                
                full_mask = np.zeros((h, w), dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = mask_contour
                return full_mask
        
        return None
    
    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.image_width is None:
                self.image_height, self.image_width = self.latest_image.shape[:2]
                self.get_logger().info(f"[DETECTION] Image reçue: {self.image_width}x{self.image_height} pixels")
            
            # Faire la détection YOLO
            conf_threshold = 0.25
            results = self.model(self.latest_image, conf=conf_threshold, verbose=False)
            
            # Annoter l'image avec uniquement les détections avec confiance > 0.4
            annotated_image = self.latest_image.copy()
            detections_count = 0
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                for i, box in enumerate(results[0].boxes):
                    conf = float(box.conf[0])
                    if conf > 0.4:
                        detections_count += 1
                        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                        cls = int(box.cls[0])
                        class_name = self.model.names[cls]
                        
                        # Dessiner seulement le rectangle (pas de segmentation visuelle pour réduire le lag)
                        cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # Ajouter le label avec la confiance
                        label = f"{class_name} {conf:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_image, (int(x1), int(y1) - label_size[1] - 10), 
                                     (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                        cv2.putText(annotated_image, label, (int(x1), int(y1) - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                if detections_count > 0:
                    self.get_logger().debug(f"[DETECTION] {detections_count} objet(s) détecté(s) dans l'image (conf > 0.4)")
            
            # Publier l'image annotée
            try:
                img_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                img_msg.header = msg.header
                self.image_pub.publish(img_msg)
            except Exception as e:
                self.get_logger().error(f"Error publishing annotated image: {e}")
                
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
    
    def detect_callback(self, request, response):
        if self.latest_image is None:
            response.success = False
            response.message = "No camera image available"
            self.get_logger().warn("[DETECTION] Service appelé mais aucune image disponible")
            return response
        
        object_class_requested = request.object_class if request.object_class else "tous objets"
        self.get_logger().info(f"[DETECTION] Requête de détection reçue pour: {object_class_requested}")
        
        try:
            # Seuil de confiance minimum
            conf_threshold = 0.25
            self.get_logger().debug(f"[DETECTION] Exécution détection YOLO (seuil: {conf_threshold})")
            results = self.model(self.latest_image, conf=conf_threshold, verbose=False)
            
            if len(results) == 0 or len(results[0].boxes) == 0:
                response.success = False
                response.message = "No objects detected"
                self.get_logger().info(f"[DETECTION] Aucun objet détecté (seuil: {conf_threshold})")
                return response
            
            best_detection = None
            best_confidence = 0.0
            total_detections = len(results[0].boxes)
            self.get_logger().info(f"[DETECTION] {total_detections} détection(s) trouvée(s) par YOLO")
            
            for i, box in enumerate(results[0].boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls]
                
                # Log toutes les détections
                self.get_logger().info(f"[DETECTION] Détection {i+1}/{total_detections}: {class_name} (confiance: {conf:.2f})")
                
                if request.object_class and request.object_class.lower() != class_name.lower():
                    self.get_logger().debug(f"[DETECTION] {class_name} ignoré (recherche: {request.object_class})")
                    continue
                
                if conf > best_confidence:
                    best_confidence = conf
                    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                    best_detection = {
                        'bbox': [x1, y1, x2, y2],
                        'class': class_name,
                        'confidence': conf
                    }
                    self.get_logger().info(f"[DETECTION] Meilleure détection mise à jour: {class_name} (conf: {conf:.2f})")
            
            if best_detection is None:
                response.success = False
                response.message = f"Object class '{request.object_class}' not found" if request.object_class else "No objects detected"
                return response
            
            x1, y1, x2, y2 = best_detection['bbox']
            
            # Faire la segmentation seulement si la confiance est suffisante
            segmentation_mask = None
            if best_detection['confidence'] >= self.segmentation_conf_threshold:
                self.get_logger().info(f"[DETECTION] Confiance suffisante ({best_detection['confidence']:.2f} >= {self.segmentation_conf_threshold}), tentative de segmentation")
                segmentation_mask = self.segment_object_in_bbox(self.latest_image, [x1, y1, x2, y2])
                if segmentation_mask is not None:
                    mask_pixels = segmentation_mask.sum()
                    self.get_logger().info(f"[DETECTION] Segmentation réussie: {mask_pixels} pixels dans le masque")
                else:
                    self.get_logger().warn(f"[DETECTION] Segmentation échouée, utilisation du centre de la bbox")
            else:
                self.get_logger().debug(f"[DETECTION] Segmentation ignorée (conf {best_detection['confidence']:.2f} < {self.segmentation_conf_threshold})")
            
            # Calculer le centre : utiliser le centroïde du masque de segmentation si disponible
            if segmentation_mask is not None:
                center_x, center_y = self.calculate_mask_centroid(segmentation_mask)
                if center_x is None or center_y is None:
                    # Fallback vers le centre de la bbox
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    self.get_logger().warn("[DETECTION] Calcul centroïde segmentation échoué, utilisation centre bbox")
                else:
                    self.get_logger().info(f"[DETECTION] Coordonnées depuis segmentation: ({center_x:.1f}, {center_y:.1f}) pixels")
            else:
                # Pas de segmentation disponible, utiliser le centre de la bbox
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                if best_detection['confidence'] < self.segmentation_conf_threshold:
                    self.get_logger().debug(f"[DETECTION] Pas de segmentation (conf {best_detection['confidence']:.2f} < {self.segmentation_conf_threshold}), centre bbox: ({center_x:.1f}, {center_y:.1f})")
                else:
                    self.get_logger().debug(f"[DETECTION] Masque segmentation non disponible, centre bbox: ({center_x:.1f}, {center_y:.1f})")
            
            depth_to_use = request.known_object_size if request.known_object_size > 0.0 else self.fixed_depth
            self.get_logger().info(f"[DETECTION] Conversion pixels -> 3D avec profondeur: {depth_to_use:.3f} m")
            camera_point = self.convert_pixel_to_camera_point(center_x, center_y, depth_to_use)
            self.get_logger().info(f"[DETECTION] Position caméra: X={camera_point[0]:.3f} Y={camera_point[1]:.3f} Z={camera_point[2]:.3f} m")
            base_point = self.transform_camera_to_base(*camera_point) if (camera_point[0] != 0.0 or camera_point[1] != 0.0 or camera_point[2] != 0.0) else None
            if base_point is not None:
                self.get_logger().info(f"[DETECTION] Position base: X={base_point[0]:.3f} Y={base_point[1]:.3f} Z={base_point[2]:.3f} m")
            else:
                self.get_logger().warn("[DETECTION] Transformation caméra->base échouée, utilisation position caméra")
            
            bbox_payload = {
                "bbox": [x1, y1, x2, y2],
                "center": [center_x, center_y],
                "uses_segmentation": segmentation_mask is not None,
                "depth_used": float(depth_to_use),
                "position_camera_m": [float(camera_point[0]), float(camera_point[1]), float(camera_point[2])]
            }
            
            if base_point is not None:
                bbox_payload["position_base_m"] = [
                    float(base_point[0]),
                    float(base_point[1]),
                    float(base_point[2])
                ]

            response.position_camera_frame = Point()
            response.position_camera_frame.x = float(camera_point[0])
            response.position_camera_frame.y = float(camera_point[1])
            response.position_camera_frame.z = float(camera_point[2])

            response.position_3d = Point()
            if base_point is not None:
                response.position_3d.x = float(base_point[0])
                response.position_3d.y = float(base_point[1])
                response.position_3d.z = float(base_point[2])
            else:
                response.position_3d.x = float(camera_point[0])
                response.position_3d.y = float(camera_point[1])
                response.position_3d.z = float(camera_point[2])
            
            response.success = True
            response.message = json.dumps(bbox_payload)
            response.detected_class = best_detection['class']
            response.confidence = best_detection['confidence']
            
            method = "segmentation" if segmentation_mask is not None else "bbox"
            self.get_logger().info(
                f"[DETECTION] Détection finale: {best_detection['class']} (conf: {best_detection['confidence']:.2f}) "
                f"bbox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] "
                f"centre_px: ({center_x:.1f}, {center_y:.1f}) "
                f"camera_3d: ({camera_point[0]:.3f}, {camera_point[1]:.3f}, {camera_point[2]:.3f}) m "
                f"(méthode: {method})"
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

