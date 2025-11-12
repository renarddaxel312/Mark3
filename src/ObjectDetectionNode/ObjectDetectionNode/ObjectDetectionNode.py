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

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import Image
from object_detection_interface.srv import DetectObject3D
from ultralytics import YOLO


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_width = None
        self.image_height = None
        
        self.image_sub = self.create_subscription(
            Image, '/camera/raw', self.image_callback, 10)
        
        self.image_pub = self.create_publisher(Image, '/camera/object', 10)
        
        self.service = self.create_service(
            DetectObject3D, 'detect_object_3d', self.detect_callback)
        
        model_path = self.declare_parameter('model_path', 'yolov8n.pt').value
        self.model = YOLO(model_path)
        self.get_logger().info(f"YOLO model loaded: {model_path}")
        
        self.get_logger().info("Object Detection Node ready (image coordinates mode)")
    
    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.image_width is None:
                self.image_height, self.image_width = self.latest_image.shape[:2]
            
            # Faire la détection YOLO
            conf_threshold = 0.25
            results = self.model(self.latest_image, conf=conf_threshold, verbose=False)
            
            # Annoter l'image avec uniquement les détections avec confiance > 0.5
            annotated_image = self.latest_image.copy()
            if len(results) > 0 and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    if conf > 0.4:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                        cls = int(box.cls[0])
                        class_name = self.model.names[cls]
                        
                        # Dessiner le rectangle
                        cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # Ajouter le label avec la confiance
                        label = f"{class_name} {conf:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_image, (int(x1), int(y1) - label_size[1] - 10), 
                                     (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                        cv2.putText(annotated_image, label, (int(x1), int(y1) - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
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
            return response
        
        try:
            # Seuil de confiance minimum
            conf_threshold = 0.25
            results = self.model(self.latest_image, conf=conf_threshold, verbose=False)
            
            if len(results) == 0 or len(results[0].boxes) == 0:
                response.success = False
                response.message = "No objects detected"
                self.get_logger().debug("No detections with confidence >= " + str(conf_threshold))
                return response
            
            best_detection = None
            best_confidence = 0.0
            
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls]
                
                # Log toutes les détections pour debug
                self.get_logger().debug(f"Detection: {class_name} (conf: {conf:.2f})")
                
                if request.object_class and request.object_class.lower() != class_name.lower():
                    continue
                
                if conf > best_confidence:
                    best_confidence = conf
                    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                    best_detection = {
                        'bbox': [x1, y1, x2, y2],
                        'class': class_name,
                        'confidence': conf
                    }
            
            if best_detection is None:
                response.success = False
                response.message = f"Object class '{request.object_class}' not found" if request.object_class else "No objects detected"
                return response
            
            x1, y1, x2, y2 = best_detection['bbox']
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            bbox_payload = {
                "bbox": [x1, y1, x2, y2],
                "center": [center_x, center_y]
            }

            response.position_camera_frame = Point()
            response.position_camera_frame.x = float(center_x)
            response.position_camera_frame.y = float(center_y)
            response.position_camera_frame.z = 0.0

            response.position_3d = Point()
            response.position_3d.x = 0.0
            response.position_3d.y = 0.0
            response.position_3d.z = 0.0
            
            response.success = True
            response.message = json.dumps(bbox_payload)
            response.detected_class = best_detection['class']
            response.confidence = best_detection['confidence']
            
            self.get_logger().info(
                f"Detected {best_detection['class']} (conf: {best_detection['confidence']:.2f}) "
                f"bbox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] center: ({center_x:.1f}, {center_y:.1f})"
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

