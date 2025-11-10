#!/usr/bin/env python3
"""
Script de test pour diagnostiquer le service de détection 3D
"""
import sys
import os

# Forcer l'utilisation du venv Python si disponible
venv_python = os.path.expanduser('~/venv/bin/python3')
if os.path.exists(venv_python) and sys.executable != venv_python:
    try:
        import rclpy
    except ImportError:
        os.execv(venv_python, [venv_python] + sys.argv)

import rclpy
from rclpy.node import Node
from object_detection_interface.srv import DetectObject3D
import time

class TestDetectionClient(Node):
    def __init__(self):
        super().__init__('test_detection_client')
        self.client = self.create_client(DetectObject3D, 'detect_object_3d')
        
    def test_service(self):
        self.get_logger().info("Attente du service detect_object_3d...")
        if not self.client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Service detect_object_3d non disponible!")
            return False
        
        self.get_logger().info("Service disponible!")
        
        # Faire plusieurs appels de test
        for i in range(3):
            self.get_logger().info(f"\n--- Test {i+1}/3 ---")
            
            request = DetectObject3D.Request()
            request.object_class = 'screwdriver'
            request.known_object_size = 0.0
            
            self.get_logger().info(f"Envoi de la requête: object_class='{request.object_class}'")
            
            future = self.client.call_async(request)
            
            # Attendre la réponse
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            
            if future.done():
                try:
                    response = future.result()
                    self.get_logger().info(f"Réponse reçue:")
                    self.get_logger().info(f"  success: {response.success}")
                    self.get_logger().info(f"  message: {response.message}")
                    self.get_logger().info(f"  detected_class: {response.detected_class}")
                    self.get_logger().info(f"  confidence: {response.confidence}")
                    
                    if response.success:
                        self.get_logger().info(f"  Position 3D (base):")
                        self.get_logger().info(f"    X: {response.position_3d.x:.3f}m")
                        self.get_logger().info(f"    Y: {response.position_3d.y:.3f}m")
                        self.get_logger().info(f"    Z: {response.position_3d.z:.3f}m")
                        
                        self.get_logger().info(f"  Position caméra:")
                        self.get_logger().info(f"    X: {response.position_camera_frame.x:.3f}m")
                        self.get_logger().info(f"    Y: {response.position_camera_frame.y:.3f}m")
                        self.get_logger().info(f"    Z: {response.position_camera_frame.z:.3f}m")
                    else:
                        self.get_logger().warn(f"  Échec: {response.message}")
                except Exception as e:
                    self.get_logger().error(f"Erreur lors de la réception de la réponse: {e}")
            else:
                self.get_logger().error("Timeout: pas de réponse après 10 secondes")
            
            time.sleep(1.0)
        
        return True

def main(args=None):
    rclpy.init(args=args)
    
    node = TestDetectionClient()
    
    try:
        success = node.test_service()
        if success:
            node.get_logger().info("\n✓ Tests terminés")
        else:
            node.get_logger().error("\n✗ Tests échoués")
    except KeyboardInterrupt:
        node.get_logger().info("\nTest interrompu")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

