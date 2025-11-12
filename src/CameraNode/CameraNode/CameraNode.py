#!/usr/bin/env python3
import os
import sys

venv_python = os.path.expanduser('~/venv/bin/python3')
if os.path.exists(venv_python) and sys.executable != venv_python:
    os.execv(venv_python, [venv_python] + sys.argv)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/raw', 10)
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.bridge = CvBridge()
        
        self.cap = None
        self.camera_index = 0
        self.consecutive_failures = 0
        self.last_warning_time = 0.0
        self.warning_interval = 5.0
        
        self.init_camera()

    def init_camera(self):
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.get_logger().error(f"Impossible d'ouvrir la caméra (index {self.camera_index})")
            self.get_logger().info("Vérifiez que la caméra est branchée et accessible")
            self.cap = None
        else:
            self.get_logger().info(f"Caméra ouverte (index {self.camera_index}), publication sur /camera/raw")
            self.consecutive_failures = 0

    def timer_callback(self):
        if self.cap is None or not self.cap.isOpened():
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time - self.last_warning_time > self.warning_interval:
                self.get_logger().warn("Caméra non disponible, tentative de reconnexion...")
                self.last_warning_time = current_time
                self.init_camera()
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.consecutive_failures += 1
            current_time = self.get_clock().now().nanoseconds / 1e9
            
            if self.consecutive_failures > 10 and current_time - self.last_warning_time > self.warning_interval:
                self.get_logger().warn(f"Caméra ne répond plus ({self.consecutive_failures} échecs), tentative de reconnexion...")
                self.last_warning_time = current_time
                self.init_camera()
            return
        
        self.consecutive_failures = 0
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(msg)

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
