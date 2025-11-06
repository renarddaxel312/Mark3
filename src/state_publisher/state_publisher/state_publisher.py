#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, String, Header
import numpy as np
import json
import time


class JointPublisher(Node):
    
    def __init__(self):
        super().__init__('joint_publisher')
        
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/pos/angles',
            self.angles_callback,
            10
        )
        
        self.config_subscription = self.create_subscription(
            String,
            '/config/axis',
            self.config_callback,
            10
        )
        
        self.q = None
        self.q_target = None
        self.q_rad = np.array([])
        self.joint_names = []
        self.n_joints = 0
        self.is_interpolating = False
        
        self.get_logger().info("JointPublisher prêt.")
        self.get_logger().info("En attente de /config/axis et /pos/angles...")
    
    def config_callback(self, msg):
        try:
            config_data = json.loads(msg.data)
            
            if isinstance(config_data, list):
                joint_types = config_data
            elif 'axes' in config_data:
                joint_types = config_data['axes']
            else:
                return
            
            self.n_joints = len(joint_types)
            self.joint_names = [f'joint_{i}' for i in range(self.n_joints)]
            
            self.get_logger().info(f'Configuration mise à jour: {self.n_joints} joints')
            self.get_logger().info(f'Noms des joints: {self.joint_names}')
            
            self.q = np.zeros(self.n_joints)
            self.q_rad = np.zeros(self.n_joints)
            
            self.get_logger().info('Angles réinitialisés à zéro')
            
            self.publish_zero_position()
            
        except Exception as e:
            self.get_logger().error(f'Erreur config: {e}')
    
    def publish_zero_position(self):
        if self.n_joints == 0 or not self.joint_names:
            return
        
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [0.0] * self.n_joints
        
        self.publisher_.publish(msg)
        self.get_logger().info(f'Position zéro publiée: {self.n_joints} joints à 0.0 rad')
    
    def angles_callback(self, msg):
        try:
            q_target_new = np.array(msg.data)
            
            if not self.joint_names or len(self.joint_names) != len(q_target_new):
                self.n_joints = len(q_target_new)
                self.joint_names = [f'joint_{i}' for i in range(self.n_joints)]
                self.get_logger().info(f'Auto-config: {self.n_joints} joints détectés')
                self.get_logger().info(f'Noms des joints: {self.joint_names}')
            
            if self.q is None:
                self.q = q_target_new.copy()
                self.q_target = q_target_new.copy()
                self.q_rad = np.deg2rad(self.q)
                self.get_logger().info(f"Position initiale: {np.round(self.q, 2).tolist()} deg")
                return
            
            q_start = self.q.copy()
            self.q_target = q_target_new.copy()
            
            n_steps = 180
            
            self.get_logger().info(
                f"Nouvelle cible reçue: {np.round(self.q_target, 2).tolist()} deg"
            )
            self.get_logger().info(f"   Interpolation en {n_steps} étapes...")
            
            for step in range(n_steps + 1):
                alpha = step / n_steps
                self.q = q_start + alpha * (self.q_target - q_start)
                self.q_rad = np.deg2rad(self.q)
                
                msg = JointState()
                msg.header = Header()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = self.joint_names
                msg.position = self.q_rad.tolist()
                
                self.publisher_.publish(msg)
                
                time.sleep(0.02)
            
            self.get_logger().info(f"Position finale atteinte: {np.round(self.q, 2).tolist()} deg")
            
        except Exception as e:
            self.get_logger().error(f'Erreur dans angles_callback: {e}')
    
    def timer_callback(self):
        if self.q is None or len(self.q_rad) == 0:
            return
        
        if len(self.joint_names) != len(self.q_rad):
            self.get_logger().warn(
                f'Mismatch: {len(self.joint_names)} noms vs {len(self.q_rad)} angles'
            )
            return
        
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.q_rad.tolist()

        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = JointPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
