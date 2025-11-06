#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import signal
import os


class URDFReloader(Node):
    
    def __init__(self):
        super().__init__('urdf_reloader')
        
        self.subscription = self.create_subscription(
            String,
            '/urdf_updated',
            self.urdf_updated_callback,
            10
        )
        
        self.robot_state_pub_process = None
        self.urdf_path = None
        
        self.get_logger().info("URDFReloader prêt. En attente de /urdf_updated...")
    
    def urdf_updated_callback(self, msg):
        new_urdf_path = msg.data
        
        self.get_logger().info(f'URDF mis à jour: {new_urdf_path}')
        
        if self.robot_state_pub_process is not None:
            try:
                self.get_logger().info('Arrêt de robot_state_publisher...')
                self.robot_state_pub_process.send_signal(signal.SIGTERM)
                self.robot_state_pub_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.robot_state_pub_process.kill()
            except Exception as e:
                self.get_logger().error(f'Erreur lors de l\'arrêt: {e}')
        
        try:
            self.get_logger().info('Redémarrage de robot_state_publisher...')
            self.robot_state_pub_process = subprocess.Popen([
                'ros2', 'run', 'robot_state_publisher', 'robot_state_publisher',
                new_urdf_path
            ])
            self.urdf_path = new_urdf_path
            self.get_logger().info('robot_state_publisher redémarré')
        except Exception as e:
            self.get_logger().error(f'Erreur lors du redémarrage: {e}')
    
    def shutdown(self):
        if self.robot_state_pub_process is not None:
            try:
                self.robot_state_pub_process.terminate()
                self.robot_state_pub_process.wait(timeout=2)
            except:
                pass


def main(args=None):
    rclpy.init(args=args)
    
    node = URDFReloader()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
