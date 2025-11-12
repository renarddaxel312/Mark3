#!/usr/bin/env python3
import os
import sys

venv_python = os.path.expanduser('~/venv/bin/python3')
if os.path.exists(venv_python) and sys.executable != venv_python:
    os.execv(venv_python, [venv_python] + sys.argv)
import numpy as np
import rclpy
from rclpy.node import Node
from ik_service_interface.srv import SolveIK
from std_msgs.msg import String, Float32MultiArray
from IKsolverNode.dh_utils import urdf, robot_dh
from IKsolverNode.kinematics import inverse_kinematics_urdf, parse_urdf
from ament_index_python.packages import get_package_share_directory
import json
import time


URDF_PATH = os.path.expanduser(os.path.join(get_package_share_directory("IKsolverNode"), "urdf/robot_dh.urdf"))

class IKServiceServer(Node):
    def __init__(self, initial_joint_types=None):
        super().__init__('ik_service_server')
        
        if initial_joint_types is None:
            initial_joint_types = ['rot360', 'rot180', 'rot180', 'rot360', 'rot180', 'rot360']
        
        self.joint_types = initial_joint_types
        self.config_ready = False
        self.q_init = None
        
        self.urdf_updated_pub = self.create_publisher(String, '/urdf_updated', 10)
        
        self.config_sub = self.create_subscription(
            String,
            '/config/axis',
            self.config_callback,
            10
        )

        self.q_sub = self.create_subscription(
            Float32MultiArray,
            '/pos/angles',
            self.q_callback,
            10
        )
        
        self.srv = self.create_service(SolveIK, 'solve_ik', self.solve_ik_callback)
        
        self.get_logger().info('Serveur IK démarré.')
        self.get_logger().info('En attente de la configuration sur /config/axis...')
        
        self.update_robot_config(initial_joint_types)
    
    def config_callback(self, msg):
        try:
            config_data = json.loads(msg.data)
            
            if 'axes' in config_data:
                new_joint_types = config_data['axes']
            elif isinstance(config_data, list):
                new_joint_types = config_data
            else:
                self.get_logger().warn(f'Format de config non reconnu: {msg.data}')
                return
            
            if new_joint_types != self.joint_types:
                self.get_logger().info(f'Nouvelle configuration reçue: {new_joint_types}')
                self.update_robot_config(new_joint_types)
            
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Erreur de parsing JSON: {e}')
        except Exception as e:
            self.get_logger().error(f'Erreur lors de la mise à jour de config: {e}')

    def q_callback(self, msg):
        try:
            self.q_init = np.array(msg.data)
        except Exception as e:
            self.get_logger().error(f'Erreur lors de la mise à jour de la position initiale: {e}')
    

    def update_robot_config(self, joint_types):
        try:
            self.joint_types = joint_types
            self.dh_config = robot_dh(joint_types, 'gripper')
            urdf_config = joint_types + ['gripper']
            
            os.makedirs(os.path.dirname(URDF_PATH), exist_ok=True)
            urdf_str = urdf(urdf_config, name="modular_robot")
            with open(URDF_PATH, "w") as f:
                f.write(urdf_str)
            
            self.urdf_info = parse_urdf(URDF_PATH)
            self.config_ready = True
            
            self.get_logger().info(f'Configuration mise à jour: {joint_types}')
            self.get_logger().info(f'   Nombre de joints: {self.urdf_info["n_joints"]}')
            
            time.sleep(0.1)
            msg = String()
            msg.data = URDF_PATH
            self.urdf_updated_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'Erreur lors de la mise à jour: {e}')
            self.config_ready = False
    
    def solve_ik_callback(self, request, response):
        if not self.config_ready:
            response.joint_angles = []
            response.success = False
            response.message = "Configuration robot non prête"
            self.get_logger().warn('Requête IK reçue mais config non prête')
            return response
        
        try:
            target_pos = np.array([
                request.target_position.x,
                request.target_position.y,
                request.target_position.z
            ])
            
            self.get_logger().info(f'Requête IK reçue: pos={target_pos}')
            
            if request.use_orientation:
                target_rpy = np.array([
                    request.target_orientation.x,
                    request.target_orientation.y,
                    request.target_orientation.z
                ])
                self.get_logger().info(f'  avec orientation (RPY): {target_rpy}')
            else:
                target_rpy = None
                
            q_solution = inverse_kinematics_urdf(
                URDF_PATH,
                target_pos=target_pos,
                target_rpy=target_rpy,
                q_init=self.q_init,
                max_iter=2000,
                lr=0.3
            )
            
            response.joint_angles = q_solution.tolist()
            response.success = True
            response.message = f"IK résolue avec succès. {len(q_solution)} angles calculés."
            
            self.get_logger().info(f'Solution trouvée: {np.round(q_solution, 2).tolist()}')
            
        except Exception as e:
            response.joint_angles = []
            response.success = False
            response.message = f"Erreur lors du calcul IK: {str(e)}"
            self.get_logger().error(f'Erreur IK: {str(e)}')
        
        return response


def main():
    rclpy.init()
    
    server = IKServiceServer()
    
    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        pass
    
    server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
