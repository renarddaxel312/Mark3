#!/usr/bin/env python3
"""
Nœud ROS2 pour résoudre l'IK avec position ET orientation de l'effecteur.
Version étendue du IKsolverNode avec support de l'orientation.
"""

import os
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Header
from IKsolverNode.dh_utils import urdf, robot_dh
from IKsolverNode.kinematics import (
    is_reachable,
    inverse_kinematics_urdf,
    forward_kinematics_urdf,
    parse_urdf,
    matrix_to_rpy,
    xyz_rpy_to_matrix
)
from ament_index_python.packages import get_package_share_directory


URDF_PATH = os.path.expanduser(
    os.path.join(get_package_share_directory("IKsolverNode"), "urdf/robot_dh.urdf")
)


class IKSolverWithOrientation(Node):
    """
    Nœud ROS2 qui résout l'IK avec position et orientation.
    
    Subscribes:
        /target_pose (geometry_msgs/PoseStamped): Pose cible (position + orientation)
    
    Publishes:
        /joint_states (sensor_msgs/JointState): États des joints calculés
        /current_pose (geometry_msgs/PoseStamped): Pose actuelle de l'effecteur
    """
    
    def __init__(self, urdf_path, joint_names, use_orientation=True):
        super().__init__('ik_solver_with_orientation')
        
        self.urdf_path = urdf_path
        self.joint_names = joint_names
        self.use_orientation = use_orientation
        
        # Parser l'URDF
        self.urdf_info = parse_urdf(urdf_path)
        self.n_joints = self.urdf_info['n_joints']
        
        # Configuration actuelle
        self.current_q = np.zeros(self.n_joints)
        
        # Publishers
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.pose_pub = self.create_publisher(PoseStamped, 'current_pose', 10)
        
        # Subscriber
        self.target_sub = self.create_subscription(
            PoseStamped,
            'target_pose',
            self.target_pose_callback,
            10
        )
        
        # Timer pour publier l'état actuel
        self.timer = self.create_timer(0.1, self.publish_current_state)
        
        mode = "position + orientation" if use_orientation else "position seulement"
        self.get_logger().info(f"IK Solver initialisé (mode: {mode})")
        self.get_logger().info(f"Nombre de joints: {self.n_joints}")
        self.get_logger().info("En attente de pose cible sur /target_pose...")
    
    def target_pose_callback(self, msg):
        """Callback pour résoudre l'IK à partir d'une pose cible."""
        # Extraire la position
        target_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        
        # Extraire l'orientation (quaternion -> RPY)
        target_rpy = None
        if self.use_orientation:
            q = msg.pose.orientation
            target_rpy = self.quaternion_to_rpy(q.x, q.y, q.z, q.w)
            self.get_logger().info(
                f"Résolution IK - Pos: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] "
                f"Orient: [{np.rad2deg(target_rpy[0]):.1f}°, {np.rad2deg(target_rpy[1]):.1f}°, {np.rad2deg(target_rpy[2]):.1f}°]"
            )
        else:
            self.get_logger().info(
                f"Résolution IK - Pos: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]"
            )
        
        # Résoudre l'IK
        try:
            q_solution = inverse_kinematics_urdf(
                self.urdf_path,
                target_pos=target_pos,
                target_rpy=target_rpy,
                q_init=self.current_q,
                max_iter=2000,
                lr=0.3,
                orientation_weight=0.15 if self.use_orientation else 0.0
            )
            
            self.current_q = q_solution
            self.get_logger().info(f"IK réussie - Angles: {np.round(q_solution, 2)}")
            
            # Vérifier le résultat
            T = forward_kinematics_urdf(self.urdf_info, q_solution, return_full_pose=True)
            pos_achieved = T[:3, 3]
            error_pos = np.linalg.norm(target_pos - pos_achieved)
            
            if self.use_orientation:
                rpy_achieved = matrix_to_rpy(T[:3, :3])
                error_orient = np.linalg.norm(target_rpy - rpy_achieved)
                self.get_logger().info(
                    f"Erreur - Pos: {error_pos*1000:.2f}mm, Orient: {np.rad2deg(error_orient):.2f}°"
                )
            else:
                self.get_logger().info(f"Erreur de position: {error_pos*1000:.2f}mm")
                
        except Exception as e:
            self.get_logger().error(f"Erreur lors de la résolution IK: {e}")
    
    def publish_current_state(self):
        """Publie l'état actuel des joints et la pose de l'effecteur."""
        # Publier les joint states
        joint_msg = JointState()
        joint_msg.header = Header()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = self.joint_names
        joint_msg.position = np.deg2rad(self.current_q).tolist()
        self.joint_pub.publish(joint_msg)
        
        # Calculer et publier la pose actuelle
        T = forward_kinematics_urdf(self.urdf_info, self.current_q, return_full_pose=True)
        
        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "base_link"
        
        # Position
        pose_msg.pose.position.x = T[0, 3]
        pose_msg.pose.position.y = T[1, 3]
        pose_msg.pose.position.z = T[2, 3]
        
        # Orientation (RPY -> quaternion)
        rpy = matrix_to_rpy(T[:3, :3])
        qx, qy, qz, qw = self.rpy_to_quaternion(*rpy)
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        
        self.pose_pub.publish(pose_msg)
    
    @staticmethod
    def quaternion_to_rpy(x, y, z, w):
        """Convertit un quaternion en angles roll, pitch, yaw."""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    @staticmethod
    def rpy_to_quaternion(roll, pitch, yaw):
        """Convertit des angles roll, pitch, yaw en quaternion."""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return qx, qy, qz, qw


def main():
    """Point d'entrée principal."""
    # Configuration du robot
    joint_types = ['rot360', 'rot180', 'rot180', 'rot180', 'rot180']
    dh_config = robot_dh(joint_types, 'gripper')
    urdf_config = joint_types + ['gripper']
    
    # Générer l'URDF
    os.makedirs(os.path.dirname(URDF_PATH), exist_ok=True)
    urdf_str = urdf(urdf_config, name="modular_robot")
    with open(URDF_PATH, "w") as f:
        f.write(urdf_str)
    
    # Parser pour obtenir les noms de joints
    urdf_info = parse_urdf(URDF_PATH)
    n_joints = urdf_info['n_joints']
    joint_names = [f"joint_{i}" for i in range(n_joints)]
    
    # Initialiser ROS2
    rclpy.init()
    
    # Créer le nœud (use_orientation=True pour activer le mode avec orientation)
    node = IKSolverWithOrientation(
        urdf_path=URDF_PATH,
        joint_names=joint_names,
        use_orientation=True  # Mettre False pour mode position seulement
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

