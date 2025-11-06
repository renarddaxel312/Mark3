#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ik_service_interface.srv import SolveIK
from geometry_msgs.msg import Point, Vector3
import numpy as np


class IKClient(Node):
    def __init__(self):
        super().__init__('ik_client')
        self.client = self.create_client(SolveIK, 'solve_ik')
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('En attente du service...')
    
    def send_request(self, target_pos, target_rpy=None):
        request = SolveIK.Request()
        
        request.target_position = Point()
        request.target_position.x = float(target_pos[0])
        request.target_position.y = float(target_pos[1])
        request.target_position.z = float(target_pos[2])
        
        if target_rpy is not None:
            request.use_orientation = True
            request.target_orientation = Vector3()
            request.target_orientation.x = float(target_rpy[0])
            request.target_orientation.y = float(target_rpy[1])
            request.target_orientation.z = float(target_rpy[2])
            self.get_logger().info(f'Requête IK: pos={target_pos}, rpy={target_rpy}')
        else:
            request.use_orientation = False
            self.get_logger().info(f'Requête IK: pos={target_pos} (position seulement)')
        
        future = self.client.call_async(request)
        return future


def main():
    rclpy.init()
    
    client = IKClient()
    
    print("\n=== Test 1: Position seulement ===")
    target_pos = [0.2, 0.2, 0.3]
    future = client.send_request(target_pos)
    
    rclpy.spin_until_future_complete(client, future)
    
    if future.result() is not None:
        response = future.result()
        if response.success:
            print(f"Succès: {response.message}")
            print(f"Angles (degrés): {np.round(response.joint_angles, 2).tolist()}")
        else:
            print(f"Échec: {response.message}")
    
    print("\n=== Test 2: Position + Orientation ===")
    target_pos = [0.25, 0.1, 0.2]
    target_rpy = [0.0, -np.pi/2, 0.0]
    future = client.send_request(target_pos, target_rpy)
    
    rclpy.spin_until_future_complete(client, future)
    
    if future.result() is not None:
        response = future.result()
        if response.success:
            print(f"Succès: {response.message}")
            print(f"Angles (degrés): {np.round(response.joint_angles, 2).tolist()}")
        else:
            print(f"Échec: {response.message}")
    
    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
