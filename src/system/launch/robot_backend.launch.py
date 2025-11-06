#!/usr/bin/env python3
"""
Launch file pour démarrer seulement le backend (sans interface graphique)
Utile pour les tests en ligne de commande ou avec des clients externes
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Déclaration des arguments
    urdf_file_arg = DeclareLaunchArgument(
        'urdf_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('IKsolverNode'),
            'urdf',
            'robot_dh.urdf'
        ]),
        description='Chemin vers le fichier URDF'
    )
    
    # Node 1: Serveur IK
    ik_server_node = Node(
        package='IKsolverNode',
        executable='IKsolverNode',
        name='ik_service_server',
        output='screen',
        emulate_tty=True,
    )
    
    # Node 2: State Publisher
    state_publisher_node = Node(
        package='state_publisher',
        executable='state_publisher',
        name='joint_publisher',
        output='screen',
        emulate_tty=True,
    )
    
    # Node 3: Robot State Publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        emulate_tty=True,
        arguments=[LaunchConfiguration('urdf_file')]
    )
    
    return LaunchDescription([
        urdf_file_arg,
        ik_server_node,
        state_publisher_node,
        robot_state_publisher_node,
    ])

