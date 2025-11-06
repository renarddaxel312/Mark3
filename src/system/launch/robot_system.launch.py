#!/usr/bin/env python3
"""
Launch file pour démarrer tout le système robotique
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os


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
    
    camera__node = Node(
        package='CameraNode',
        executable='CameraNode',
        name='camera_subscriber',
        output='screen',
        emulate_tty=True,
    )

    # Node 1: Serveur IK
    ik_server_node = Node(
        package='IKsolverNode',
        executable='IKsolverNode',
        name='ik_service_server',
        output='screen',
        emulate_tty=True,
        parameters=[],
        remappings=[]
    )
    
    # Node 2: State Publisher
    state_publisher_node = Node(
        package='state_publisher',
        executable='state_publisher',
        name='joint_publisher',
        output='screen',
        emulate_tty=True,
        parameters=[],
        remappings=[]
    )
    
    # Node 3: URDF Reloader (gère robot_state_publisher dynamiquement)
    urdf_reloader_node = Node(
        package='state_publisher',
        executable='urdf_reloader',
        name='urdf_reloader',
        output='screen',
        emulate_tty=True,
        parameters=[],
        remappings=[]
    )
    
    # Node 4: Interface
    interface_node = Node(
        package='Interface',
        executable='Interface',
        name='IHM',
        output='screen',
        emulate_tty=True,
        parameters=[],
        remappings=[]
    )
    
    return LaunchDescription([
        urdf_file_arg,
        camera__node,
        ik_server_node,
        state_publisher_node,
        urdf_reloader_node,
        interface_node,
    ])

