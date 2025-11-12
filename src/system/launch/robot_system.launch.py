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
    
    # Node 4: Object Detection
    object_detection_node = Node(
        package='ObjectDetectionNode',
        executable='object_detection_node',
        name='object_detection_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'calibration_file': os.path.expanduser('~/Mark3_ws/config/camera_calibration.yaml'),
            'model_path': os.path.expanduser('~/Mark3_ws/model/tools_detection.pt'),
            'camera_frame': 'camera_frame',
            'base_frame': 'base_link',
            'fixed_depth': 0.0
        }],
        remappings=[]
    )
    
    # Node 5: Interface
    interface_node = Node(
        package='Interface',
        executable='Interface',
        name='IHM',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'yolo_model': os.path.expanduser('~/Mark3_ws/model/tools_detection.pt')
        }],
        remappings=[]
    )

    audio_capture_node = Node(
        package='audio_capture_node',
        executable='audio_capture_node',
        name='audio_capture_node',
        output='screen',
        emulate_tty=True,
    )

    voice_command_node = Node(
        package='voice_command_node',
        executable='voice_command_node',
        name='voice_command_node',
        output='screen',
        emulate_tty=True,
    )
    
    return LaunchDescription([
        urdf_file_arg,
        camera__node,
        ik_server_node,
        state_publisher_node,
        urdf_reloader_node,
        object_detection_node,
        interface_node,
        audio_capture_node,
        voice_command_node,
    ])

