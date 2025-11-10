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
            'model_path': os.path.expanduser('~/Mark3_ws/yolo_training/tools_detector2/weights/best.pt'),
            'camera_frame': 'camera_frame',
            'base_frame': 'base_link',
            'fixed_depth': 0.5,
            'use_midas': False,  # Désactivé - utilise estimation basée sur taille bbox
            'use_bbox_size': True,  # Estimation de profondeur basée sur la taille du bounding box
            'known_object_size': 0.20,  # Taille réelle du tournevis en mètres (20 cm)
            'midas_depth_scale': 0.5,
            'capture_confidence_threshold': 0.6,
            'stable_frames_required': 3
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
            'yolo_model': os.path.expanduser('~/Mark3_ws/yolo_training/tools_detector2/weights/best.pt')
        }],
        remappings=[]
    )
    
    return LaunchDescription([
        urdf_file_arg,
        camera__node,
        ik_server_node,
        state_publisher_node,
        urdf_reloader_node,
        object_detection_node,
        interface_node,
    ])

