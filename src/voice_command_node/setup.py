from setuptools import find_packages, setup

package_name = 'voice_command_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools',
                'rclpy',
                'std_msgs',
                'ik_service_interface',
                'object_detection_interface',
                'geometry_msgs',
                'audio_common_msgs',
                'speech_recognition',
            ],
    zip_safe=True,
    maintainer='axel',
    maintainer_email='renardd.axel@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'voice_command_node = voice_command_node.voice_command_node:main',
        ],
    },
)
