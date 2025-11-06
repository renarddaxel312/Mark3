from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'IKsolverNode'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'urdf/meshes'), glob('urdf/meshes/*.stl')),
    ],
    install_requires=['rclpy',
                'sensor_msgs',
                'os',
                'numpy',
                'std_msgs',
                'dh_utils',
                'kinematics',
                ],
    zip_safe=True,
    maintainer='axel',
    maintainer_email='axel@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'IKsolverNode = IKsolverNode.IKsolverNode:main',
            'ik_client = IKsolverNode.ik_client:main',
            'IKsolverNode_animated = IKsolverNode.IKsolverNode_animated:main'
        ],
    },
)
