from setuptools import find_packages, setup

package_name = 'ObjectDetectionNode'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Axel Renard',
    maintainer_email='renardd.axel@gmail.com',
    description='Object detection node with YOLO and 3D estimation using calibrated camera',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detection_node = ObjectDetectionNode.ObjectDetectionNode:main',
        ],
    },
    python_requires='>=3.8',
)

