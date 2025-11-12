from setuptools import find_packages, setup

package_name = 'audio_capture_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools','rclpy','audio-common-msgs','std_msgs','pyaudio','numpy'],
    zip_safe=True,
    maintainer='aibot',
    maintainer_email='aibot@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'audio_capture_node = audio_capture_node.audio_capture_node:main',
        ],
    },
)
