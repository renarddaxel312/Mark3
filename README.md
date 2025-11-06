# Mark3

**Dynamically reconfigurable modular robot platform with real-time inverse kinematics and 3D visualization**

Mark3 is a ROS2-based modular robotic arm system that allows you to dynamically configure your robot's structure and control it in real-time through an intuitive graphical interface. Build custom robotic arms by combining different joint types (360° rotation, 180° rotation, translation) and control them with real-time inverse kinematics solving.

## Features

- **Dynamic Robot Configuration**: Configure your robot structure on-the-fly by selecting joint types through a GUI
- **Real-time Inverse Kinematics**: Solve IK problems in real-time using Jacobian-based optimization
- **3D Visualization**: Visualize your robot with STL meshes in real-time using VTK
- **ROS2 Integration**: Fully integrated with ROS2 services, topics, and TF
- **Smooth Interpolation**: Smooth joint motion interpolation for fluid robot movement
- **Camera Integration**: Built-in camera support for visual feedback
- **URDF Generation**: Automatic URDF generation based on your robot configuration

## Architecture

The system consists of several ROS2 nodes working together:

- **IKsolverNode**: Inverse kinematics service server that solves IK problems and generates URDF files
- **Interface**: Qt-based GUI for robot configuration and control
- **state_publisher**: Publishes joint states and manages robot state
- **urdf_reloader**: Dynamically reloads robot_state_publisher when configuration changes
- **CameraNode**: Camera publisher for visual feedback

## Prerequisites

- ROS2 (Humble or later)
- Python 3.8+
- Qt6 (PySide6)
- VTK
- OpenCV
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/renarddaxel312/Mark3.git
cd Mark3
```

2. Install system dependencies:
```bash
sudo apt-get update
sudo apt-get install -y \
    ros-humble-desktop \
    python3-pip \
    python3-pyside6 \
    python3-vtk \
    python3-opencv \
    ros-humble-cv-bridge \
    ros-humble-robot-state-publisher
```

3. Install Python dependencies:
```bash
pip3 install -r requirements.txt
```

   **Note**: ROS2 packages (rclpy, sensor_msgs, geometry_msgs, etc.) are installed via apt-get in step 2. The requirements.txt file contains only additional Python packages.

4. Build the workspace:
```bash
colcon build --symlink-install
source install/setup.bash
```

## Usage

### Launch the complete system:

```bash
ros2 launch system robot_system.launch.py
```

This will start:
- IK solver service
- Joint state publisher
- URDF reloader
- GUI interface
- Camera node (if available)

### Launch backend only (without GUI):

```bash
ros2 launch system robot_backend.launch.py
```

### Using the GUI:

1. **Configure Robot Structure**:
   - Add/remove axes using the interface
   - Select joint types: "Rotation (360)", "Rotation (180)", or "Translation"
   - Click "Envoyer la configuration" to apply changes

2. **Control Robot Position**:
   - Enter target X, Y, Z coordinates
   - Click "Envoyer la position" to solve IK and move the robot
   - Use "Retour maison" to return to home position (all joints at 0°)

3. **3D Visualization**:
   - View your robot in real-time in the 3D viewer
   - Rotate, zoom, and pan using mouse controls
   - Monitor end-effector position

## Project Structure

```
mark3/
├── src/
│   ├── IKsolverNode/          # IK solver and URDF generation
│   │   ├── IKsolverNode/      # Main IK service server
│   │   ├── kinematics.py      # Forward/inverse kinematics
│   │   └── dh_utils.py        # DH parameters and URDF generation
│   ├── Interface/             # Qt GUI application
│   │   └── Interface/         # Main interface and 3D viewer
│   ├── state_publisher/       # Joint state publisher
│   ├── ik_service_interface/  # ROS2 service definitions
│   ├── CameraNode/            # Camera publisher
│   └── system/                # Launch files
├── README.md
└── ...
```

## Topics and Services

### Topics:
- `/config/axis` (std_msgs/String): Robot configuration (JSON array of joint types)
- `/pos/angles` (std_msgs/Float32MultiArray): Joint angles in degrees
- `/pos/coord` (std_msgs/Float32MultiArray): Target position coordinates
- `/joint_states` (sensor_msgs/JointState): Joint states for TF
- `/urdf_updated` (std_msgs/String): URDF path when configuration changes
- `/camera/raw` (sensor_msgs/Image): Camera feed

### Services:
- `/solve_ik` (ik_service_interface/SolveIK): Inverse kinematics solver

## Technologies

- **ROS2**: Robot Operating System 2 for middleware
- **PySide6**: Qt framework for GUI
- **VTK**: 3D visualization and rendering
- **NumPy**: Numerical computations
- **OpenCV**: Image processing
- **URDF**: Robot description format

## Configuration

The robot supports three joint types:
- **Rotation (360)**: Continuous rotation joint (-180° to +180°)
- **Rotation (180)**: Limited rotation joint (-90° to +90°)
- **Translation**: Prismatic joint (not fully implemented)

## Development

### Building:
```bash
colcon build --symlink-install
```

### Running tests:
```bash
colcon test
```

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Authors

- **Axel Renard**
- **Rayane Yettefti**
- **Elric Colin**

