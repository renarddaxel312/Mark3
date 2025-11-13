#!/usr/bin/env python3
import os
import sys

# Forcer l'utilisation du venv Python si disponible
venv_python = os.path.expanduser('~/venv/bin/python3')
if os.path.exists(venv_python) and sys.executable != venv_python:
    try:
        import ultralytics
    except ImportError:
        # Si ultralytics n'est pas disponible, utiliser le venv
        os.execv(venv_python, [venv_python] + sys.argv)


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String, Bool
from ik_service_interface.srv import SolveIK
from object_detection_interface.srv import DetectObject3D
from geometry_msgs.msg import Point, Vector3
from ament_index_python.packages import get_package_share_directory

from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import json

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QScrollArea, QFrame, QLineEdit, QGroupBox, QCheckBox, QPlainTextEdit
)

from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap

from Interface.robot_3d_viewer import Robot3DViewer


class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('IHM')
        self.bridge = CvBridge()
        self.image = None
        self.joint_states = None
        self.detection_enabled = False
        self.last_detection = None
        self.detection_lock = threading.Lock()
        self.yolo_processing = False

        self.raw_image_sub = self.create_subscription(
            Image, '/camera/raw', self.raw_image_callback, 10)
        
        self.object_image_sub = self.create_subscription(
            Image, '/camera/object', self.object_image_callback, 10)

        self.joint_states_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, 10)
        
        self.urdf_updated_sub = self.create_subscription(
            String, '/urdf_updated', self.urdf_updated_callback, 10)
        
        self.reachability_intervals_sub = self.create_subscription(
            String, '/reachability/intervals', self.reachability_intervals_callback, 10)
        
        self.reachability_intervals = None
        self.urdf_update_callback = None

        self.pos_pub = self.create_publisher(Float32MultiArray, '/pos/coord', 10)

        self.angles_pub = self.create_publisher(Float32MultiArray, '/pos/angles', 10)
        self.voice_enable_pub = self.create_publisher(Bool, '/voice/commands_enabled', 10)
        self.voice_commands_enabled = False
        
        self.ik_client = self.create_client(SolveIK, 'solve_ik')
        self.object_detection_client = self.create_client(DetectObject3D, 'detect_object_3d')
        
        model_path = self.declare_parameter('yolo_model', 'yolov8n.pt').value
        try:
            self.yolo_model = YOLO(model_path)
            self.get_logger().info(f"YOLO model loaded: {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            self.yolo_model = None

        self.get_logger().info("Node ROS2 démarré")
        
        if not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Service IK non disponible')
        
        if not self.object_detection_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Service object detection non disponible')
    
    def joint_states_callback(self, msg):
        self.joint_states = msg
    
    def urdf_updated_callback(self, msg):
        urdf_path = msg.data
        self.get_logger().info(f"URDF mis à jour reçu: {urdf_path}")
        
        if self.urdf_update_callback:
            self.urdf_update_callback(urdf_path)
    
    def reachability_intervals_callback(self, msg):
        try:
            intervals = json.loads(msg.data)
            self.reachability_intervals = intervals
            self.get_logger().info(f"Intervalles reachability reçus: {intervals}")
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Erreur parsing intervalles reachability: {e}")
        except Exception as e:
            self.get_logger().error(f"Erreur callback intervalles reachability: {e}")

    def raw_image_callback(self, msg):
        if not self.detection_enabled:
            try:
                self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().error(f"Erreur conversion image raw: {e}")

    def object_image_callback(self, msg):
        if self.detection_enabled:
            try:
                self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                
                if not self.yolo_processing:
                    threading.Thread(target=self.request_detection_info, daemon=True).start()
            except Exception as e:
                self.get_logger().error(f"Erreur conversion image object: {e}")

    def request_detection_info(self):
        with self.detection_lock:
            if self.yolo_processing:
                return
            self.yolo_processing = True
        
        try:
            if self.object_detection_client.service_is_ready():
                request = DetectObject3D.Request()
                request.object_class = 'screwdriver'
                request.known_object_size = 0.0
                
                future = self.object_detection_client.call_async(request)
                future.add_done_callback(self._detection_response_callback)
        except Exception as e:
            self.get_logger().error(f"Erreur appel service détection: {e}")
        finally:
            with self.detection_lock:
                self.yolo_processing = False
    
    def _detection_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                detection_info = {}

                detection_info['class'] = response.detected_class
                detection_info['confidence'] = float(response.confidence)
                
                detection_info['position_camera_m'] = {
                    'x': float(response.position_camera_frame.x),
                    'y': float(response.position_camera_frame.y),
                    'z': float(response.position_camera_frame.z)
                }
                detection_info['position_3d'] = {
                    'x': float(response.position_3d.x),
                    'y': float(response.position_3d.y),
                    'z': float(response.position_3d.z)
                }

                if response.message:
                    try:
                        payload = json.loads(response.message)
                        if isinstance(payload, dict):
                            if 'bbox' in payload:
                                detection_info['bbox'] = [float(v) for v in payload['bbox']]
                            if 'center' in payload and isinstance(payload['center'], (list, tuple)) and len(payload['center']) >= 2:
                                detection_info['center'] = {
                                    'x': float(payload['center'][0]),
                                    'y': float(payload['center'][1])
                                }
                            if 'uses_segmentation' in payload:
                                detection_info['uses_segmentation'] = bool(payload['uses_segmentation'])
                            if 'position_base_m' in payload and isinstance(payload['position_base_m'], (list, tuple)) and len(payload['position_base_m']) >= 3:
                                detection_info['position_base'] = {
                                    'x': float(payload['position_base_m'][0]),
                                    'y': float(payload['position_base_m'][1]),
                                    'z': float(payload['position_base_m'][2])
                                }
                    except json.JSONDecodeError:
                        pass

                self.last_detection = detection_info
            else:
                self.last_detection = None
        except Exception as e:
            self.get_logger().error(f"Erreur callback détection: {e}")
            self.last_detection = None

    def set_voice_commands_enabled(self, enabled: bool):
        enabled = bool(enabled)
        if self.voice_commands_enabled == enabled:
            # Still publish to ensure consumers get state (e.g., on reconnect)
            msg = Bool()
            msg.data = enabled
            self.voice_enable_pub.publish(msg)
            return

        self.voice_commands_enabled = enabled
        msg = Bool()
        msg.data = enabled
        self.voice_enable_pub.publish(msg)
        state_label = "activées" if enabled else "désactivées"
        self.get_logger().info(f"Commandes vocales {state_label}")

    def publish_position(self, x, y, z, use_orientation=False, roll=0.0, pitch=0.0, yaw=0.0):
        msg = Float32MultiArray()
        msg.data = [x, y, z]
        self.pos_pub.publish(msg)
        self.get_logger().info(f"Position cible : x={x}, y={y}, z={z}")

        if not self.ik_client.service_is_ready():
            self.get_logger().error("Service IK non disponible!")
            return
        
        request = SolveIK.Request()
        request.target_position = Point()
        request.target_position.x = float(x)
        request.target_position.y = float(y)
        request.target_position.z = float(z)
        request.use_orientation = use_orientation
        
        if use_orientation:
            request.target_orientation = Vector3()
            request.target_orientation.x = float(roll)
            request.target_orientation.y = float(pitch)
            request.target_orientation.z = float(yaw)
            self.get_logger().info(f"   avec orientation RPY=({roll:.2f}, {pitch:.2f}, {yaw:.2f})")
        
        try:
            future = self.ik_client.call_async(request)
            future.add_done_callback(self._ik_response_callback)
        except Exception as e:
            self.get_logger().error(f"Erreur lors de l'appel IK: {e}")
    
    def _ik_response_callback(self, future):
        try:
            if not future.done():
                self.get_logger().warn("Callback IK appelé mais future pas terminé")
                return
            
            # Essayer de récupérer la réponse avec gestion d'erreur robuste
            try:
                response = future.result()
            except Exception as result_error:
                error_msg = str(result_error)
                if "Input/output error" in error_msg or "Errno 5" in error_msg:
                    # Erreur de timing/communication - ignorer silencieusement
                    # Le service IK a quand même réussi (on le voit dans les logs)
                    return
                else:
                    raise
            
            if response.success:
                joint_angles = response.joint_angles
                self.get_logger().info(f"IK résolue: {np.round(joint_angles, 2).tolist()}")
                
                angles_msg = Float32MultiArray()
                angles_msg.data = [float(angle) for angle in joint_angles]
                self.angles_pub.publish(angles_msg)
                self.get_logger().info(f"Angles publiés sur /pos/angles")
            else:
                self.get_logger().error(f"Échec IK: {response.message}")
        except Exception as e:
            error_msg = str(e)
            if "Input/output error" in error_msg or "Errno 5" in error_msg:
                self.get_logger().warn(f"Erreur de communication IK (ignorée): {error_msg}")
            else:
                self.get_logger().error(f"Erreur réponse IK: {e}")


class RosThread(QThread):
    new_image = Signal(object)
    log_message = Signal(str)  # Signal pour les logs

    def __init__(self):
        super().__init__()
        rclpy.init(args=None)
        self.node = CameraSubscriber()
        self.running = True
        
        # Configurer un handler de logging personnalisé pour ROS2
        self.setup_ros_logging()
    
    def setup_ros_logging(self):
        """Configure le logging ROS2 pour capturer les logs et les envoyer à la console."""
        import logging
        
        class ROS2LogHandler(logging.Handler):
            def __init__(self, signal):
                super().__init__()
                self.signal = signal
                # Format simplifié pour la console
                self.setFormatter(logging.Formatter('[%(levelname)s] %(name)s: %(message)s'))
            
            def emit(self, record):
                try:
                    msg = self.format(record)
                    self.signal.emit(msg)
                except Exception:
                    pass
        
        # Capturer les logs de tous les loggers ROS2
        # rclpy utilise plusieurs loggers (rclpy, rclpy.impl, etc.)
        for logger_name in ['rclpy', 'rclpy.impl', 'rclpy.logging']:
            ros_logger = logging.getLogger(logger_name)
            ros_logger.setLevel(logging.DEBUG)
            handler = ROS2LogHandler(self.log_message)
            handler.setLevel(logging.DEBUG)
            ros_logger.addHandler(handler)
        
        # Capturer aussi les logs du node spécifique
        if hasattr(self.node, 'get_logger'):
            # Le logger du node sera capturé via le handler rclpy
            pass

    def run(self):

        while rclpy.ok() and self.running:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.node.image is not None:
                self.new_image.emit(self.node.image)

    def stop(self):
        self.running = False
        self.node.destroy_node()
        rclpy.shutdown()
        self.quit()


class AxeWidget(QWidget):
    def __init__(self, numero):
        super().__init__()
        layout = QHBoxLayout()

        self.label = QLabel(f"Axe {numero} :")
        self.combo = QComboBox()
        self.combo.addItems(["Rotation (360)", "Rotation (180)", "Translation"])

        layout.addWidget(self.label)
        layout.addWidget(self.combo)
        layout.addStretch()

        self.setLayout(layout)

        self.setFixedHeight(40)

        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border: 1px solid #444;
                border-radius: 4px;
            }
            QLabel {
                color: #f0f0f0;
                padding-left: 5px;
            }
            QComboBox {
                background-color: #333;
                color: #f0f0f0;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 2px;
            }
        """)


class MainWindow(QWidget):
    urdf_reload_signal = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Configuration du robot modulaire")
        self.resize(1280, 720)
        
        self.urdf_reload_signal.connect(self.reload_urdf_slot)

        grid = QGridLayout(self)
        grid.setSpacing(15)

        self.left_panel = QFrame()
        self.left_panel.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout()

        pos_group = QGroupBox()
        pos_layout = QVBoxLayout()

        title_label = QLabel("Commande de position (XYZ)")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 14pt; margin-bottom: 10px;")
        pos_layout.addWidget(title_label)

        grid_inputs = QGridLayout()
        self.x_input = QLineEdit()
        self.y_input = QLineEdit()
        self.z_input = QLineEdit()

        self.x_input.setPlaceholderText("X")
        self.y_input.setPlaceholderText("Y")
        self.z_input.setPlaceholderText("Z")

        self.x_interval_label = QLabel("")
        self.x_interval_label.setStyleSheet("color: #888; font-size: 9pt;")
        self.y_interval_label = QLabel("")
        self.y_interval_label.setStyleSheet("color: #888; font-size: 9pt;")
        self.z_interval_label = QLabel("")
        self.z_interval_label.setStyleSheet("color: #888; font-size: 9pt;")

        self.send_button = QPushButton("Envoyer la position")
        self.send_button.clicked.connect(self.send_position)

        self.use_detection_button = QPushButton("Utiliser coordonnées détection")
        self.use_detection_button.clicked.connect(self.fill_position_from_detection)

        os.environ["QT_QPA_PLATFORMTHEME"] = "qt5ct"

        grid_inputs.addWidget(QLabel("X :"), 0, 0)
        grid_inputs.addWidget(self.x_input, 0, 1)
        grid_inputs.addWidget(self.x_interval_label, 0, 2)
        grid_inputs.addWidget(QLabel("Y :"), 1, 0)
        grid_inputs.addWidget(self.y_input, 1, 1)
        grid_inputs.addWidget(self.y_interval_label, 1, 2)
        grid_inputs.addWidget(QLabel("Z :"), 2, 0)
        grid_inputs.addWidget(self.z_input, 2, 1)
        grid_inputs.addWidget(self.z_interval_label, 2, 2)
        grid_inputs.addWidget(self.send_button, 3, 0, 1, 2)
        grid_inputs.addWidget(self.use_detection_button, 4, 0, 1, 2)

        pos_layout.addLayout(grid_inputs)
        pos_group.setLayout(pos_layout)
        left_layout.addWidget(pos_group)

        self.home_button = QPushButton("Retour maison")
        self.home_button.clicked.connect(self.send_home)
        left_layout.addWidget(self.home_button)

        switch_layout = QHBoxLayout()
        self.tool_switch = QCheckBox("Activer l'outil")
        self.tool_switch.setStyleSheet("""
            QCheckBox {
                font-weight: bold;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 40px;
                height: 20px;
                border-radius: 10px;
                background-color: #555;
                border: 1px solid #777;
            }
            QCheckBox::indicator:checked {
                background-color: #2ecc71;
            }
        """)
        self.tool_switch.stateChanged.connect(self.toggle_tool)
        switch_layout.addWidget(self.tool_switch)
        left_layout.addLayout(switch_layout)

        self.detection_toggle = QCheckBox("Activer la détection YOLO en temps réel")
        self.detection_toggle.setStyleSheet("""
            QCheckBox {
                font-weight: bold;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 40px;
                height: 20px;
                border-radius: 10px;
                background-color: #555;
                border: 1px solid #777;
            }
            QCheckBox::indicator:checked {
                background-color: #2ecc71;
            }
        """)
        self.detection_toggle.stateChanged.connect(self.toggle_detection)
        left_layout.addWidget(self.detection_toggle)

        self.voice_toggle = QCheckBox("Activer les commandes vocales")
        self.voice_toggle.setStyleSheet("""
            QCheckBox {
                font-weight: bold;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 40px;
                height: 20px;
                border-radius: 10px;
                background-color: #555;
                border: 1px solid #777;
            }
            QCheckBox::indicator:checked {
                background-color: #3498db;
            }
        """)
        self.voice_toggle.stateChanged.connect(self.toggle_voice_commands)
        left_layout.addWidget(self.voice_toggle)

        left_layout.addStretch()
        self.left_panel.setLayout(left_layout)




        self.zone_visu = QFrame()
        self.zone_visu.setFrameShape(QFrame.StyledPanel)
        visu_layout = QVBoxLayout()
        
        title_3d = QLabel("<h2>Visualisation 3D du robot</h2>")
        title_3d.setAlignment(Qt.AlignCenter)
        title_3d.setStyleSheet("color: white; margin-bottom: 10px;")
        visu_layout.addWidget(title_3d)
        
        self.robot_viewer = Robot3DViewer(self)
        visu_layout.addWidget(self.robot_viewer)
        
        self.zone_visu.setLayout(visu_layout)

        self.viz_timer = QTimer()
        self.viz_timer.timeout.connect(self.update_3d_view)
        self.viz_timer.start(100)

        video_group = QGroupBox("Caméra avec détection YOLO")
        video_layout = QVBoxLayout()
        
        self.video_label = QLabel("Flux caméra /camera/raw")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedHeight(300)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        video_layout.addWidget(self.video_label)
        
        self.detection_info_label = QLabel("Aucune détection")
        self.detection_info_label.setAlignment(Qt.AlignCenter)
        self.detection_info_label.setStyleSheet("background-color: #2b2b2b; color: #2ecc71; padding: 5px; font-weight: bold; font-size: 10pt;")
        self.detection_info_label.setFixedHeight(80)
        self.detection_info_label.setWordWrap(True)
        video_layout.addWidget(self.detection_info_label)
        
        video_group.setLayout(video_layout)

        self.zone_config = QFrame()
        self.zone_config.setFrameShape(QFrame.StyledPanel)
        config_layout = QVBoxLayout()
        config_layout.addWidget(QLabel("<h3>Composition du robot</h3>"))

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll.setWidget(self.scroll_content)

        self.axes = []
        self.separators = []

        self.add_axe()

        self.add_button = QPushButton("Ajouter un axe")
        self.add_button.setStyleSheet("font-weight: bold; background-color: #3a3a3a;")
        self.add_button.clicked.connect(self.add_axe)

        self.remove_button = QPushButton("Supprimer un axe")
        self.remove_button.setStyleSheet("font-weight: bold; background-color: #5a2a2a; color: white;")
        self.remove_button.clicked.connect(self.remove_axe)

        self.send_config_button = QPushButton("Envoyer la configuration")
        self.send_config_button.setStyleSheet("font-weight: bold; background-color: #2e5a8a; color: white;")
        self.send_config_button.clicked.connect(self.send_axis_config)
        


        config_layout.addWidget(self.scroll)
        config_layout.addWidget(self.add_button)
        config_layout.addWidget(self.remove_button)
        config_layout.addWidget(self.send_config_button)


        self.zone_config.setLayout(config_layout)


        right_panel = QVBoxLayout()
        right_panel.addWidget(video_group)
        right_panel.addWidget(self.zone_config)

        right_container = QWidget()
        right_container.setLayout(right_panel)
        right_container.setFixedWidth(420)

        # Console de sortie
        console_group = QGroupBox("Console de sortie")
        console_layout = QVBoxLayout()
        
        self.console_output = QPlainTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                font-size: 9pt;
                border: 1px solid #444;
                border-radius: 3px;
            }
        """)
        self.console_output.setMaximumBlockCount(1000)  # Limiter à 1000 lignes
        console_layout.addWidget(self.console_output)
        
        console_group.setLayout(console_layout)
        console_group.setFixedHeight(150)

        grid.addWidget(self.left_panel, 0, 0, 2, 1)
        grid.addWidget(self.zone_visu, 0, 1, 2, 1)
        grid.addWidget(right_container, 0, 2, 2, 1)
        grid.addWidget(console_group, 2, 0, 1, 3)  # Console sur toute la largeur en bas

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 4)
        grid.setColumnStretch(2, 0)
        grid.setRowStretch(0, 2)
        grid.setRowStretch(1, 1)
        grid.setRowStretch(2, 0)  # Console fixe



        # Rediriger stdout/stderr vers la console
        import sys
        from io import StringIO
        
        class ConsoleWriter:
            def __init__(self, console_widget, original_stream):
                self.console = console_widget
                self.original = original_stream
                self.buffer = StringIO()
            
            def write(self, text):
                # Écrire aussi dans le stream original pour garder la compatibilité
                if self.original:
                    self.original.write(text)
                    self.original.flush()
                
                # Ajouter à notre console
                if text.strip():  # Ignorer les lignes vides
                    self.buffer.write(text)
                    if '\n' in text or len(self.buffer.getvalue()) > 100:
                        self.flush()
            
            def flush(self):
                content = self.buffer.getvalue()
                if content:
                    # Nettoyer et afficher
                    lines = content.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            self.console.appendPlainText(line.strip())
                    self.buffer = StringIO()
                    # Auto-scroll vers le bas
                    scrollbar = self.console.verticalScrollBar()
                    scrollbar.setValue(scrollbar.maximum())
            
            def isatty(self):
                return False
            
            def fileno(self):
                return -1
        
        # Sauvegarder les streams originaux
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Rediriger stdout et stderr vers la console (mais garder aussi l'original)
        self.console_writer_stdout = ConsoleWriter(self.console_output, self.original_stdout)
        self.console_writer_stderr = ConsoleWriter(self.console_output, self.original_stderr)
        sys.stdout = self.console_writer_stdout
        sys.stderr = self.console_writer_stderr

        self.ros_thread = RosThread()
        self.ros_thread.new_image.connect(self.update_image)
        self.ros_thread.log_message.connect(self.append_to_console)  # Connecter les logs à la console
        self.ros_thread.start()
        
        self.ros_thread.node.urdf_update_callback = self.on_urdf_updated
        self.ros_thread.node.set_voice_commands_enabled(False)
        
        try:
            urdf_path = os.path.join(
                get_package_share_directory("IKsolverNode"),
                "urdf/robot_dh.urdf"
            )
            self.robot_viewer.set_urdf_path(urdf_path)
        except Exception as e:
            print(f"Impossible de charger l'URDF: {e}")
    
    def on_urdf_updated(self, urdf_path):
        self.urdf_reload_signal.emit(urdf_path)
    
    def reload_urdf_slot(self, urdf_path):
        self.robot_viewer.set_urdf_path(urdf_path)
    
    def update_3d_view(self):
        if self.ros_thread.node.joint_states is not None:
            joint_states = self.ros_thread.node.joint_states
            q_deg = [np.rad2deg(pos) for pos in joint_states.position]
            self.robot_viewer.update_joints(q_deg)
        
        # Mettre à jour les intervalles de reachability
        self.update_reachability_intervals_display()
    
    def update_reachability_intervals_display(self):
        """Met à jour l'affichage des intervalles de reachability."""
        if self.ros_thread.node.reachability_intervals is None:
            self.x_interval_label.setText("")
            self.y_interval_label.setText("")
            self.z_interval_label.setText("")
            return
        
        intervals = self.ros_thread.node.reachability_intervals
        
        try:
            x_min, x_max = intervals['x']
            y_min, y_max = intervals['y']
            z_min, z_max = intervals['z']
            
            self.x_interval_label.setText(f"[{x_min:.3f}, {x_max:.3f}]")
            self.y_interval_label.setText(f"[{y_min:.3f}, {y_max:.3f}]")
            self.z_interval_label.setText(f"[{z_min:.3f}, {z_max:.3f}]")
        except (KeyError, TypeError, ValueError) as e:
            self.x_interval_label.setText("")
            self.y_interval_label.setText("")
            self.z_interval_label.setText("")

    def toggle_tool(self, state):
        from std_msgs.msg import Bool

        if not hasattr(self, "tool_pub"):
            self.tool_pub = self.ros_thread.node.create_publisher(Bool, '/pos/tool', 10)

        msg = Bool()
        msg.data = self.tool_switch.isChecked()

        if msg.data:
            print("Outil activé")
        else:
            print("Outil désactivé")

        self.tool_pub.publish(msg)

    def send_axis_config(self):
        if not self.axes:
            print("Aucun axe défini.")
            return

        from std_msgs.msg import String
        import json

        axis_mapping = {
            "Rotation (360)": "rot360",
            "Rotation (180)": "rot180",
            "Translation": "translation"
        }
        
        axis_types = [axis_mapping[axe.combo.currentText()] for axe in self.axes]
        num_axes = len(axis_types)

        config_json = json.dumps(axis_types)
        print(f"Envoi configuration axes {num_axes} axes: {axis_types}")

        if not hasattr(self, "axis_pub"):
            self.axis_pub = self.ros_thread.node.create_publisher(String, '/config/axis', 10)

        msg = String()
        msg.data = config_json
        self.axis_pub.publish(msg)

        print("Envoi automatique de la position home après changement de config")
        QTimer.singleShot(200, self.send_home_after_config)


    def send_home_after_config(self):
        try:
            from std_msgs.msg import Float32MultiArray
            num_axes = len(self.axes)
            msg = Float32MultiArray()
            msg.data = [0.0] * num_axes
            
            if not hasattr(self, "angles_pub_home"):
                self.angles_pub_home = self.ros_thread.node.create_publisher(Float32MultiArray, '/pos/angles', 10)
            
            self.angles_pub_home.publish(msg)
            print(f"Position home envoyée ({num_axes} axes) sur /pos/angles: {msg.data}")
        except Exception as e:
            print(f"Erreur lors de l'envoi de la position home: {e}")

    def send_home(self):
        try:
            from std_msgs.msg import Float32MultiArray
            num_axes = len(self.axes) if self.axes else 5
            msg = Float32MultiArray()
            msg.data = [0.0] * num_axes
            
            if not hasattr(self, "angles_pub_home"):
                self.angles_pub_home = self.ros_thread.node.create_publisher(Float32MultiArray, '/pos/angles', 10)
            
            self.angles_pub_home.publish(msg)
            print(f"Retour maison envoyé sur /pos/angles ({num_axes} axes): {msg.data}")
        except Exception as e:
            print(f"Erreur lors de l'envoi du retour maison: {e}")

    def remove_axe(self):
        if not self.axes:
            print("Aucun axe à supprimer.")
            return

        last_axe = self.axes.pop()
        self.scroll_layout.removeWidget(last_axe)
        last_axe.deleteLater()

        if self.separators:
            last_sep = self.separators.pop()
            self.scroll_layout.removeWidget(last_sep)
            last_sep.deleteLater()

        print(f"Axe supprimé. Axes restants : {len(self.axes)}")

    def add_axe(self):
        numero = len(self.axes) + 1
        axe_widget = AxeWidget(numero)
        self.axes.append(axe_widget)
        self.scroll_layout.addWidget(axe_widget)
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        self.scroll_layout.addWidget(separator)

    def update_image(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(scaled)

        if self.ros_thread.node.last_detection:
            detection = self.ros_thread.node.last_detection
            info_text = f"Objet: {detection['class']} ({detection['confidence']:.1%})\n"
            
            if 'center' in detection:
                center = detection['center']
                info_text += f"Centre image: u={center['x']:.1f}px v={center['y']:.1f}px"
            
            if 'bbox' in detection:
                bbox = detection['bbox']
                if isinstance(bbox, list) and len(bbox) == 4:
                    info_text += f"\nBBox: x1={bbox[0]:.1f}, y1={bbox[1]:.1f}, x2={bbox[2]:.1f}, y2={bbox[3]:.1f}"

            if 'position_base' in detection:
                pos_base = detection['position_base']
                info_text += (
                    f"\nPosition base: X={pos_base['x']:.3f} m "
                    f"Y={pos_base['y']:.3f} m Z={pos_base['z']:.3f} m"
                )
            elif 'position_3d' in detection:
                pos_3d = detection['position_3d']
                info_text += (
                    f"\nPosition 3D: X={pos_3d['x']:.3f} m "
                    f"Y={pos_3d['y']:.3f} m Z={pos_3d['z']:.3f} m"
                )
            elif 'position_camera_m' in detection:
                pos_cam = detection['position_camera_m']
                info_text += (
                    f"\nPosition caméra: X={pos_cam['x']:.3f} m "
                    f"Y={pos_cam['y']:.3f} m Z={pos_cam['z']:.3f} m"
                )
            
            self.detection_info_label.setText(info_text)
            self.detection_info_label.setStyleSheet("background-color: #2b2b2b; color: #2ecc71; padding: 5px; font-weight: bold; font-size: 10pt;")
        else:
            self.detection_info_label.setText("Aucune détection")
            self.detection_info_label.setStyleSheet("background-color: #2b2b2b; color: #888; padding: 5px; font-weight: bold; font-size: 10pt;")
    
    def toggle_detection(self, state):
        self.ros_thread.node.detection_enabled = (state == Qt.CheckState.Checked.value)
        if self.ros_thread.node.detection_enabled:
            self.ros_thread.node.get_logger().info("Détection YOLO activée")
        else:
            self.ros_thread.node.get_logger().info("Détection YOLO désactivée")
            self.ros_thread.node.last_detection = None

    def fill_position_from_detection(self):
        detection = self.ros_thread.node.last_detection
        if not detection:
            print("Aucune détection disponible pour remplir les coordonnées.")
            return

        source = detection.get('position_base') or detection.get('position_3d') or detection.get('position_camera_m')
        if not source:
            print("Aucune coordonnée 3D disponible dans la dernière détection.")
            return

        try:
            self.x_input.setText(f"{float(source['x']):.3f}")
            self.y_input.setText(f"{float(source['y']):.3f}")
            self.z_input.setText(f"{float(source['z']):.3f}")
        except (KeyError, ValueError, TypeError):
            print("Format de coordonnées inattendu dans la détection.")
            return

        confidence = detection.get('confidence')
        if confidence is not None:
            print(f"Coordonnées de détection utilisées (confiance {confidence:.2f}).")
        else:
            print("Coordonnées de détection utilisées.")

    def toggle_voice_commands(self, state):
        enabled = (state == Qt.CheckState.Checked.value)
        self.ros_thread.node.set_voice_commands_enabled(enabled)
        if enabled:
            print("Commandes vocales activées.")
        else:
            print("Commandes vocales désactivées.")

    def send_position(self):
        try:
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            z = float(self.z_input.text())
            self.ros_thread.node.publish_position(x, y, z)
        except ValueError:
            print("Coordonnées invalides")

    def append_to_console(self, text):
        """Ajoute du texte à la console."""
        self.console_output.appendPlainText(text)
        # Auto-scroll vers le bas
        scrollbar = self.console_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event):
        # Restaurer stdout/stderr
        import sys
        if hasattr(self, 'original_stdout'):
            sys.stdout = self.original_stdout
        else:
            sys.stdout = sys.__stdout__
        if hasattr(self, 'original_stderr'):
            sys.stderr = self.original_stderr
        else:
            sys.stderr = sys.__stderr__
        
        self.ros_thread.stop()
        super().closeEvent(event)

def main(args=None):
    app = QApplication(sys.argv)

    dark_stylesheet = """
        QWidget {
            background-color: #1e1e1e;
            color: #f0f0f0;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QGroupBox {
            border: 1px solid #444;
            border-radius: 5px;
            margin-top: 10px;
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QLineEdit {
            background-color: #333;
            color: #f0f0f0;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 3px;
        }
        QPushButton {
            background-color: #3a3a3a;
            color: #f0f0f0;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 5px 10px;
        }
        QPushButton:hover {
            background-color: #505050;
        }
        QPushButton:pressed {
            background-color: #666;
        }
        QLabel {
            color: #f0f0f0;
        }
        QScrollArea {
            background-color: #2b2b2b;
            border: none;
        }
        QComboBox {
            background-color: #333;
            color: #f0f0f0;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 3px;
        }
        QFrame {
            background-color: #2b2b2b;
            border: 1px solid #444;
        }
    """

    app.setStyleSheet(dark_stylesheet)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
