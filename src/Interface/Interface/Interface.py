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
from std_msgs.msg import String
from ik_service_interface.srv import SolveIK
from object_detection_interface.srv import DetectObject3D
from geometry_msgs.msg import Point, Vector3
from ament_index_python.packages import get_package_share_directory

from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from ultralytics import YOLO

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QScrollArea, QFrame, QLineEdit, QGroupBox, QCheckBox
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
        self.service_call_in_progress = False  # Éviter les appels multiples en parallèle
        self.service_call_id = 0  # ID unique pour chaque appel de service
        self.pending_service_calls = {}  # Dictionnaire pour suivre les appels en cours

        self.subscription = self.create_subscription(
            Image, '/camera/raw', self.callback, 10)

        self.joint_states_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, 10)
        
        self.urdf_updated_sub = self.create_subscription(
            String, '/urdf_updated', self.urdf_updated_callback, 10)
        
        self.urdf_update_callback = None

        self.pos_pub = self.create_publisher(Float32MultiArray, '/pos/coord', 10)
        
        self.angles_pub = self.create_publisher(Float32MultiArray, '/pos/angles', 10)
        
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

    def callback(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            if self.detection_enabled and self.yolo_model is not None and self.image is not None:
                self.detect_and_annotate()
        except Exception as e:
            self.get_logger().error(f"Erreur conversion image : {e}")
    
    def detect_and_annotate(self):
        try:
            # Seuil de confiance minimum (0.25 par défaut pour YOLO)
            conf_threshold = 0.25
            results = self.yolo_model(self.image, conf=conf_threshold, verbose=False)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                annotated_image = results[0].plot()
                
                best_detection = None
                best_confidence = 0.0
                
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.yolo_model.names[cls]
                    
                    # Filtrer uniquement les tournevis (screwdriver)
                    if class_name.lower() != 'screwdriver':
                        continue
                    
                    if conf > best_confidence:
                        best_confidence = conf
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        best_detection = {
                            'bbox': [x1, y1, x2, y2],
                            'class': class_name,
                            'confidence': conf
                        }
                
                # Log pour debug
                if best_detection:
                    self.get_logger().info(f"Détection: {best_detection['class']} (conf: {best_detection['confidence']:.2f})")
                else:
                    self.get_logger().debug("Aucun tournevis détecté avec confiance suffisante")
                
                if best_detection:
                    x1, y1, x2, y2 = best_detection['bbox']
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    label = f"{best_detection['class']} {best_detection['confidence']:.2f}"
                    cv2.putText(annotated_image, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Mettre à jour last_detection sans écraser les coordonnées existantes
                    if self.last_detection is None:
                        self.last_detection = {}
                    # Mettre à jour seulement les champs de détection, pas les coordonnées
                    detection_data = {
                        'class': best_detection['class'],
                        'confidence': best_detection['confidence'],
                        'bbox': best_detection['bbox']
                    }
                    # Préserver les coordonnées existantes si elles existent
                    if 'position_3d' in self.last_detection:
                        detection_data['position_3d'] = self.last_detection['position_3d']
                    if 'position_camera' in self.last_detection:
                        detection_data['position_camera'] = self.last_detection['position_camera']
                    self.last_detection.update(detection_data)
                    self.image = annotated_image
                    
                    # Éviter les appels multiples en parallèle
                    if not self.service_call_in_progress and self.object_detection_client.service_is_ready():
                        self.service_call_in_progress = True
                        self.service_call_id += 1
                        call_id = self.service_call_id
                        
                        request = DetectObject3D.Request()
                        request.object_class = best_detection['class']
                        request.known_object_size = 0.0
                        
                        self.get_logger().info(f"📞 Appel service détection #{call_id} pour {best_detection['class']} (conf: {best_detection['confidence']:.2f})")
                        future = self.object_detection_client.call_async(request)
                        
                        # Stocker les informations de l'appel
                        self.pending_service_calls[call_id] = {
                            'future': future,
                            'detection': best_detection.copy(),
                            'timestamp': time.time()
                        }
                        
                        future.add_done_callback(lambda f, cid=call_id: self._detection_response_callback(f, cid))
                    elif self.service_call_in_progress:
                        self.get_logger().debug("Appel service déjà en cours, ignoré")
                    else:
                        self.get_logger().warn("Service détection non disponible")
        except Exception as e:
            self.get_logger().error(f"Erreur détection YOLO: {e}")
    
    def _detection_response_callback(self, future, call_id):
        try:
            # Vérifier si cet appel est toujours valide (pas trop ancien)
            if call_id not in self.pending_service_calls:
                self.get_logger().warn(f"Callback pour appel #{call_id} ignoré (appel expiré ou annulé)")
                self.service_call_in_progress = False
                return
            
            call_info = self.pending_service_calls.pop(call_id)
            response = future.result()
            self.service_call_in_progress = False  # Libérer le verrou
            
            self.get_logger().info(f"📥 Callback détection #{call_id} reçu: success={response.success}, message='{response.message}'")
            
            if response.success:
                self.get_logger().info(f"  ✓ Coordonnées reçues: 3D=({response.position_3d.x:.3f}, {response.position_3d.y:.3f}, {response.position_3d.z:.3f})")
                # Mettre à jour last_detection avec les coordonnées
                if self.last_detection is None:
                    self.last_detection = {}
                
                # S'assurer que les coordonnées sont bien ajoutées
                self.last_detection['position_3d'] = {
                    'x': float(response.position_3d.x),
                    'y': float(response.position_3d.y),
                    'z': float(response.position_3d.z)
                }
                self.last_detection['position_camera'] = {
                    'x': float(response.position_camera_frame.x),
                    'y': float(response.position_camera_frame.y),
                    'z': float(response.position_camera_frame.z)
                }
                
                self.get_logger().info(
                    f"Coordonnées reçues et mises à jour - 3D: X={response.position_3d.x:.3f}m "
                    f"Y={response.position_3d.y:.3f}m Z={response.position_3d.z:.3f}m | "
                    f"Caméra: X={response.position_camera_frame.x:.3f}m "
                    f"Y={response.position_camera_frame.y:.3f}m Z={response.position_camera_frame.z:.3f}m"
                )
                self.get_logger().info(f"last_detection keys après mise à jour: {list(self.last_detection.keys())}")
            else:
                self.get_logger().warn(f"Service détection #{call_id} échoué: {response.message}")
                # Mettre à jour last_detection même en cas d'échec pour éviter "En attente" permanent
                if self.last_detection is None:
                    self.last_detection = {}
        except Exception as e:
            if call_id in self.pending_service_calls:
                self.pending_service_calls.pop(call_id)
            self.service_call_in_progress = False  # Libérer le verrou en cas d'erreur
            self.get_logger().error(f"Erreur callback détection #{call_id}: {e}", exc_info=True)

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
            response = future.result()
            
            if response.success:
                joint_angles = response.joint_angles
                self.get_logger().info(f"IK résolue: {np.round(joint_angles, 2).tolist()}")
                
                angles_msg = Float32MultiArray()
                angles_msg.data = joint_angles
                self.angles_pub.publish(angles_msg)
                self.get_logger().info(f"Angles publiés sur /pos/angles")
            else:
                self.get_logger().error(f"Échec IK: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Erreur réponse IK: {e}")


class RosThread(QThread):
    new_image = Signal(object)

    def __init__(self):
        super().__init__()
        rclpy.init(args=None)
        self.node = CameraSubscriber()
        self.running = True

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

        self.send_button = QPushButton("Envoyer la position")
        self.send_button.clicked.connect(self.send_position)
        
        self.use_screwdriver_button = QPushButton("Utiliser position du tournevis")
        self.use_screwdriver_button.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        self.use_screwdriver_button.clicked.connect(self.use_screwdriver_position)

        os.environ["QT_QPA_PLATFORMTHEME"] = "qt5ct"

        grid_inputs.addWidget(QLabel("X :"), 0, 0)
        grid_inputs.addWidget(self.x_input, 0, 1)
        grid_inputs.addWidget(QLabel("Y :"), 1, 0)
        grid_inputs.addWidget(self.y_input, 1, 1)
        grid_inputs.addWidget(QLabel("Z :"), 2, 0)
        grid_inputs.addWidget(self.z_input, 2, 1)
        grid_inputs.addWidget(self.send_button, 3, 0, 1, 2)
        grid_inputs.addWidget(self.use_screwdriver_button, 4, 0, 1, 2)

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
        
        self.detection_toggle = QCheckBox("Activer la détection YOLO en temps réel")
        self.detection_toggle.setStyleSheet("""
            QCheckBox {
                font-weight: bold;
                spacing: 10px;
                color: white;
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
        video_layout.addWidget(self.detection_toggle)
        
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

        grid.addWidget(self.left_panel, 0, 0, 2, 1)
        grid.addWidget(self.zone_visu, 0, 1, 2, 1)
        grid.addWidget(right_container, 0, 2, 2, 1)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 4)
        grid.setColumnStretch(2, 0)
        grid.setRowStretch(0, 2)
        grid.setRowStretch(1, 1)



        self.ros_thread = RosThread()
        self.ros_thread.new_image.connect(self.update_image)
        self.ros_thread.start()
        
        self.ros_thread.node.urdf_update_callback = self.on_urdf_updated
        
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
            class_name = detection.get('class', 'unknown')
            confidence = detection.get('confidence', 0.0)
            info_text = f"Objet: {class_name} ({confidence:.1%})\n"
            
            # Vérifier et afficher les coordonnées
            has_position_3d = 'position_3d' in detection and detection['position_3d'] is not None
            has_position_camera = 'position_camera' in detection and detection['position_camera'] is not None
            
            if has_position_3d:
                pos = detection['position_3d']
                # Vérifier que les valeurs ne sont pas toutes à zéro
                if isinstance(pos, dict) and any(abs(pos.get(k, 0)) > 0.001 for k in ['x', 'y', 'z']):
                    info_text += f"Position 3D: X={pos.get('x', 0):.3f}m Y={pos.get('y', 0):.3f}m Z={pos.get('z', 0):.3f}m"
                    self.ros_thread.node.get_logger().info(f"✓ Affichage position_3d: X={pos.get('x', 0):.3f} Y={pos.get('y', 0):.3f} Z={pos.get('z', 0):.3f}")
                else:
                    info_text += "Position 3D: En attente... (valeurs nulles)"
                    self.ros_thread.node.get_logger().warn(f"Position 3D présente mais valeurs nulles: {pos}")
            elif has_position_camera:
                pos = detection['position_camera']
                if isinstance(pos, dict) and any(abs(pos.get(k, 0)) > 0.001 for k in ['x', 'y', 'z']):
                    info_text += f"Position caméra: X={pos.get('x', 0):.3f}m Y={pos.get('y', 0):.3f}m Z={pos.get('z', 0):.3f}m"
                    self.ros_thread.node.get_logger().info(f"✓ Affichage position_camera: X={pos.get('x', 0):.3f} Y={pos.get('y', 0):.3f} Z={pos.get('z', 0):.3f}")
                else:
                    info_text += "Position 3D: En attente... (valeurs nulles)"
                    self.ros_thread.node.get_logger().warn(f"Position caméra présente mais valeurs nulles: {pos}")
            else:
                info_text += "Position 3D: En attente..."
                self.ros_thread.node.get_logger().warn(
                    f"⚠ Aucune position 3D disponible. Keys dans detection: {list(detection.keys())}"
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

    def send_position(self):
        try:
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            z = float(self.z_input.text())
            self.ros_thread.node.publish_position(x, y, z)
        except ValueError:
            print("Coordonnées invalides")
    
    def use_screwdriver_position(self):
        """Utilise automatiquement les coordonnées du tournevis détecté"""
        if not self.ros_thread.node.last_detection:
            self.ros_thread.node.get_logger().warn("Aucun tournevis détecté")
            return
        
        detection = self.ros_thread.node.last_detection
        
        # Vérifier que c'est bien un tournevis
        if detection.get('class', '').lower() != 'screwdriver':
            self.ros_thread.node.get_logger().warn(f"Objet détecté n'est pas un tournevis: {detection.get('class', 'unknown')}")
            return
        
        # Utiliser les coordonnées 3D si disponibles
        if 'position_3d' in detection:
            pos = detection['position_3d']
            x = pos['x']
            y = pos['y']
            z = pos['z']
            self.ros_thread.node.get_logger().info(f"Utilisation position tournevis: X={x:.3f}m Y={y:.3f}m Z={z:.3f}m")
            self.ros_thread.node.publish_position(x, y, z)
            
            # Mettre à jour les champs de saisie pour affichage
            self.x_input.setText(f"{x:.3f}")
            self.y_input.setText(f"{y:.3f}")
            self.z_input.setText(f"{z:.3f}")
        elif 'position_camera' in detection:
            pos = detection['position_camera']
            x = pos['x']
            y = pos['y']
            z = pos['z']
            self.ros_thread.node.get_logger().info(f"Utilisation position caméra tournevis: X={x:.3f}m Y={y:.3f}m Z={z:.3f}m")
            self.ros_thread.node.publish_position(x, y, z)
            
            # Mettre à jour les champs de saisie
            self.x_input.setText(f"{x:.3f}")
            self.y_input.setText(f"{y:.3f}")
            self.z_input.setText(f"{z:.3f}")
        else:
            self.ros_thread.node.get_logger().warn("Position 3D du tournevis non disponible")

    def closeEvent(self, event):
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
