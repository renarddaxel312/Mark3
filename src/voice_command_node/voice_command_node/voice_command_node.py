#!/usr/bin/env python3
import os
import sys

venv_python = os.path.expanduser('~/venv/bin/python3')
if os.path.exists(venv_python) and sys.executable != venv_python:
    os.execv(venv_python, [venv_python] + sys.argv)

import rclpy
from rclpy.node import Node
import speech_recognition as sr
from std_msgs.msg import Float32MultiArray, Bool, Float32, String
from geometry_msgs.msg import Point, Vector3
from ik_service_interface.srv import SolveIK
from object_detection_interface.srv import DetectObject3D
from audio_common_msgs.msg import AudioData
import json
import numpy as np
class VoiceCommandNode(Node):
    def __init__(self):
        super().__init__('voice_command_node')

        # Voice Detecttion
        self.recognizer = sr.Recognizer()

        # Variables
        self.sample_rate = 16000
        self.buffer = []
        self.listening = True
        self.voice_mode = True # De base le voice mode est désactiver sur l'IHM FALSE DE BASE
        self.silence_counter = 0
        self.threshold = 800         # seuil d’amplitude pour démarrer l’enregistrement
        self.silence_max = 10         # nombre de cycles de silence avant d’arrêter
        self.recording = False
        self.n_joints = 0
        self.joint_names = []
        # Pub
        self.pos_pub = self.create_publisher(Float32MultiArray, '/pos/coord', 10)
        self.angles_pub = self.create_publisher(Float32MultiArray, '/pos/angles', 10)

        # Sub
        self.create_subscription(Float32, '/audio/amplitude', self.amp_callback, 10)
        self.create_subscription(AudioData, '/audio/raw', self.raw_callback, 10)
        self.config_subscription = self.create_subscription(
            String,
            '/config/axis',
            self.config_callback,
            10
        )
        # self.create_subscription(Bool, '/micro/enable', self.mode_callback, 10) # Changer le / ici
        
        # Clients
        self.ik_client = self.create_client(SolveIK, 'solve_ik')
        self.object_detection_client = self.create_client(DetectObject3D, 'detect_object_3d')

        self.get_logger().info("✅ Noeud de commandes vocales démarré")

        # Attente des services
        self.ik_client.wait_for_service(timeout_sec=3.0)
        self.object_detection_client.wait_for_service(timeout_sec=3.0)

    def config_callback(self, msg):
        try:
            config_data = json.loads(msg.data)
            if isinstance(config_data, list):
                joint_types = config_data
            elif 'axes' in config_data:
                joint_types = config_data['axes']
            else:
                return    
            self.n_joints = len(joint_types)
            self.joint_names = [f'joint_{i}' for i in range(self.n_joints)]

        except Exception as e:
            self.get_logger().error(f'Erreur config: {e}')

    def mode_callback(self, msg: Bool):
        self.voice_mode = msg.data
        self.get_logger().info(f"🎙️ Mode vocal {'activé' if self.voice_mode else 'désactivé'}")

    def amp_callback(self, msg: Float32):
        if self.listening and self.voice_mode:
            amp = msg.data
            
            if amp > self.threshold:
                if not self.recording:
                    self.get_logger().info(f"Début enregistrement (amp {amp:.1f} > thr {self.threshold})")
                    self.recording = True
                    self.buffer.clear()
                self.silence_counter = 0
            else:
                if self.recording:
                    self.silence_counter += 1
                    if self.silence_counter > self.silence_max:
                        self.get_logger().info('Fin enregistrement, traitement…')
                        self.listening = False
                        self.process_buffer()
                        self.recording = False
                        self.silence_counter = 0

    def raw_callback(self, msg: AudioData):
        if self.recording:
            self.buffer.append(msg.data)

    def process_buffer(self):
        audio_bytes = b''.join(self.buffer)
        audio_data = sr.AudioData(audio_bytes, self.sample_rate, 2)
        try:
            text = self.recognizer.recognize_google(audio_data, language='fr-FR')
        except sr.UnknownValueError:
            self.get_logger().warning('Audio non compris')
            self.listening = True
            return
        except sr.RequestError as e:
            self.get_logger().error(f'Erreur reconnaissance: {e}')
            return
        self.get_logger().info(f'Texte reconnu: "{text}"')
        self.process_command(text)

    # Traitement des commandes vocales
    def process_command(self, command: str):
        if "position initiale" in command or "retour maison" in command or "retour origine" in command:
            self.send_home()
        # elif command.startswith("va en"):
        #     self.handle_go_to_position(command)
        elif "récupère" in command or "récupérer" in command:
            self.handle_pick_object(command)
        else:
            self.get_logger().warn("Commande inconnue.")
        self.listening = True

    def send_home(self):
        if self.n_joints == 0 or not self.joint_names:
            return
        msg = Float32MultiArray()
        msg.data = [0.0] * self.n_joints
        self.angles_pub.publish(msg)
        self.get_logger().info(f'Position zéro publiée: {self.n_joints} joints à 0.0 rad')
    
    # Commande : Va en X Y Z
    # def handle_go_to_position(self, command: str):
    #     try:
    #         # Exemple : "va en 0.2 0.1 0.3"
    #         parts = command.replace("va en", "").strip().split()
    #         x, y, z = [float(v.replace(",", ".")) for v in parts[:3]]

    #         self.publish_and_call_ik(x, y, z)

    #     except Exception as e:
    #         self.get_logger().error(f"Erreur dans la commande de déplacement : {e}")

    # Commande : Va récupérer un objet
    def handle_pick_object(self, command: str):
        try:
            # Exemple : "va récupérer le tournevis"
            object_name = command.split("récup")[1].strip().replace("le", "").replace("la", "").strip()
            self.get_logger().info(f"🔎 Recherche de l'objet : {object_name}")

            if not self.object_detection_client.service_is_ready():
                self.get_logger().error("Service de détection non disponible.")
                return

            req = DetectObject3D.Request()
            req.object_class = object_name
            req.known_object_size = 0.0

            future = self.object_detection_client.call_async(req)
            future.add_done_callback(self._handle_object_detection_response)

        except Exception as e:
            self.get_logger().error(f"Erreur détection d'objet : {e}")

    def _handle_object_detection_response(self, future):
        try:
            response = future.result()
            if not response.success:
                self.get_logger().warn(f"Détection échouée : {response.message}")
                return

            pos = response.position_3d
            self.get_logger().info(f"✅ Objet détecté à : X={pos.x:.3f} Y={pos.y:.3f} Z={pos.z:.3f}")
            self.publish_and_call_ik(pos.x, pos.y, pos.z)

        except Exception as e:
            self.get_logger().error(f"Erreur callback détection : {e}")

    # Pub IK et appel
    def publish_and_call_ik(self, x, y, z):
        msg = Float32MultiArray()
        msg.data = [x, y, z]
        self.pos_pub.publish(msg)
        self.get_logger().info(f"📤 Position publiée : x={x} y={y} z={z}")

        if not self.ik_client.service_is_ready():
            self.get_logger().error("Service IK non disponible.")
            return

        req = SolveIK.Request()
        req.target_position = Point(x=float(x), y=float(y), z=float(z))
        req.use_orientation = False

        future = self.ik_client.call_async(req)
        future.add_done_callback(self._handle_ik_response)

    def _handle_ik_response(self, future):
        try:
            response = future.result()
            if response.success:
                msg = Float32MultiArray()
                msg.data = response.joint_angles
                self.angles_pub.publish(msg)
                self.get_logger().info("✅ Angles IK publiés.")
            else:
                self.get_logger().warn(f"Échec IK : {response.message}")
        except Exception as e:
            self.get_logger().error(f"Erreur callback IK : {e}")

def main(args=None):
    rclpy.init(args=args)
    node = VoiceCommandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()