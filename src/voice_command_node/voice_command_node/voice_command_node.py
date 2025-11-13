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
        self.voice_mode = False  # Activé via l'IHM
        self.silence_counter = 0
        self.threshold = 800
        self.silence_max = 15
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
        self.create_subscription(Bool, '/voice/commands_enabled', self.mode_callback, 10)
       
        # Clients
        self.ik_client = self.create_client(SolveIK, 'solve_ik')
        self.object_detection_client = self.create_client(DetectObject3D, 'detect_object_3d')


        self.get_logger().info("Noeud de commandes vocales démarré")


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
        state = "activé" if self.voice_mode else "désactivé"
        self.get_logger().info(f"[VOICE] Mode vocal {state}")
        if self.voice_mode:
            self.get_logger().info(f"[VOICE] En attente de commandes vocales...")


    def amp_callback(self, msg: Float32):
        if self.listening and self.voice_mode:
            amp = msg.data
           
            if amp > self.threshold:
                if not self.recording:
                    self.get_logger().info(f"[VOICE] Début enregistrement audio (amplitude {amp:.1f} > seuil {self.threshold})")
                    self.recording = True
                    self.buffer.clear()
                self.silence_counter = 0
            else:
                if self.recording:
                    self.silence_counter += 1
                    if self.silence_counter > self.silence_max:
                        self.get_logger().info(f"[VOICE] Fin enregistrement (silence détecté), traitement de l'audio...")
                        self.listening = False
                        self.process_buffer()
                        self.recording = False
                        self.silence_counter = 0


    def raw_callback(self, msg: AudioData):
        if self.recording:
            self.buffer.append(msg.data)


    def process_buffer(self):
        audio_bytes = b''.join(self.buffer)
        audio_length = len(audio_bytes) / (self.sample_rate * 2)  # Durée en secondes
        self.get_logger().info(f"[VOICE] Traitement de {len(audio_bytes)} bytes d'audio ({audio_length:.2f}s)")
        audio_data = sr.AudioData(audio_bytes, self.sample_rate, 2)
        try:
            text = self.recognizer.recognize_google(audio_data, language='fr-FR')
        except sr.UnknownValueError:
            self.get_logger().warning('[VOICE] Audio non compris par le service de reconnaissance')
            self.listening = True
            return
        except sr.RequestError as e:
            self.get_logger().error(f'[VOICE] Erreur service de reconnaissance vocale: {e}')
            self.listening = True
            return
        self.get_logger().info(f'[VOICE] Texte reconnu: "{text}"')
        self.process_command(text)


    # Traitement des commandes vocales
    def process_command(self, command: str):
        self.get_logger().info(f"[VOICE] Analyse de la commande: \"{command}\"")
        if "position initiale" in command or "retour maison" in command or "retour origine" in command:
            self.get_logger().info("[VOICE] Commande détectée: Retour à la position initiale")
            self.send_home()
        elif "va en" in command or "va" in command:
            self.get_logger().info("[VOICE] Commande détectée: Déplacement vers position")
            self.handle_go_to_position(command)
        elif "récupère" in command or "récupérer" in command:
            self.get_logger().info("[VOICE] Commande détectée: Récupération d'objet")
            self.handle_pick_object(command)
        else:
            self.get_logger().warn(f"[VOICE] Commande inconnue: \"{command}\"")
        self.listening = True


    def send_home(self):
        if self.n_joints == 0 or not self.joint_names:
            self.get_logger().warn("[VOICE] Impossible d'envoyer position initiale: configuration non disponible")
            return
        msg = Float32MultiArray()
        msg.data = [0.0] * self.n_joints
        self.angles_pub.publish(msg)
        self.get_logger().info(f'[VOICE] Position initiale publiée: {self.n_joints} joints à 0.0 rad sur /pos/angles')
   
    # Commande : Va en X Y Z
    def handle_go_to_position(self, command: str):
        try:
            # Exemple : "va en 0.2 0.1 0.3"
            parts = command.replace("va en", "").strip().split()
            x, y, z = [float(v.replace(",", ".")) for v in parts[:3]]
            self.get_logger().info(f"[VOICE] Position extraite de la commande: X={x:.3f} Y={y:.3f} Z={z:.3f}")
            self.publish_and_call_ik(x, y, z)
        except ValueError as e:
            self.get_logger().error(f"[VOICE] Erreur parsing coordonnées dans \"{command}\": {e}")
        except Exception as e:
            self.get_logger().error(f"[VOICE] Erreur dans la commande de déplacement: {e}")


    # Commande : Va récupérer un objet
    def handle_pick_object(self, command: str):
        try:
            # Tournevis
            if "tournevis" in command or "screwdriver" in command or "outil" in command:
               
                # A FINIRRRRRRRRRRRR


                self.get_logger().info(f"[VOICE] Recherche de l'objet: tournevis")
                object_name = "screwdriver"

                if not self.object_detection_client.service_is_ready():
                    self.get_logger().error("[VOICE] Service de détection d'objets non disponible")
                    return

                self.get_logger().info(f"[VOICE] Appel du service de détection pour: {object_name}")
                req = DetectObject3D.Request()
                req.object_class = object_name
                req.known_object_size = 0.0

                future = self.object_detection_client.call_async(req)
                future.add_done_callback(self._handle_object_detection_response)
                self.get_logger().info(f"[VOICE] Requête de détection envoyée, en attente de réponse...")
            else:
                self.get_logger().warn(f"[VOICE] Objet non reconnu dans la commande: \"{command}\"")


        except Exception as e:
            self.get_logger().error(f"Erreur détection d'objet : {e}")


    def _handle_object_detection_response(self, future):
        try:
            response = future.result()
            if not response.success:
                self.get_logger().warn(f"[VOICE] Détection d'objet échouée: {response.message}")
                return

            pos = response.position_3d
            confidence = response.confidence
            detected_class = response.detected_class
            self.get_logger().info(f"[VOICE] Objet détecté: {detected_class} (confiance: {confidence:.2f})")
            self.get_logger().info(f"[VOICE] Position 3D: X={pos.x:.3f} Y={pos.y:.3f} Z={pos.z:.3f}")
            self.publish_and_call_ik(pos.x, pos.y, pos.z)


        except Exception as e:
            self.get_logger().error(f"Erreur callback détection : {e}")


    # Pub IK et appel
    def publish_and_call_ik(self, x, y, z):
        msg = Float32MultiArray()
        msg.data = [x, y, z]
        self.pos_pub.publish(msg)
        self.get_logger().info(f"[VOICE] Position publiée sur /pos/coord: x={x:.3f} y={y:.3f} z={z:.3f}")

        if not self.ik_client.service_is_ready():
            self.get_logger().error("[VOICE] Service IK non disponible")
            return

        self.get_logger().info(f"[VOICE] Appel du service IK pour la position: ({x:.3f}, {y:.3f}, {z:.3f})")
        req = SolveIK.Request()
        req.target_position = Point(x=float(x), y=float(y), z=float(z))
        req.use_orientation = False

        future = self.ik_client.call_async(req)
        future.add_done_callback(self._handle_ik_response)
        self.get_logger().info(f"[VOICE] Requête IK envoyée, en attente de réponse...")


    def _handle_ik_response(self, future):
        try:
            response = future.result()
            if response.success:
                joint_angles = response.joint_angles
                msg = Float32MultiArray()
                msg.data = joint_angles
                self.angles_pub.publish(msg)
                self.get_logger().info(f"[VOICE] IK résolue avec succès: {len(joint_angles)} angles calculés")
                self.get_logger().info(f"[VOICE] Angles publiés sur /pos/angles: {[f'{a:.2f}' for a in joint_angles]}")
            else:
                self.get_logger().warn(f"[VOICE] Échec résolution IK: {response.message}")
        except Exception as e:
            self.get_logger().error(f"[VOICE] Erreur callback IK: {e}")


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

