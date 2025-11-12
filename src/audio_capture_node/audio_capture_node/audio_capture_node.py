#!/usr/bin/env python3
import os
import sys

venv_python = os.path.expanduser('~/venv/bin/python3')
if os.path.exists(venv_python) and sys.executable != venv_python:
    os.execv(venv_python, [venv_python] + sys.argv)
    
import rclpy
from rclpy.node import Node
from audio_common_msgs.msg import AudioData
from std_msgs.msg import Float32
import pyaudio
import numpy as np

class AudioCaptureNode(Node):
    def __init__(self):
        super().__init__('audio_capture_node')
        # Paramètres
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('chunk_size', 1024)
        rate = self.get_parameter('sample_rate').value
        chunk = self.get_parameter('chunk_size').value

        # Publisher audio et amplitude
        self.pub_audio = self.create_publisher(AudioData, 'audio/raw', 10)
        self.pub_amp = self.create_publisher(Float32, 'audio/amplitude', 10)

        # Initialisation PyAudio
        self.pa = pyaudio.PyAudio()
        for attempt in range(10):
            try:
                self.stream = self.pa.open(
                    rate=rate,
                    channels=1,
                    format=pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=chunk)
                break
            except Exception as e:
                self.get_logger().warn(f"Tentative {attempt+1}/10 : Micro non disponible. Nouvelle tentative dans 1s...")
                time.sleep(1)
        else:
            self.get_logger().error("Impossible d'ouvrir le flux audio après 10 tentatives.")
            raise RuntimeError("Audio stream init failed")


        # Boucle timer pour lecture
        self.create_timer(chunk / rate, self.capture_callback)
        self.get_logger().info('AudioCaptureNode initialisé.')

    def capture_callback(self):
        data = self.stream.read(self.get_parameter('chunk_size').value, exception_on_overflow=False)
        # Publier audio brut
        msg = AudioData(data=data)
        self.pub_audio.publish(msg)
        # Calcul amplitude RMS
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples**2)))
        self.pub_amp.publish(Float32(data=rms))

    def destroy_node(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.pa.terminate()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AudioCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()