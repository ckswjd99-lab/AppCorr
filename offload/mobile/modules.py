import multiprocessing
import socket
import time
from offload.common import send_msg, recv_msg, InferenceResult

class MobileSender(multiprocessing.Process):
    """Connects to Server Receiver and streams Config/Patches."""

    def __init__(self, server_ip, server_port, input_queue):
        super().__init__()
        self.server_ip = server_ip
        self.server_port = server_port
        self.input_queue = input_queue

    def run(self):
        connected = False
        while not connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.server_ip, self.server_port))
                connected = True
                print(f"[MobileSender] Connected to {self.server_ip}:{self.server_port}")
            except ConnectionRefusedError:
                print("[MobileSender] Connection failed. Retrying in 1s...")
                time.sleep(1)

        with s:
            while True:
                item = self.input_queue.get()
                
                if item == 'STOP':
                    send_msg(s, 'STOP')
                    break
                
                send_msg(s, item)
        print("[MobileSender] Stopped.")

class MobileReceiver(multiprocessing.Process):
    """Connects to Server Sender and logs InferenceResults."""

    def __init__(self, server_ip, server_port, feedback_queue):
        super().__init__()
        self.server_ip = server_ip
        self.server_port = server_port
        self.feedback_queue = feedback_queue

    def run(self):
        # Connection Retry Logic
        connected = False
        while not connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.server_ip, self.server_port))
                connected = True
                print(f"[MobileReceiver] Connected to {self.server_ip}:{self.server_port}")
            except ConnectionRefusedError:
                print("[MobileReceiver] Connection failed. Retrying in 1s...")
                time.sleep(1)

        with s:
            while True:
                result = recv_msg(s)
                if isinstance(result, InferenceResult) or isinstance(result, float):
                    self.feedback_queue.put(result)
                elif result is None:
                    break
        print("[MobileReceiver] Stopped.")