import multiprocessing
import socket
from offload.common import send_msg, recv_msg, ExperimentConfig

class ServerReceiver(multiprocessing.Process):
    """Receives Config and Patches from Mobile."""
    
    # [FIX] Accept shutdown_event
    def __init__(self, port, sched_queue, control_queue, shutdown_event):
        super().__init__()
        self.port = port
        self.sched_queue = sched_queue
        self.control_queue = control_queue
        self.shutdown_event = shutdown_event

    def run(self):
        print(f"[ServerReceiver] Listening on {self.port}...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', self.port))
            s.listen(1)
            
            # Wait for connection (Blocking)
            # If we want to exit cleanly even without connection, we need timeout/select,
            # but for now, we assume mobile connects at least once.
            conn, addr = s.accept()
            
            with conn:
                print(f"[ServerReceiver] Connected by {addr}")
                
                # Handshake
                msg = recv_msg(conn)
                if isinstance(msg, tuple) and msg[0] == 'CONFIG':
                    config = msg[1]
                    print(f"[ServerReceiver] Handshake success. Policy: {config.scheduler_policy_name}")
                    self.control_queue.put(msg)
                else:
                    print(f"[ServerReceiver] Handshake failed or invalid format: {type(msg)}")
                
                # 2. Data Loop
                while True:
                    msg = recv_msg(conn)
                    
                    if msg is None or msg == 'STOP':
                        print("[ServerReceiver] Received STOP signal or Client disconnected.")
                        break
                    
                    if isinstance(msg, tuple):
                         if msg[0] == 'TIME_SYNC':
                             self.control_queue.put(msg)
                         else:
                             print(f"[ServerReceiver] Ignoring unknown control tuple: {msg[0]}")
                    else:
                        self.sched_queue.put(msg)
                    
        print("[ServerReceiver] Triggering Server Shutdown...")
        self.shutdown_event.set() # Signal Main process to exit

class ServerSender(multiprocessing.Process):
    """Sends InferenceResult back to Mobile."""
    
    def __init__(self, port, input_queue):
        super().__init__()
        self.port = port
        self.input_queue = input_queue

    def run(self):
        print(f"[ServerSender] Waiting for Mobile connection on {self.port}...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', self.port))
            s.listen(1)
            conn, addr = s.accept()
            
            with conn:
                print(f"[ServerSender] Mobile connected from {addr}")
                while True:
                    result = self.input_queue.get()
                    if result == 'STOP': break
                    send_msg(conn, result)