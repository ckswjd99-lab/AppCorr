import multiprocessing
import socket
import time
from offload.common import send_msg, recv_msg

DOWNLINK_READY = "__APPCORR_DOWNLINK_READY__"


def _connect_with_retry(host, port, label):
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        try:
            sock.connect((host, port))
            sock.settimeout(None)
            print(
                f"{label} Connected to {host}:{port} "
                f"(local={sock.getsockname()}, peer={sock.getpeername()})",
                flush=True,
            )
            return sock
        except OSError as exc:
            sock.close()
            print(f"{label} Connection failed ({exc}). Retrying in 1s...", flush=True)
            time.sleep(1)


class ServerReceiver(multiprocessing.Process):
    """Receives Config and Patches from Mobile."""
    
    # Accept shutdown_event
    def __init__(
        self,
        port,
        sched_queue,
        control_queue,
        shutdown_event,
        listen_host="0.0.0.0",
        connect_host=None,
    ):
        super().__init__()
        self.port = port
        self.sched_queue = sched_queue
        self.control_queue = control_queue
        self.shutdown_event = shutdown_event
        self.listen_host = listen_host
        self.connect_host = connect_host

    def run(self):
        if self.connect_host:
            conn = _connect_with_retry(self.connect_host, self.port, "[ServerReceiver]")
            with conn:
                self._handle_connection(conn)
        else:
            print(f"[ServerReceiver] Listening on {self.listen_host}:{self.port}...")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.listen_host, self.port))
                s.listen(1)
                
                # Wait for connection (Blocking)
                # If we want to exit cleanly even without connection, we need timeout/select,
                # but for now, we assume mobile connects at least once.
                conn, addr = s.accept()
                
                with conn:
                    print(f"[ServerReceiver] Connected by {addr}")
                    self._handle_connection(conn)
                    
        print("[ServerReceiver] Triggering Server Shutdown...")
        self.shutdown_event.set() # Signal Main process to exit

    def _handle_connection(self, conn):
        # Handshake
        msg = recv_msg(conn)
        if isinstance(msg, tuple) and msg[0] == 'CONFIG':
            config = msg[1]
            print(f"[ServerReceiver] Handshake success. Policy: {config.scheduler_policy_name}")
            self.control_queue.put(msg)
        else:
            if msg is None:
                print("[ServerReceiver] Handshake failed: client disconnected before sending CONFIG.")
            elif msg == 'STOP':
                print("[ServerReceiver] Handshake aborted: client sent STOP before CONFIG.")
            elif isinstance(msg, tuple) and msg and msg[0] == 'ERROR':
                print(f"[ServerReceiver] Handshake aborted by client error: {msg[1]}")
            else:
                print(f"[ServerReceiver] Handshake failed or invalid format: {type(msg)}")
            return

        # Data Loop
        while True:
            msg = recv_msg(conn)

            if msg is None or msg == 'STOP':
                print("[ServerReceiver] Received STOP signal or Client disconnected.")
                break

            if hasattr(msg, 'arrival_time'):
                msg.arrival_time = time.time()

            if isinstance(msg, tuple):
                if msg[0] == 'TIME_SYNC':
                    self.control_queue.put(msg)
                else:
                    print(f"[ServerReceiver] Ignoring unknown control tuple: {msg[0]}")
            else:
                self.sched_queue.put(msg)


class ServerSender(multiprocessing.Process):
    """Sends InferenceResult back to Mobile."""

    def __init__(self, port, input_queue, listen_host="0.0.0.0", connect_host=None):
        super().__init__()
        self.port = port
        self.input_queue = input_queue
        self.listen_host = listen_host
        self.connect_host = connect_host

    def run(self):
        pending_result = None
        while True:
            conn = self._open_connection()
            try:
                with conn:
                    self._send_ready(conn)
                    pending_result = self._send_loop(conn, pending_result)
                    if pending_result == 'STOP':
                        return
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                print(f"[ServerSender] Connection lost ({exc}). Reconnecting...", flush=True)
                continue

    def _open_connection(self):
        if self.connect_host:
            return _connect_with_retry(self.connect_host, self.port, "[ServerSender]")

        print(f"[ServerSender] Waiting for Mobile connection on {self.listen_host}:{self.port}...", flush=True)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.listen_host, self.port))
            s.listen(1)
            conn, addr = s.accept()
            print(f"[ServerSender] Mobile connected from {addr}", flush=True)
            return conn

    def _send_ready(self, conn):
        send_msg(conn, (DOWNLINK_READY, time.time()))
        print(
            f"[ServerSender] Downlink ready sent "
            f"(local={conn.getsockname()}, peer={conn.getpeername()})",
            flush=True,
        )

    def _send_loop(self, conn, pending_result=None):
        result = pending_result
        while True:
            if result is None:
                result = self.input_queue.get()

            if result == 'STOP':
                return 'STOP'
            try:
                send_msg(conn, result)
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                print(f"[ServerSender] Send failed ({exc}). Will retry current result.", flush=True)
                return result
            result = None
