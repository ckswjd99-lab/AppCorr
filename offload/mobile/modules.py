import multiprocessing
import socket
import time
from offload.common import send_msg, recv_msg, InferenceResult

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


def _accept_one_connection(host, port, label):
    print(f"{label} Listening on {host}:{port}...", flush=True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(1)
        conn, addr = listener.accept()
        print(
            f"{label} Server connected from {addr} "
            f"(local={conn.getsockname()}, peer={conn.getpeername()})",
            flush=True,
        )
        return conn


class MobileSender(multiprocessing.Process):
    """Connects to Server Receiver and streams Config/Patches."""

    def __init__(self, server_ip, server_port, input_queue, listen=False, listen_host="0.0.0.0"):
        super().__init__()
        self.server_ip = server_ip
        self.server_port = server_port
        self.input_queue = input_queue
        self.listen = listen
        self.listen_host = listen_host

    def run(self):
        if self.listen:
            sock = _accept_one_connection(self.listen_host, self.server_port, "[MobileSender]")
        else:
            sock = _connect_with_retry(self.server_ip, self.server_port, "[MobileSender]")

        with sock:
            while True:
                item = self.input_queue.get()

                if item == 'STOP':
                    send_msg(sock, 'STOP')
                    break

                send_msg(sock, item)
        print("[MobileSender] Stopped.", flush=True)


class MobileReceiver(multiprocessing.Process):
    """Connects to Server Sender and logs InferenceResults."""

    def __init__(self, server_ip, server_port, feedback_queue, listen=False, listen_host="0.0.0.0"):
        super().__init__()
        self.server_ip = server_ip
        self.server_port = server_port
        self.feedback_queue = feedback_queue
        self.listen = listen
        self.listen_host = listen_host

    def run(self):
        while True:
            if self.listen:
                sock = _accept_one_connection(self.listen_host, self.server_port, "[MobileReceiver]")
            else:
                sock = _connect_with_retry(self.server_ip, self.server_port, "[MobileReceiver]")

            with sock:
                while True:
                    result = recv_msg(sock)
                    if isinstance(result, InferenceResult) or isinstance(result, float):
                        self.feedback_queue.put(result)
                    elif isinstance(result, tuple) and result and result[0] == DOWNLINK_READY:
                        print(
                            f"[MobileReceiver] Downlink ready from server "
                            f"(server_time={result[1]:.6f})",
                            flush=True,
                        )
                    elif result is None:
                        print("[MobileReceiver] Connection closed. Waiting for reconnect...", flush=True)
                        break
                    else:
                        print(f"[MobileReceiver] Ignoring unexpected result type: {type(result)}", flush=True)
