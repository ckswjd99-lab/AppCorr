import socket
import struct
import pickle
from typing import Any, Optional

def send_msg(sock: socket.socket, data: Any) -> None:
    """Send pickled data with a length prefix."""
    try:
        serialized = pickle.dumps(data)
        # Pack length as 4-byte big-endian unsigned int
        length_header = struct.pack('>I', len(serialized))
        sock.sendall(length_header + serialized)
    except Exception as e:
        print(f"[Network] Send Error: {e}")
        raise e

def recv_msg(sock: socket.socket) -> Optional[Any]:
    """Receive length-prefixed data and unpickle it."""
    try:
        # Read length (4 bytes)
        raw_len = _recv_all(sock, 4)
        if not raw_len:
            return None
        
        msg_len = struct.unpack('>I', raw_len)[0]
        
        # Read payload
        raw_data = _recv_all(sock, msg_len)
        if not raw_data:
            return None
            
        return pickle.loads(raw_data)
        
    except (ConnectionResetError, BrokenPipeError):
        return None
    except Exception as e:
        print(f"[Network] Recv Error: {e}")
        return None

def _recv_all(sock: socket.socket, n: int) -> Optional[bytes]:
    """Helper to receive exactly n bytes."""
    data = b''
    while len(data) < n:
        try:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        except OSError:
            return None
    return data