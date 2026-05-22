import argparse
import sys
import os
import multiprocessing
import time

sys.path.append(os.getcwd())

from offload.server.modules import ServerReceiver, ServerSender
from offload.server.scheduler import SchedulerModule
from offload.server.worker import WorkerModule

def run_server(
    recv_port,
    send_port,
    reverse_connect=False,
    mobile_ip="127.0.0.1",
    listen_host="0.0.0.0",
):
    print(f"=== Starting AppCorr Server ===")
    if reverse_connect:
        print(f"[*] Reverse Connect: connecting to mobile at {mobile_ip}")
        print(f"[*] Data Uplink Port: {recv_port}")
        print(f"[*] Result Downlink Port: {send_port}")
    else:
        print(f"[*] Listening for Data on {listen_host}:{recv_port}")
        print(f"[*] Listening for Results on {listen_host}:{send_port}")
    
    # IPC Queues
    sched_q = multiprocessing.Queue()
    worker_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    control_q = multiprocessing.Queue()
    feedback_q = multiprocessing.Queue()
    
    # Shutdown Event
    shutdown_event = multiprocessing.Event()

    connect_host = mobile_ip if reverse_connect else None

    # Pass shutdown_event to Receiver
    receiver = ServerReceiver(
        recv_port,
        sched_q,
        control_q,
        shutdown_event,
        listen_host=listen_host,
        connect_host=connect_host,
    )
    scheduler = SchedulerModule(sched_q, worker_q, control_q, feedback_q)
    worker = WorkerModule(worker_q, result_q, feedback_q)
    sender = ServerSender(
        send_port,
        result_q,
        listen_host=listen_host,
        connect_host=connect_host,
    )
    
    procs = [receiver, scheduler, worker, sender]

    def request_graceful_shutdown():
        """Let child processes exit through their normal loops so profilers flush."""
        try:
            control_q.put(('STOP', None))
        except Exception:
            pass
        try:
            worker_q.put('STOP')
        except Exception:
            pass
        try:
            result_q.put('STOP')
        except Exception:
            pass

    try:
        for p in procs:
            p.start()
        
        while not shutdown_event.is_set():
            time.sleep(1)
            
        print("\n[Main] Shutdown signal received. Cleaning up...")
            
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by User...")
        
    finally:
        request_graceful_shutdown()
        for p in procs:
            if p.is_alive():
                p.join(timeout=10)
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join()
        print("Server Stopped.")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    parser = argparse.ArgumentParser(description="AppCorr Server")
    parser.add_argument("--recv-port", type=int, default=39990)
    parser.add_argument("--send-port", type=int, default=39991)
    parser.add_argument("--listen-host", type=str, default="0.0.0.0", help="Host/IP to bind in default listen mode")
    parser.add_argument("-rc", "--reverse-connect", action="store_true", help="Connect to a mobile listener instead of listening on this server")
    parser.add_argument("--mobile-ip", type=str, default="127.0.0.1", help="Mobile IP to connect to when --reverse-connect is enabled")
    
    args = parser.parse_args()
    run_server(
        args.recv_port,
        args.send_port,
        reverse_connect=args.reverse_connect,
        mobile_ip=args.mobile_ip,
        listen_host=args.listen_host,
    )
