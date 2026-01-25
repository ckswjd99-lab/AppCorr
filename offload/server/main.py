import argparse
import sys
import os
import multiprocessing
import time

sys.path.append(os.getcwd())

from offload.server.modules import ServerReceiver, ServerSender
from offload.server.scheduler import SchedulerModule
from offload.server.worker import WorkerModule

def run_server(recv_port, send_port):
    print(f"=== Starting AppCorr Server ===")
    print(f"[*] Listening for Data on Port: {recv_port}")
    print(f"[*] Listening for Results on Port: {send_port}")
    
    # IPC Queues
    sched_q = multiprocessing.Queue()
    worker_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    control_q = multiprocessing.Queue()
    
    # [FIX] Shutdown Event
    shutdown_event = multiprocessing.Event()

    # Pass shutdown_event to Receiver
    receiver = ServerReceiver(recv_port, sched_q, control_q, shutdown_event)
    scheduler = SchedulerModule(sched_q, worker_q, control_q)
    worker = WorkerModule(worker_q, result_q)
    sender = ServerSender(send_port, result_q)
    
    procs = [receiver, scheduler, worker, sender]

    try:
        for p in procs:
            p.start()
        
        # [FIX] Wait until shutdown signal is received
        while not shutdown_event.is_set():
            time.sleep(1)
            
        print("\n[Main] Shutdown signal received. Cleaning up...")
            
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by User...")
        
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join()
        print("Server Stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AppCorr Server")
    parser.add_argument("--recv-port", type=int, default=9998)
    parser.add_argument("--send-port", type=int, default=9999)
    
    args = parser.parse_args()
    run_server(args.recv_port, args.send_port)