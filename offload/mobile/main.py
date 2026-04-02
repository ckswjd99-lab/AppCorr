import argparse
import sys
import os
import multiprocessing
import time
import json

# Add current directory to path
sys.path.append(os.getcwd())

from offload.mobile.modules import MobileSender, MobileReceiver
from offload.mobile.hint import MobileHintWorker, mobile_hint_enabled
from offload.mobile.source import SourceModule
from offload.common import ExperimentConfig
from offload.common.protocol import normalize_appcorr_kwargs

def run_mobile(server_ip, recv_port, send_port, data_root, config_path):
    print(f"=== Starting AppCorr Mobile Client ===")
    print(f"[*] Target Server: {server_ip}")
    print(f"[*] ImageNet Root: {data_root}")
    print(f"[*] Config Path: {config_path}")

    # Load Configuration
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Check and convert types (Lists to Tuples for shapes)
    if 'image_shape' in config_data:
        config_data['image_shape'] = tuple(config_data['image_shape'])
    if 'patch_size' in config_data:
        config_data['patch_size'] = tuple(config_data['patch_size'])

    # Create ExperimentConfig
    config = ExperimentConfig(**config_data)

    print(f"[*] Batch Size (Images): {config.batch_size}")

    # IPC Queues
    send_queue = multiprocessing.Queue()
    feedback_queue = multiprocessing.Queue()
    hint_input_queue = multiprocessing.Queue() if mobile_hint_enabled(config) else None
    hint_event_queue = multiprocessing.Queue() if hint_input_queue is not None else None

    # Initialize processes
    sender = MobileSender(server_ip, recv_port, send_queue)
    receiver = MobileReceiver(server_ip, send_port, feedback_queue)
    source = SourceModule(
        send_queue,
        feedback_queue,
        config,
        data_root,
        config.batch_size,
        hint_input_queue=hint_input_queue,
        hint_event_queue=hint_event_queue,
    )
    hint_worker = None
    if hint_input_queue is not None:
        appcorr_options = normalize_appcorr_kwargs(config.appcorr_kwargs)
        hint_weights = appcorr_options.get("mobile_hint_model_weights")
        if hint_weights is None:
            raise ValueError("mobile hint enabled but mobile_hint_model_weights is not configured")
        hint_worker = MobileHintWorker(hint_input_queue, send_queue, hint_event_queue, config)

    procs = [sender, receiver, source] + ([hint_worker] if hint_worker is not None else [])

    try:
        for p in procs:
            p.start()
        
        source.join()
        print("[Main] Source module finished.")
        if hint_input_queue is not None:
            hint_input_queue.put('STOP')
            if hint_worker is not None:
                hint_worker.join()
        send_queue.put('STOP')
        sender.join()
        receiver.join(timeout=5)
        
    except KeyboardInterrupt:
        print("\n[Main] Stopping client...")
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join()
        print("[Main] Client stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AppCorr Mobile Client")
    
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Target Server IP")
    parser.add_argument("--recv-port", type=int, default=39998, help="Uplink port")
    parser.add_argument("--send-port", type=int, default=39999, help="Downlink port")
    parser.add_argument("--data", type=str, default="~/data/imagenet_val", help="Path to ImageNet")
    parser.add_argument("--config", type=str, default="offload/config/sequential.json", help="Path to Config JSON")
    
    args = parser.parse_args()
    
    run_mobile(args.ip, args.recv_port, args.send_port, args.data, args.config)
