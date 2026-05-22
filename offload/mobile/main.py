import argparse
import sys
import os
import multiprocessing
import time
import json

# Add current directory to path
sys.path.append(os.getcwd())

from offload.mobile.modules import MobileSender, MobileReceiver
from offload.mobile.source import SourceModule
from offload.common import ExperimentConfig
from offload.common.config_overrides import apply_config_overrides

def run_mobile(
    server_ip,
    recv_port,
    send_port,
    data_root,
    config_path,
    num_request=None,
    num_warmup=1,
    reverse_connect=False,
    listen_host="0.0.0.0",
    config_overrides=None,
):
    print(f"=== Starting AppCorr Mobile Client ===")
    if reverse_connect:
        print(f"[*] Reverse Connect: listening on {listen_host}")
        print(f"[*] Data Uplink Port: {recv_port}")
        print(f"[*] Result Downlink Port: {send_port}")
    else:
        print(f"[*] Target Server: {server_ip}")
    print(f"[*] Dataset Root: {data_root}")
    print(f"[*] Config Path: {config_path}")
    print(f"[*] Warm-up Requests: {num_warmup}")
    if num_request is not None:
        print(f"[*] Num Requests: {num_request}")

    # Load Configuration
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    apply_config_overrides(config_data, config_overrides)
    
    # Check and convert types (Lists to Tuples for shapes)
    if 'image_shape' in config_data:
        config_data['image_shape'] = tuple(config_data['image_shape'])
    if 'patch_size' in config_data:
        config_data['patch_size'] = tuple(config_data['patch_size'])

    # Create ExperimentConfig
    config = ExperimentConfig(**config_data)

    print(f"[*] Batch Size (Images): {config.batch_size}")
    if config_overrides:
        print(f"[*] Config Overrides: {config_overrides}")

    # IPC Queues
    send_queue = multiprocessing.Queue()
    feedback_queue = multiprocessing.Queue()

    # Initialize processes
    sender = MobileSender(
        server_ip,
        recv_port,
        send_queue,
        listen=reverse_connect,
        listen_host=listen_host,
    )
    receiver = MobileReceiver(
        server_ip,
        send_port,
        feedback_queue,
        listen=reverse_connect,
        listen_host=listen_host,
    )
    source = SourceModule(
        send_queue,
        feedback_queue,
        config,
        data_root,
        config.batch_size,
        num_request,
        num_warmup,
    )

    procs = [sender, receiver, source]

    try:
        for p in procs:
            p.start()
        
        source.join()
        print("[Main] Source module finished.")
        sender.join(timeout=5)
        
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
    parser.add_argument("--recv-port", type=int, default=39990, help="Uplink port")
    parser.add_argument("--send-port", type=int, default=39991, help="Downlink port")
    parser.add_argument("--listen-host", type=str, default="0.0.0.0", help="Host/IP to bind when --reverse-connect is enabled")
    parser.add_argument("-rc", "--reverse-connect", action="store_true", help="Listen for the server instead of connecting to it")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset root (overrides config)")
    parser.add_argument("--config", type=str, default="offload/config/sequential.json", help="Path to Config JSON")
    parser.add_argument("-nr", "--num-request", type=int, default=None, help="Run only N requests; omit to run all")
    parser.add_argument("-nw", "--num-warmup", type=int, default=1, help="Run N warm-up requests before measurement")
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value using dot paths, e.g. --set appcorr_kwargs.token_keep_thres=0.0002",
    )
    
    args = parser.parse_args()
    if args.num_request is not None and args.num_request <= 0:
        parser.error("--num-request must be a positive integer")
    if args.num_warmup < 0:
        parser.error("--num-warmup must be zero or a positive integer")
    
    # Resolve data_root: CLI arg > config's dataset_kwargs > default
    config_data_root = args.data
    if config_data_root is None:
        with open(args.config, 'r') as f:
            _cfg = json.load(f)
        apply_config_overrides(_cfg, args.config_overrides)
        config_data_root = _cfg.get("dataset_kwargs", {}).get("data_root")
        if config_data_root is None and _cfg.get("dataset_name", "imagenet-1k") == "imagenet-1k":
            config_data_root = "~/data/imagenet_val"

    run_mobile(
        args.ip,
        args.recv_port,
        args.send_port,
        config_data_root,
        args.config,
        args.num_request,
        args.num_warmup,
        reverse_connect=args.reverse_connect,
        listen_host=args.listen_host,
        config_overrides=args.config_overrides,
    )
