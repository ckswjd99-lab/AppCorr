import argparse
import sys
import os
import multiprocessing
import time

# Add current directory to path
sys.path.append(os.getcwd())

from offload.mobile.modules import MobileSender, MobileReceiver
from offload.mobile.source import SourceModule
from offload.common import ExperimentConfig

def run_mobile(server_ip, recv_port, send_port, data_root, img_batch_size):
    print(f"=== Starting AppCorr Mobile Client ===")
    print(f"[*] Target Server: {server_ip}")
    print(f"[*] ImageNet Root: {data_root}")
    print(f"[*] Batch Size (Images): {img_batch_size}")

    # Configure experiment settings
    # 256 patches = 1 Image (16x16 patch size, 256x256 image size)
    config = ExperimentConfig(
        batch_size=img_batch_size,    
        patches_per_image=256,
        image_shape=(256, 256, 3),
        patch_size=(16, 16),
        scheduler_policy_name="BatchCountBased",
        transmission_policy_name="Raw"
    )

    # IPC Queues
    send_queue = multiprocessing.Queue()
    feedback_queue = multiprocessing.Queue()

    # Initialize processes
    sender = MobileSender(server_ip, recv_port, send_queue)
    receiver = MobileReceiver(server_ip, send_port, feedback_queue)
    source = SourceModule(send_queue, feedback_queue, config, data_root, img_batch_size)

    procs = [sender, receiver, source]

    try:
        for p in procs:
            p.start()
        
        source.join()
        print("[Main] Source module finished.")
        
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
    parser.add_argument("--recv-port", type=int, default=9998, help="Uplink port")
    parser.add_argument("--send-port", type=int, default=9999, help="Downlink port")
    parser.add_argument("--data", type=str, default="~/data/imagenet_val", help="Path to ImageNet")
    parser.add_argument("--batch-size", type=int, default=32, help="DataLoader batch size (images)")
    
    args = parser.parse_args()
    
    run_mobile(args.ip, args.recv_port, args.send_port, args.data, args.batch_size)