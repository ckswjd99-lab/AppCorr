import multiprocessing
import time
import torch
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from dataclasses import asdict

from offload.common import ExperimentConfig
from offload.policies import get_transmission

import os
import json
import datetime

def perform_time_sync(output_queue, feedback_queue, rounds=10):
    """
    Perform multiple rounds of ping-pong to estimate clock offset.
    Offset = ServerTime - LocalTime
    """
    print(f"[Source] Synchronizing time with server ({rounds} rounds)...")
    
    # Warm-up round (ignore result)
    print("[Source] Sending warm-up ping...")
    output_queue.put(('TIME_SYNC', None))
    feedback_queue.get()
    
    offsets = []
    
    for _ in range(rounds):
        t1 = time.time()
        output_queue.put(('TIME_SYNC', None))
        
        t_server = feedback_queue.get()
        t2 = time.time()
        
        rtt = t2 - t1
        # Assumes symmetric network delay
        estimated_server_time_at_t1 = t_server - (rtt / 2)
        offset = estimated_server_time_at_t1 - t1
        offsets.append(offset)
        
    avg_offset = sum(offsets) / len(offsets)
    print(f"[Source] Time Sync Complete. Avg Offset: {avg_offset*1000:.2f} ms")
    return avg_offset

def load_imagenet1k_val(root, image_size=256, batch_size=32, num_workers=4):
    """Load ImageNet with specified batch size."""
    if image_size == 256:
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
    else:
        val_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    val_dataset = datasets.ImageFolder(root=root, transform=val_transforms)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False 
    )
    return val_loader

class SourceModule(multiprocessing.Process):
    """
    Experiment Loop: Request Batch -> Wait Response -> Calculate Metrics.
    Handles partial batches via padding and tracks data transfer size.
    """

    def __init__(self, output_queue, feedback_queue, config: ExperimentConfig, data_root: str, loader_batch_size: int):
        super().__init__()
        self.output_queue = output_queue
        self.feedback_queue = feedback_queue
        self.config = config
        self.data_root = data_root
        self.loader_batch_size = loader_batch_size

    def run(self):
        print(f"[Source] Loading ImageNet from {self.data_root} (Loader Batch: {self.loader_batch_size})...")
        try:
            # Assuming load_imagenet1k_val is defined above in the file
            loader = load_imagenet1k_val(self.data_root, batch_size=self.loader_batch_size)
        except Exception as e:
            print(f"[Source] Failed to load ImageNet: {e}")
            return

        # Generate Timestamp: YYYYMMDD_HHMMSS
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config.exp_id is None:
            self.config.exp_id = f"exp_{now_str}"
        else:
            self.config.exp_id = f"{self.config.exp_id}_{now_str}"

        # Setup Logging
        log_dir = os.path.join("logs", self.config.exp_id)
        os.makedirs(log_dir, exist_ok=True)
        events_log_path = os.path.join(log_dir, "events.jsonl")
        summary_log_path = os.path.join(log_dir, "summary.json")
        
        print(f"[Source] Logs will be saved to {log_dir}")
        events_file = open(events_log_path, "w")

        # Handshake
        self.output_queue.put(('CONFIG', self.config))
        time.sleep(1) # Wait for worker to load model

        # Time Synchronization
        time_offset = perform_time_sync(self.output_queue, self.feedback_queue)


        policy = get_transmission(self.config.transmission_policy_name)
        
        # Initialize metrics
        total_top1 = 0
        total_top5 = 0
        total_samples = 0
        total_bytes = 0 
        total_latency = 0.0 
        
        # Event Statistics Accumulator
        # { type: { count, sum, min, max } }
        event_stats_accumulator = {}

        # Constants for padding
        SERVER_BATCH_SIZE = self.config.batch_size
        IMG_H, IMG_W, IMG_C = self.config.image_shape

        print("[Source] Starting Batch Evaluation Loop...")

        pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
        for batch_idx, (images, labels) in pbar:
            curr_bs = images.size(0)
            
            # Send Phase
            t_load_start = time.time()
            # Prepare Full Batch Container (Pad with zeros)
            full_batch_np = np.zeros((SERVER_BATCH_SIZE, IMG_H, IMG_W, IMG_C), dtype=np.uint8)
            
            real_imgs_np = (images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            full_batch_np[:curr_bs] = real_imgs_np
            t_load_end = time.time()
            
            # Encode data
            t_encode_start = time.time()
            all_patches = policy.encode(full_batch_np, self.config)
            t_encode_end = time.time()
            
            batch_bytes = sum(len(p.data) for p in all_patches)
            total_bytes += batch_bytes
            batch_kb = batch_bytes / 1024.0

            t_send_start = time.time()
            
            # Send Patches
            for p in all_patches:
                self.output_queue.put(p)
            
            t_send_end = time.time()
            
            # Wait for Response
            result = self.feedback_queue.get()
            
            # Calculate Metrics
            t_result_recv = time.time()
            
            # Server Events
            server_events = result.server_events
            # Adjust timestamps
            for e in server_events:
                e['timestamp'] = e['start'] - time_offset # Map to Mobile Time
                e['duration'] = e['end'] - e['start']
                del e['start'], e['end']
            
            # Prepare Local Events
            local_events = [
                {'type': 'MOBILE_LOAD', 'timestamp': t_load_start, 'duration': t_load_end - t_load_start},
                {'type': 'MOBILE_ENCODE', 'timestamp': t_encode_start, 'duration': t_encode_end - t_encode_start},
                {'type': 'MOBILE_SEND', 'timestamp': t_send_start, 'duration': t_send_end - t_send_start},
                {'type': 'MOBILE_RECEIVE', 'timestamp': t_result_recv, 'duration': 0}
            ]
            
            # Combine
            all_events = local_events + server_events
            all_events.sort(key=lambda x: x.get('timestamp', 0))
            
            # Group Stats Calculation
            group_stats = {}
            for p in all_patches:
                gid = p.group_id
                if gid not in group_stats:
                    group_stats[gid] = {'count': 0, 'bytes': 0}
                group_stats[gid]['count'] += 1
                group_stats[gid]['bytes'] += len(p.data)

            # Aggregate Event Stats (Per Request)
            request_latency_map = {}
            for event in all_events:
                etype = event['type']
                dur_ms = event.get('duration', 0) * 1000.0 # Convert to ms
                
                if etype not in request_latency_map:
                    request_latency_map[etype] = 0.0
                request_latency_map[etype] += dur_ms
                
            # Update Global Accumulator with Per-Request Total
            for etype, total_dur_ms in request_latency_map.items():
                if etype not in event_stats_accumulator:
                    event_stats_accumulator[etype] = {'count': 0, 'sum': 0.0, 'min': float('inf'), 'max': float('-inf')}
                
                stats = event_stats_accumulator[etype]
                stats['count'] += 1
                stats['sum'] += total_dur_ms
                stats['min'] = min(stats['min'], total_dur_ms)
                stats['max'] = max(stats['max'], total_dur_ms)

            # Log Line
            log_entry = {
                'req_id': batch_idx,
                'acc1': 0, # Calc below
                'acc5': 0,
                'bytes': batch_bytes,
                'group_stats': group_stats,
                'events': all_events,
                'labels': labels.tolist()
            }
            
            # Calculate metrics
            valid_preds = result.output[:curr_bs]
            valid_labels = labels.tolist()
            
            latency = time.time() - result.timestamp # Note: result.timestamp is server time.
            # We should use local measurements.
            latency = t_result_recv - t_send_start # End-to-End approximation

            # Calculate accuracy
            batch_top1 = 0
            batch_top5 = 0
            for i in range(curr_bs):
                label = valid_labels[i]
                preds = valid_preds[i]
                if not preds: continue # skip empty results logic
                
                if preds[0] == label:
                    batch_top1 += 1
                if label in preds:
                    batch_top5 += 1

            # Update globals
            total_top1 += batch_top1
            total_top5 += batch_top5
            total_samples += curr_bs
            total_latency += latency
            
            acc1 = total_top1 / total_samples * 100
            acc5 = total_top5 / total_samples * 100
            
            # Update Log Entry
            log_entry['acc1'] = batch_top1 / curr_bs * 100
            log_entry['acc5'] = batch_top5 / curr_bs * 100
            log_entry['latency'] = latency
            events_file.write(json.dumps(log_entry) + "\n")
            events_file.flush()
            
            pbar.set_description(f"Acc@1: {acc1:.2f}% | Acc@5: {acc5:.2f}% | Avg. Transfer: {total_bytes/1024/total_samples:.2f} KB/image")
            
            # if (batch_idx+1) == 10:
            #     break # TEMP

        print(f"[Source] Final | Top-1: {acc1:.2f}% | Top-5: {acc5:.2f}% | Avg. Transfer: {total_bytes/1024/total_samples:.2f} KB/image")
        
        # Calculate Latency Breakdown
        latency_breakdown = {}
        print("\n=== Latency Breakdown (ms) ===")
        print(f"{'Event Type':<25} | {'Avg':<8} | {'Min':<8} | {'Max':<8} | {'Count':<6}")
        print("-" * 65)
        
        sorted_stats = sorted(event_stats_accumulator.items(), key=lambda x: x[1]['sum'], reverse=True)
        
        for etype, stats in sorted_stats:
            avg = stats['sum'] / stats['count'] if stats['count'] > 0 else 0
            latency_breakdown[etype] = {
                'avg_ms': avg,
                'min_ms': stats['min'],
                'max_ms': stats['max'],
                'count': stats['count']
            }
            print(f"{etype:<25} | {avg:<8.2f} | {stats['min']:<8.2f} | {stats['max']:<8.2f} | {stats['count']:<6}")
        print("=" * 65 + "\n")
        
        # Write Summary
        summary = {
            'exp_id': self.config.exp_id,
            'total_samples': total_samples,
            'top1_acc': acc1,
            'top5_acc': acc5,
            'avg_bytes_per_sample': total_bytes / total_samples,
            'avg_latency_per_batch': total_latency / (batch_idx + 1),
            'time_offset_ms': time_offset * 1000,
            'latency_breakdown': latency_breakdown,
            'config': asdict(self.config)
        }
        with open(summary_log_path, "w") as f:
            json.dump(summary, f, indent=4)
            
        events_file.close()
        self.output_queue.put('STOP')
            
